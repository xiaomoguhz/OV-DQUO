import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor
import torch.nn.functional as F
from models.clip.clip import _MODELS, _download, available_models, tokenize
from models.clip.model import Transformer
from models.clip.prompts import imagenet_templates
import os

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, out_indices: List):
        super().__init__()
        for _, parameter in backbone.named_parameters():
            parameter.requires_grad_(False)
        return_layers={}
        indices=[1,2,3]
        out_channels=[]
        out_strides=[]
        for indice in indices:
            if indice in out_indices:
                return_layers.update({f'layer{indice+1}':f'layer{indice+1}'})
                out_channels.append(num_channels//(4/indice))
                out_strides.append(indice*8)
        self.num_channels=out_channels
        self.strides=out_strides
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class Classifier(torch.nn.Module):
    def __init__(self, model_name: str, token_len=77,pretrained=None):
        super().__init__()
        self.cache = {}

        if pretrained:
            model_path=pretrained
        elif model_name.replace('CLIP_', '') in _MODELS:
            raw_model_name = model_name.replace('CLIP_', '')
            model_path = _download(_MODELS[raw_model_name], os.path.expanduser("~/.cache/clip"))
        else:
            raise RuntimeError(f"Model {model_name.replace('CLIP_', '')} not found; available models = {available_models()}")
        
        with open(model_path, "rb") as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu")
            state_dict = model.state_dict()
        embed_dim = state_dict["text_projection"].shape[1]
        # self.context_length = state_dict["positional_embedding"].shape[0]
        self.context_length = token_len
        self.vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(
                k.split(".")[2]
                for k in state_dict
                if k.startswith("transformer.resblocks")
            )
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim)
        )

        # load
        self.transformer.load_state_dict(
            {
                k.replace("transformer.", ""): v
                for k, v in state_dict.items()
                if k.startswith("transformer.")
            }
        )
        self.token_embedding.load_state_dict(
            {
                k.replace("token_embedding.", ""): v
                for k, v in state_dict.items()
                if "token_embedding" in k
            }
        )
        self.ln_final.load_state_dict(
            {
                k.replace("ln_final.", ""): v
                for k, v in state_dict.items()
                if "ln_final" in k
            }
        )
        self.positional_embedding.data = state_dict["positional_embedding"]
        self.text_projection.data = state_dict["text_projection"]

        for v in self.parameters():
            v.requires_grad_(False)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text):
        x = self.token_embedding(text)

        x = x + self.positional_embedding[: self.context_length]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(
            x.dtype
        )

        return x

    def forward_feature(self, category_list):
        templates = imagenet_templates
        texts = [
            template.format(cetegory)
            for cetegory in category_list
            for template in templates
        ]  # format with class
        texts = tokenize(texts, context_length=self.context_length, truncate=True).to(
            self.positional_embedding.device
        )
        class_embeddings = []
        cursor = 0
        step = 3000
        while cursor <= len(texts):
            class_embeddings.append(self.encode_text(texts[cursor : cursor + step]))
            cursor += step
        class_embeddings = torch.cat(class_embeddings)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embeddings = class_embeddings.unflatten(
            0, (len(category_list), len(templates))
        )
        class_embedding = class_embeddings.mean(dim=1)
        class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        return class_embedding

    def forward(self, category_list):
        new_category = [
            category for category in category_list if category not in self.cache
        ]
        with torch.no_grad():
            new_class_embedding = self.forward_feature(new_category)
            for category, feat in zip(new_category, new_class_embedding):
                self.cache[category] = feat.to("cpu")
        class_embedding = torch.stack(
            [self.cache[category] for category in category_list]
        ).to(self.positional_embedding.device)

        return class_embedding
