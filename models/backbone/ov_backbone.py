import json
from typing import Dict
from models.ov_dquo.position_encoding import build_position_encoding
import open_clip
from functools import partial
import os
import torch
from torch import nn
from mmcv.runner import BaseModule
from torch.nn import functional as F
from mmcv.utils.logging import print_log
from util.misc import NestedTensor
from .base import BackboneBase, FrozenBatchNorm2d, Classifier
from models.clip.clip import _MODELS, _download, available_models
from models.clip.model import ModifiedResNet

class EvaCLIPViT(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[6, 10, 14, 23]):
        super().__init__()
        self.vit_layers = out_indices
        self.out_indices=out_indices
        self.model_name = model_name
        self.pretrained = pretrained  # the pretrained .pt file
        clip_model = open_clip.create_model(model_name,
                                            pretrained="eva",
                                            cache_dir=pretrained)
        self.embed_dim = clip_model.embed_dim  # output dim
        self.width = width = clip_model.visual.embed_dim
        self.patch_size = clip_model.visual.patch_embed.patch_size[0]
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate2 = nn.Identity()
        self.interpolate3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual = clip_model.visual
    
    def init_weights(self):
        clip_model = open_clip.create_model(self.model_name,
                                            pretrained="eva",
                                            cache_dir=self.pretrained,
                                            device="cpu")
        print_log(self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        print(f"Set train mode for EVA: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        return self

    def forward(self, input):
        if isinstance(input,NestedTensor):
            x=input.tensors
        else:
            x=input
        detr_out: Dict[str, NestedTensor] = {}
        visual = self.visual
        bs, _, h, w = x.shape
        h = h // visual.patch_embed.patch_size[0]
        w = w // visual.patch_embed.patch_size[1]
        with torch.no_grad():
            x = visual.patch_embed(x)
            batch_size, seq_len, _ = x.size()
            cls_tokens = visual.cls_token.expand(batch_size, -1, -1)  
            x = torch.cat((cls_tokens, x), dim=1)
            if visual.pos_embed is not None:
                x = x + visual.rescale_positional_embedding(out_size=(h, w))
            x = visual.pos_drop(x)
            if os.getenv('RoPE') == '1':
                if visual.training and not isinstance(visual.patch_dropout, nn.Identity):
                    x, patch_indices_keep = visual.patch_dropout(x)
                    visual.rope.forward = partial(visual.rope.forward, patch_indices_keep=patch_indices_keep)
                else:
                    visual.rope.forward = partial(visual.rope.forward, patch_indices_keep=None)
                    x = visual.patch_dropout(x)
            else:
                x = visual.patch_dropout(x)

            rel_pos_bias = visual.rel_pos_bias() if visual.rel_pos_bias is not None else None

            outs = []
            for i, blk in enumerate(visual.blocks[:-1]):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if i in self.vit_layers:
                    outs.append(self._expand_x(x, h, w))
            x = visual.blocks[-1].forward_without_attn(x)
            if (len(visual.blocks) - 1) in self.vit_layers:
                outs.append(self._expand_x(x, h, w))
            # 768
            x = x[:, 1:]
            x = visual.norm(x)
            x = visual.head(x)
            assert visual.fc_norm is None
            x = F.normalize(x, dim=-1) 
            feature_map = x.view(bs, h, w, -1).permute(0, 3, 1, 2)
            
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())
            if isinstance(input,NestedTensor):
                m = input.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=outs[idx].shape[-2:]).to(torch.bool)[0]
                detr_out[f"layer{idx}"]= NestedTensor(outs[idx], mask)

        m = input.mask
        detr_out[f"dense"]= NestedTensor(feature_map, 
                                                        F.interpolate(m[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0])
        return detr_out

    def _expand_x(self, x, h, w):
        # x: bs q c
        x = x[:, 1:].permute(0, 2, 1).contiguous()
        x = x.view(-1, self.width, h, w)
        return x


class ResNetCLIP(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,model_name,
                out_indices,
                region_prompt_path,
                pretrained=None):
        self.out_indices=out_indices
        if pretrained:
            model_path=pretrained
        elif model_name.replace('CLIP_', '') in _MODELS:
            raw_model_name = model_name.replace('CLIP_', '')
            model_path = _download(_MODELS[raw_model_name], os.path.expanduser("~/.cache/clip"))
        else:
            raise RuntimeError(f"Model {model_name.replace('CLIP_', '')} not found; available models = {available_models()}")
        with open(model_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu")
            state_dict = model.state_dict()
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_dim, embed_dim = state_dict['visual.attnpool.c_proj.weight'].shape
        vision_heads = vision_width * 32 // 64
        image_resolution = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5) * 32
        backbone = ModifiedResNet(layers=vision_layers, output_dim=output_dim, heads=vision_heads, input_resolution=image_resolution, 
                                    width=vision_width, bn=FrozenBatchNorm2d)
        new_state_dict = dict()
        num_channels = embed_dim
        new_state_dict.update({k.replace('visual.', ''): v for k, v in state_dict.items() if k.startswith('visual.')})
        num_channels = state_dict['visual.attnpool.c_proj.weight'].size(0)
        if region_prompt_path:
            region_prompt = torch.load(region_prompt_path, map_location='cpu')
            new_state_dict.update(region_prompt)
        backbone.load_state_dict(new_state_dict)
        super().__init__(backbone, num_channels, out_indices)
        self.attn_pool = backbone.attnpool
        self.layer4 = backbone.layer4


def build_backbone(args):
    if "EVA" in args.backbone:
        backbone=EvaCLIPViT(
                    model_name=args.backbone,
                    out_indices=args.backbone_out_indice,
                    pretrained=args.pretrained,
                    )
        backbone.init_weights()
    elif "RN" in args.backbone:
        backbone=ResNetCLIP(
                    model_name=args.backbone,
                    out_indices=args.backbone_out_indice,
                    region_prompt_path=args.region_prompt_path,
                    pretrained=args.pretrained,
                    )
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}") 
    position_embedding = build_position_encoding(args)
    model = Joiner(backbone, position_embedding)
    return model


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_backbone_outs=backbone.out_indices
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        pos = dict()
        for name, x in xs.items():
            # position encoding
            pos[name] = self[1](x).to(x.tensors.dtype)
        return xs, pos

def build_classifier(args):
    if "EVA" in args.backbone:
        class_embed = torch.load(args.text_embed)
        all_classes = json.load(open(args.all_classes))
        all_embed = [class_embed[name] for name in all_classes]
        all_embed = torch.stack(all_embed, dim=0).contiguous()
        if args.pseudo_box:
            assert args.object_embbed!=""
            object_embed_dict = torch.load(args.object_embbed)
            object_embed = torch.stack([object_embed_dict[name] for name in object_embed_dict], dim=0).contiguous()
            all_embed=torch.cat([all_embed,object_embed])
        all_embed = F.normalize(all_embed, p=2, dim=1).to(args.device)
        return all_embed
    elif "RN" in args.backbone:
        classifier = Classifier(
        model_name=args.backbone,
        token_len=args.text_len,
        pretrained=args.pretrained)
        return classifier
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}") 

