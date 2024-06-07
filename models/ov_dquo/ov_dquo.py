import copy
import math
import torch
from models.backbone.ov_backbone import build_backbone, build_classifier
from models.criterion.ov_criterion_pseudo import OVSetCriterion_Pseudo
from models.matcher.ov_matcher import build_ov_matcher
from models.transformer.ov_deformable_transformer import build_ov_deformable_transformer
from util import box_ops
from models.ov_dquo.ov_dn_components import dn_post_process, prepare_for_cdn_ov
from models.ov_dquo.ov_postprocess import OVPostProcess
from models.registry import MODULE_BUILD_FUNCS
import torch.nn as nn
from .utils import MLP
from util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)
import torch.nn.functional as F
from torchvision.ops import box_iou
from ..transformer.base import sample_feature_rn,sample_feature_vit

class OV_DQUO(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        classifier,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        two_stage_add_query_num=0,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        decoder_sa_type="sa",
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        args=None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        if dn_number > 0:
            self.label_enc = nn.Embedding(args.num_label_sampled+1, hidden_dim)
        else:
            self.label_enc = None
        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        # prepare input projection layers
        if num_feature_levels > 1:
            input_proj_list = []
            for _ in range(len(backbone.num_backbone_outs)):
                in_channels = args.in_channel[_]
                input_proj_list.append(
                        nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - len(backbone.num_backbone_outs)):
                input_proj_list.append(
                        nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            self.input_proj = nn.ModuleList(input_proj_list)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, 1)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]
        if decoder_sa_type == "ca_label":
            self.label_embedding = nn.Embedding(1, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self.classifier = classifier
        self.args = args
        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(
        self,
        samples: NestedTensor,
        categories,
        targets=None,
    ):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if self.training:
            assert self.args.pseudo_box != ""
            with torch.no_grad():
                if "RN" in self.args.backbone:
                    categories.append(self.args.wildcard) # add wildcard embed
                    text_feature = self.classifier(categories)
                else:
                    assert self.args.num_label_sampled > 0
                    text_feature=self.classifier[categories]
                    text_feature=torch.cat([text_feature,self.classifier[-1][None,:]]) # add wildcard embed
        else:
            if "RN" in self.args.backbone:
                text_feature=self.classifier(categories)
            else:
                assert self.args.pseudo_box != ""
                text_feature=self.classifier[:-1] # remove wildcard embed during ovlvis inference
        ori_clip_features, ori_clip_pos_embeds = self.backbone(samples)
        clip_features = [
            ori_clip_features[k] for k in ori_clip_features.keys() if k != "dense" and k != "layer4"# discard dense feature layer
        ] 
        clip_pos_embeds = [
            ori_clip_pos_embeds[k] for k in ori_clip_pos_embeds.keys() if k != "dense" and k != "layer4"
        ]
        srcs = []
        masks = []
        for l, feat in enumerate(clip_features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](clip_features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                clip_pos_embeds.append(pos_l)
        # for ov dn
        if self.dn_number > 0 and self.training:
            proj_text_feature = self.transformer.text_proj(text_feature)
            dn_query_label, dn_query_bbox, dn_attn_mask, dn_meta = prepare_for_cdn_ov(
                dn_args=(
                    targets,
                    self.args.dn_number,
                    self.args.dn_label_noise_ratio,
                    self.args.dn_box_noise_scale,
                ),
                training=self.training,
                num_queries=self.num_queries,
                num_classes=len(proj_text_feature),
                text_embbeding=proj_text_feature,
                label_enc_embbeding=self.label_enc,
            )
        else:
            dn_query_label = None
            dn_query_bbox = None
            dn_attn_mask = None
            dn_meta = None
        # end for ov dn
        (
            hs,
            reference,
            hs_enc,
            ref_enc,
            init_box_proposal,
            classes_,
        ) = self.transformer(
            srcs=srcs,
            masks=masks,
            refpoint_embed=dn_query_bbox,
            pos_embeds=clip_pos_embeds,
            tgt=dn_query_label,
            attn_mask=dn_attn_mask,
            raw_visual_feats=ori_clip_features,
            raw_text_feats=text_feature,
            targets=targets,
            backbone=self.backbone,
        )
        outputs_coord_list = []
        for _, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = dn_post_process(
                outputs_class,
                outputs_coord_list,
                dn_meta,
                self.aux_loss,
                self._set_aux_loss,
            )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord_list)

        out["proposal_classes"] = classes_
        if self.aux_loss:
            for aux in out["aux_outputs"]:
                aux["proposal_classes"] = classes_
        # for encoder output 
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out["interm_outputs"] = {
                "pred_logits": interm_class,
                "pred_boxes": interm_coord,
            }
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": init_box_proposal,
            }
            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (
                    layer_box_embed,
                    layer_class_embed,
                    layer_hs_enc,
                    layer_ref_enc,
                ) in enumerate(
                    zip(
                        self.enc_bbox_embed,
                        self.enc_class_embed,
                        hs_enc[:-1],
                        ref_enc[:-1],
                    )
                ):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = (
                        layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    )
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out["enc_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]
        out["dn_meta"] = dn_meta
        if not self.training:
            sample_box = outputs_coord_list[-1:]
            roi_feats = []
            for coord in sample_box:
                if "RN" in self.args.backbone:
                    src_feature = ori_clip_features["layer3"] # C4 in ResNet
                    sizes = [((1 - m[0].float()).sum(), (1 - m[:, 0].float()).sum()) for m in src_feature.decompose()[1]]
                    roi_feats.append(sample_feature_rn(
                                            sizes,
                                           coord,
                                           src_feature.tensors,
                                           self.args,
                                           self.backbone,
                                           extra_conv=True))
                else:
                    src_feature = ori_clip_features["dense"] # dense feature for ViT
                    sizes = [((1 - m[0].float()).sum(), (1 - m[:, 0].float()).sum()) for m in src_feature.decompose()[1]]
                    roi_feats.append(
                    sample_feature_vit(sizes,
                                        coord,
                                        src_feature.tensors)
                    )
            roi_features = roi_feats[-1]
            clip_outputs_class = roi_features @ text_feature.t()
            if self.args.analysis: #  for analysis
                out["sim_mat"] = clip_outputs_class  
                out["ori_pred_logits"] = outputs_class[-1]  
            clip_outputs_class = torch.cat(
                [clip_outputs_class, torch.zeros_like(clip_outputs_class[:, :, :1])],
                dim=-1,
            )
            final_outputs_class = (clip_outputs_class * self.args.eval_tau).softmax(
                dim=-1
            ) * (outputs_class[-1].sigmoid() ** self.args.objectness_alpha)
            final_outputs_class = final_outputs_class[:, :, :-1]
            final_outputs_class = inverse_sigmoid(final_outputs_class)
            out["pred_logits"] = final_outputs_class
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

@MODULE_BUILD_FUNCS.registe_with_name(module_name="ov_dquo")
def build_ov_dquo(args):
    device = torch.device(args.device)
    backbone = build_backbone(args) 
    transformer = build_ov_deformable_transformer(args)
    classifier = build_classifier(args)
    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = OV_DQUO(
        # backbone,
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        classifier=classifier,
        args=args,
    )


    # prepare weight dict
    weight_dict = {"loss_ce": args.cls_loss_coef, 
                   "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DN training
    if args.use_dn:
        weight_dict["loss_ce_dn"] = args.cls_loss_coef
        weight_dict["loss_bbox_dn"] = args.bbox_loss_coef
        weight_dict["loss_giou_dn"] = args.giou_loss_coef

    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in clean_weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != "no":
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 1.0 if not no_interm_box_loss else 0.0,
            "loss_giou": 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {
                k + f"_interm": v * interm_loss_coef * _coeff_weight_dict[k]
                for k, v in clean_weight_dict_wo_dn.items()
            }
        )
        weight_dict.update(interm_weight_dict)

    losses = ["labels", "boxes"]
    ov_matcher, vanilla_matcher = build_ov_matcher(args)
    criterion = OVSetCriterion_Pseudo(
        ov_matcher=ov_matcher,
        vanilla_matcher=vanilla_matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
    )
    criterion.to(device)
    postprocessors = {"bbox": OVPostProcess(args)}

    return model, criterion, postprocessors
