import torch
from models.ov_dquo.utils import MLP, gen_encoder_output_proposals
from models.transformer.deformable_transformer import DeformableTransformer
from .base import sample_feature_vit, sample_feature_rn, get_match_pseudo_idx, COCO_INDEX, LVIS_INDEX
import torch.nn.functional as F

class OVDeformableTransformer(DeformableTransformer):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.text_proj = MLP(args.text_dim, 128, args.hidden_dim, 2)
        self.args = args
    def forward(
        self,
        srcs,
        masks,
        refpoint_embed,
        pos_embeds,
        tgt,
        attn_mask=None,
        raw_visual_feats=None,
        raw_text_feats=None,
        targets=None,
        backbone=None
    ):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  #  bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  #  bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(
            lvl_pos_embed_flatten, 1
        )  #  bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None
        #########################################################
        #  Begin Encoder
        #########################################################
        memory, _, _ = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )
        #########################################################
        #  End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        if self.two_stage_type == "standard":
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, input_hw) 
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            if self.two_stage_pat_embed > 0:  
                bs, nhw, _ = output_memory.shape
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = (self.enc_out_bbox_embed(output_memory) + output_proposals)  
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected,1,topk_proposals.unsqueeze(-1).repeat(1, 1, 4),)  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  
            # gather tgt
            tgt_undetach = torch.gather(output_memory,1,topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),)
            if self.embed_init_tgt:
                tgt_ = (self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)) 
            else:
                tgt_ = tgt_undetach.detach()
        elif self.two_stage_type == "no":
            tgt_ = (self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1))  
            refpoint_embed_ = (self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1))  
            init_box_proposal = refpoint_embed_.sigmoid()
        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        
        #########################################################
        #  begin text queries assignment
        #########################################################
        classes_, query_features, query = self.text_query_assign(
            region_proposals=refpoint_embed_,
            raw_visual_feats=raw_visual_feats,
            raw_text_feats=raw_text_feats,
            targets=targets,
            backbone=backbone,
        )
        if refpoint_embed is not None:
            refpoint_embed = torch.cat([refpoint_embed, query], dim=1)
            tgt = torch.cat(
                [tgt, tgt_ + query_features], dim=1
            )  
        else:
            refpoint_embed, tgt = query, tgt_ + query_features
        #########################################################
        #  End text queries assignment
        #  refpoint_embed: 
        #  tgt: 
        #########################################################
    
        #########################################################
        #  Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
        )

        #########################################################
        #  End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        #  Begin postprocess
        #########################################################
        if self.two_stage_type == "standard":
            if self.two_stage_keep_all_tokens:  # false
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals
            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        #  End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal, classes_

    def text_query_assign(self,region_proposals,
                       raw_visual_feats,
                       raw_text_feats,
                       targets,
                       backbone,
                       ):
        if "RN" in self.args.backbone:
            src_feature = raw_visual_feats["layer4"] # C5 in ResNet
        else:
            src_feature = raw_visual_feats["dense"] # dense feature for ViT
        text_feature = raw_text_feats
        if self.args.pseudo_box and self.training:
            pseudo_feature=text_feature[-1]
            text_feature=text_feature[:-1] # remove wildcard embedding 
            mask=get_match_pseudo_idx(region_proposals.sigmoid(),targets)
        sizes = [((1 - m[0].float()).sum(), (1 - m[:, 0].float()).sum()) for m in src_feature.decompose()[1]]
        with torch.no_grad():
            if "RN" in self.args.backbone:
                roi_features=sample_feature_rn(sizes,
                                           region_proposals.sigmoid(),
                                           src_feature.tensors,
                                           self.args,
                                           backbone)
            else:
                roi_features = sample_feature_vit(sizes,
                                            region_proposals.sigmoid(),
                                            src_feature.tensors)
        outputs_class = roi_features @ text_feature.t()
        with torch.no_grad():
            outputs_class = torch.cat([outputs_class,torch.ones_like(outputs_class[:, :, :1]) * -1.0,],dim=-1,)
            outputs_class = (outputs_class * 100).softmax(dim=-1)
            if self.args.target_class_factor != 1.0 and not self.training:
                if outputs_class.size(-1) == 66:
                    target_index = COCO_INDEX
                elif outputs_class.size(-1) == 1204:
                    target_index = LVIS_INDEX
                else:
                    assert False, "the dataset may not be supported"
                outputs_class[:, :, target_index] = (
                    outputs_class[:, :, target_index] * self.args.target_class_factor
                )
            outputs_class = outputs_class[:, :, :-1]  
            classes_ = outputs_class.max(-1)[1]
            if self.args.pseudo_box and self.training:
                classes_[mask] = self.args.num_label_sampled    # override pseudo class 
            classes_, indices = classes_.sort(-1) 
            indices = indices.unsqueeze(-1).expand(
                indices.size(0), indices.size(1), 4
            )  #  bs,num_query,  4
        query_box = torch.gather(region_proposals, 1, indices)
        if self.args.pseudo_box and self.training:
            text_feature=torch.cat((text_feature,pseudo_feature.unsqueeze(0)),dim=0)
        projected_text = self.text_proj(text_feature) 
        if classes_.dim() == 3:
            used_classes_ = classes_[:, :, 0]
        else:
            used_classes_ = classes_
        query_features = (F.one_hot(used_classes_, num_classes=text_feature.size(0)).to(text_feature.dtype)@ projected_text)
        return classes_, query_features, query_box

def build_ov_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from models.ov_dquo.utils import RandomBoxPerturber

        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise,
            y_noise_scale=args.dln_xy_noise,
            w_noise_scale=args.dln_hw_noise,
            h_noise_scale=args.dln_hw_noise,
        )
    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False
    return OVDeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,
        deformable_encoder=True,
        deformable_decoder=True,
        # num_feature_levels=args.num_feature_levels,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,
        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,
        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,
        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,
        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        args=args,
    )
