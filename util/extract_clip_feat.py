import argparse
import torch
import time
from datasets import build_dataset
from main import get_args_parser
from torch.utils.data import DataLoader
from models.dino.ov_backbone import build_backbone as build_ov_backbone
from util import box_ops
import util.misc as utils
from tqdm import tqdm
import json
import torchvision
import torch.nn.functional as F
from models.attention import multi_head_attention_forward_trans as MHA_woproj

@torch.no_grad()
def _sample_feature(backbone, sizes, pred_boxes, features, extra_conv, unflatten=True, box_emb=None
):
    rpn_boxes = [box_ops.box_cxcywh_to_xyxy(pred) for pred in pred_boxes]
    for i in range(len(rpn_boxes)):
        rpn_boxes[i][:, [0, 2]] = rpn_boxes[i][:, [0, 2]] * sizes[i][0]
        rpn_boxes[i][:, [1, 3]] = rpn_boxes[i][:, [1, 3]] * sizes[i][1]
    reso = 14
    with torch.cuda.amp.autocast(enabled=False):
        features = features.permute(0, 2, 3, 1)
        attn_pool = backbone[0].attn_pool
        q_feat = attn_pool.q_proj(features)
        k_feat = attn_pool.k_proj(features)
        v_feat = attn_pool.v_proj(features)
        positional_emb = attn_pool.positional_embedding
        q_pe = F.linear(positional_emb[:1], attn_pool.q_proj.weight)
        k_pe = F.linear(positional_emb[1:], attn_pool.k_proj.weight)
        v_pe = F.linear(positional_emb[1:], attn_pool.v_proj.weight)
        if q_pe.dim() == 3:
            assert q_pe.size(0) == 1
            q_pe = q_pe[0]
        # actually this is the correct code. I keep a bug here to trade accuracy for efficiency
        # k_pe = F.linear(attn_pool.positional_embedding, attn_pool.k_proj.weight)
        # v_pe = F.linear(attn_pool.positional_embedding, attn_pool.v_proj.weight)
        q, k, v = (
            q_feat.permute(0, 3, 1, 2),
            k_feat.permute(0, 3, 1, 2),
            v_feat.permute(0, 3, 1, 2),
        )
        q = torchvision.ops.roi_align(
            q,
            rpn_boxes,
            output_size=(reso // 2, reso // 2),
            spatial_scale=1.0,
            aligned=True,
        )
        k = torchvision.ops.roi_align(
            k,
            rpn_boxes,
            output_size=(reso // 2, reso // 2),
            spatial_scale=1.0,
            aligned=True,
        )
        v = torchvision.ops.roi_align(
            v,
            rpn_boxes,
            output_size=(reso // 2, reso // 2),
            spatial_scale=1.0,
            aligned=True,
        )

        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        q = q.mean(-1)  # NC
        q = q + q_pe  # NC
        if k_pe.dim() == 3:
            k = k + k_pe.permute(1, 2, 0)
            v = v + v_pe.permute(1, 2, 0)
        else:
            k = k + k_pe.permute(1, 0).unsqueeze(0).contiguous()  # NC(HW)
            v = v + v_pe.permute(1, 0).unsqueeze(0).contiguous()  # NC(HW)
        q = q.unsqueeze(-1)
        roi_features = MHA_woproj(
            q,
            k,
            v,
            k.size(-2),
            attn_pool.num_heads,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=attn_pool.c_proj.weight,
            out_proj_bias=attn_pool.c_proj.bias,
            training=False,
            out_dim=k.size(-2),
            need_weights=False,
        )[0][0]
        roi_features = roi_features.float()
    roi_features = roi_features / roi_features.norm(dim=-1, keepdim=True)
    if unflatten:
        roi_features = roi_features.unflatten(0, (features.size(0), -1))
    return roi_features

if __name__=="__main__":
    # clip_feat=torch.load("coco/clip_img_feat_coco_base.pkl")
    base_id=[1, 2, 3, 4, 7, 8, 9, 15, 16, 19, 20, 23, 24, 25, 27, 31, 33, 34, 35, 38, 42, 44, 48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 65, 70, 72, 73, 74, 75, 78, 79, 80, 82, 84, 85, 86, 90]
    base_cat_feature={}
    for i in base_id:
        if i not in base_cat_feature.keys():
            base_cat_feature[i]=[]
    device="cuda:7"
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    parser.set_defaults(backbone="clip_RN50")
    parser.set_defaults(device="cuda:7")
    parser.set_defaults(batch_size=4)
    parser.set_defaults(region_prompt_path="logs/region_prompt_R50.pth")
    parser.set_defaults(label_version="RN50base")
    parser.set_defaults(coco_path="/mnt/SSD8T/home/wjj/dataset/coco2017/raw")
    parser.set_defaults(ovd=True)
    parser.set_defaults(anchor_pre_matching=True)
    parser.set_defaults(label_version="normal")
    args = parser.parse_args()
    clip_IE=build_ov_backbone(args)
    dataset_train = build_dataset(image_set='train', args=args)
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train = DataLoader(dataset_train,
                                args.batch_size,
                                sampler=sampler_train,
                                drop_last=False,
                                collate_fn=utils.collate_fn,
                                num_workers=4)
    label2catid=dataset_train.label2catid
    clip_IE=clip_IE.to(device)
    for (samples, targets) in tqdm(data_loader_train):
        labels=[t['labels'] for t in targets]
        bboxes=[t['boxes'] for t in targets]
        samples=samples.to(device)
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        with torch.no_grad():
            features, pos_embeds = clip_IE(samples)
            src_feature = features["layer4"]
            sizes = [
                ((1 - m[0].float()).sum().to(device), (1 - m[:, 0].float()).sum().to(device))
                for m in src_feature.decompose()[1]
            ]
            box_emb = None
            for index,(label, size, bbox) in enumerate(zip(labels,sizes,bboxes)):
                if len(label)==0:
                    continue
                bbox=bbox.unsqueeze(0).to(device)
                size=[size]
                clip_features=_sample_feature(clip_IE, sizes, bbox, src_feature.tensors[index].unsqueeze(0), extra_conv=False, unflatten=True, box_emb=None,)
                clip_features=clip_features.squeeze(0)
                for index,l in enumerate(label):
                    l=label2catid[l.item()]
                    if l in base_id:
                        base_cat_feature[l].append(clip_features[index])
    for k,v in base_cat_feature.items():
        base_cat_feature[k]=torch.stack(v,dim=0).cpu()
    torch.save(base_cat_feature,"coco/my_clip_img_feat_coco_base.pkl")