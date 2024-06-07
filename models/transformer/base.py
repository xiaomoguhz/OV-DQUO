from util import box_ops
import torchvision
import torch
import torch.nn.functional as F
from .attention import multi_head_attention_forward_trans as MHA_woproj
COCO_INDEX = [
    4,
    5,
    11,
    12,
    15,
    16,
    21,
    23,
    27,
    29,
    32,
    34,
    45,
    47,
    54,
    58,
    63,
]
LVIS_INDEX = [
    12,
    13,
    16,
    19,
    20,
    29,
    30,
    37,
    38,
    39,
    41,
    48,
    50,
    51,
    62,
    68,
    70,
    77,
    81,
    84,
    92,
    104,
    105,
    112,
    116,
    118,
    122,
    125,
    129,
    130,
    135,
    139,
    141,
    143,
    146,
    150,
    154,
    158,
    160,
    163,
    166,
    171,
    178,
    181,
    195,
    201,
    208,
    209,
    213,
    214,
    221,
    222,
    230,
    232,
    233,
    235,
    236,
    237,
    239,
    243,
    244,
    246,
    249,
    250,
    256,
    257,
    261,
    264,
    265,
    268,
    269,
    274,
    280,
    281,
    286,
    290,
    291,
    293,
    294,
    299,
    300,
    301,
    303,
    306,
    309,
    312,
    315,
    316,
    320,
    322,
    325,
    330,
    332,
    347,
    348,
    351,
    352,
    353,
    354,
    356,
    361,
    363,
    364,
    365,
    367,
    373,
    375,
    380,
    381,
    387,
    388,
    396,
    397,
    399,
    404,
    406,
    409,
    412,
    413,
    415,
    419,
    425,
    426,
    427,
    430,
    431,
    434,
    438,
    445,
    448,
    455,
    457,
    466,
    477,
    478,
    479,
    480,
    481,
    485,
    487,
    490,
    491,
    502,
    505,
    507,
    508,
    512,
    515,
    517,
    526,
    531,
    534,
    537,
    540,
    541,
    542,
    544,
    550,
    556,
    559,
    560,
    566,
    567,
    570,
    571,
    573,
    574,
    576,
    579,
    581,
    582,
    584,
    593,
    596,
    598,
    601,
    602,
    605,
    609,
    615,
    617,
    618,
    619,
    624,
    631,
    633,
    634,
    637,
    639,
    645,
    647,
    650,
    656,
    661,
    662,
    663,
    664,
    670,
    671,
    673,
    677,
    685,
    687,
    689,
    690,
    692,
    701,
    709,
    711,
    713,
    721,
    726,
    728,
    729,
    732,
    742,
    751,
    753,
    754,
    757,
    758,
    763,
    768,
    771,
    777,
    778,
    782,
    783,
    784,
    786,
    787,
    791,
    795,
    802,
    804,
    807,
    808,
    809,
    811,
    814,
    819,
    821,
    822,
    823,
    828,
    830,
    848,
    849,
    850,
    851,
    852,
    854,
    855,
    857,
    858,
    861,
    863,
    868,
    872,
    882,
    885,
    886,
    889,
    890,
    891,
    893,
    901,
    904,
    907,
    912,
    913,
    916,
    917,
    919,
    924,
    930,
    936,
    937,
    938,
    940,
    941,
    943,
    944,
    951,
    955,
    957,
    968,
    971,
    973,
    974,
    982,
    984,
    986,
    989,
    990,
    991,
    993,
    997,
    1002,
    1004,
    1009,
    1011,
    1014,
    1015,
    1027,
    1028,
    1029,
    1030,
    1031,
    1046,
    1047,
    1048,
    1052,
    1053,
    1056,
    1057,
    1074,
    1079,
    1083,
    1115,
    1117,
    1118,
    1123,
    1125,
    1128,
    1134,
    1143,
    1144,
    1145,
    1147,
    1149,
    1156,
    1157,
    1158,
    1164,
    1166,
    1192,
]

@torch.no_grad()
def sample_feature_vit(
    sizes,
    pred_boxes,
    features,
    unflatten=True,
):
    rpn_boxes = [box_ops.box_cxcywh_to_xyxy(pred) for pred in pred_boxes]
    for i in range(len(rpn_boxes)):
        rpn_boxes[i][:, [0, 2]] = rpn_boxes[i][:, [0, 2]] * sizes[i][0]
        rpn_boxes[i][:, [1, 3]] = rpn_boxes[i][:, [1, 3]] * sizes[i][1]
    roi_features = torchvision.ops.roi_align(
        features,
        rpn_boxes,
        output_size=(1, 1),
        spatial_scale=1.0,
        aligned=True,
    )[..., 0, 0]
    normalized_roi_features =  F.normalize(roi_features, dim=-1, p=2)
    if unflatten:
        normalized_roi_features = normalized_roi_features.unflatten(0, (features.size(0), -1))
    return normalized_roi_features

@torch.no_grad()
def sample_feature_rn(
    sizes,
    pred_boxes,
    features,
    args,
    backbone,
    extra_conv=False,
    unflatten=True,
):
    rpn_boxes = [box_ops.box_cxcywh_to_xyxy(pred) for pred in pred_boxes]
    for i in range(len(rpn_boxes)):
        rpn_boxes[i][:, [0, 2]] = rpn_boxes[i][:, [0, 2]] * sizes[i][0]
        rpn_boxes[i][:, [1, 3]] = rpn_boxes[i][:, [1, 3]] * sizes[i][1]
    if "RN50x4" in args.backbone:
        reso = 18
    else: 
        reso = 14
    if extra_conv:
        roi_features = torchvision.ops.roi_align(
            features,
            rpn_boxes,
            output_size=(reso, reso),
            spatial_scale=1.0,
            aligned=True,
        )
        roi_features = backbone[0].layer4(roi_features)
        roi_features = backbone[0].attn_pool(roi_features, None)
    else:
        features = features.permute(0, 2, 3, 1)
        attn_pool = backbone[0].attn_pool
        q_feat = attn_pool.q_proj(features)
        k_feat = attn_pool.k_proj(features)
        v_feat = attn_pool.v_proj(features)
        hacked = False
        positional_emb = attn_pool.positional_embedding
        if not hacked:
            q_pe = F.linear(positional_emb[:1], attn_pool.q_proj.weight)
            k_pe = F.linear(positional_emb[1:], attn_pool.k_proj.weight)
            v_pe = F.linear(positional_emb[1:], attn_pool.v_proj.weight)
        if q_pe.dim() == 3:
            assert q_pe.size(0) == 1
            q_pe = q_pe[0]
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

@torch.no_grad()
def get_match_pseudo_idx(query,targets,thr=0.5):
    bs,nq=query.shape[:2]
    device=query.device
    pseudo_gt_box=[]
    sizes=[]
    mask=[]
    proposal=query.clone()
    proposal=box_ops.box_cxcywh_to_xyxy(proposal.flatten(0, 1))
    for t in targets:
        pseudo_mask=t['pseudo_mask'].to(torch.bool)
        bbox=box_ops.box_cxcywh_to_xyxy(t['boxes'][pseudo_mask])
        pseudo_gt_box.append(bbox)
        sizes.append(len(bbox))
    pseudo_gt_box = torch.cat(pseudo_gt_box,dim=0)
    ious = box_ops.box_iou(proposal,pseudo_gt_box)[0]
    ious=ious.view(bs,nq,-1)
    for i, iou in enumerate(ious.split(sizes, -1)):
        if iou.numel()>0:
            bs_i_iou=iou[i]
            bs_i_miou=bs_i_iou.max(-1)[0]
            mask.append(bs_i_miou>thr)
        else:
            mask.append(torch.zeros(nq,dtype=torch.bool,device=device))
    return torch.stack(mask,dim=0)
