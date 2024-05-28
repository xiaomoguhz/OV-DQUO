# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
import copy
import torch
from util.misc import inverse_sigmoid
import torch.nn.functional as F


def prepare_for_cdn_ov(
    dn_args,
    training,
    num_queries,
    num_classes,
    text_embbeding,
    label_enc_embbeding,
):
    device = text_embbeding.device
    if training:
        # * dn_number 100
        # * label_noise_ratio 0.5
        # * box_noise_scale 1.0
        # positive and negative dn queries
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        targets=targets_preprocess(targets)
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).to(device) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]  # gt sum of each img
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                # 选一个batch中label最多的
                dn_number = dn_number // (int(max(known_num) * 2))  # e.g. 12
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
       
        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)  #  single_pad个box，每个box一个pos一个neg，共有dn_number组
        positive_idx = (
            torch.tensor(range(len(boxes)))
            .long()
            .to(device)
            .unsqueeze(0)
            .repeat(dn_number, 1)
        )
        positive_idx += (
            (torch.tensor(range(dn_number)) * len(boxes) * 2)
            .long()
            .to(device)
            .unsqueeze(1)
        )
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned = known_labels_expaned.scatter(0, chosen_indice, new_label)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = (known_bboxs[:, :2] + known_bboxs[:, 2:] / 2)  #  cx,cy,w,h->x,y,x,y
            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2  # 宽高除以2
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32)
                * 2.0
                - 1.0
            )  #  -1或者1,控制box的noise方向
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = (
                known_bbox_
                + torch.mul(rand_part, diff).to(device) * box_noise_scale
            )
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = (
                known_bbox_[:, 2:] - known_bbox_[:, :2]
            )  #  x,y,x,y->cx,cy,w,h
        m = known_labels_expaned.long().to(device)
        m[m==-1]=num_classes-1
        input_label_embed = label_enc_embbeding(m)
        input_label_embed[positive_idx]+=text_embbeding[-1][None,:] # 正样本text embbeding全为object
        input_label_embed[negative_idx]+=text_embbeding[m[negative_idx]] # 负样本可以有noise text embbeding
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        padding_label = torch.zeros(pad_size, 256).to(device)
        padding_bbox = torch.zeros(pad_size, 4).to(device)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        map_known_indice = torch.tensor([]).to(device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                # [0:14,14:196]
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True

            if i == dn_number - 1:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2
                ] = True
            else:
                # [0:14,14:196]
                # [14:28,28:196]
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True
                # [0:14,:0]
                # [14:28,:14]
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i
                ] = True

        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
            "pseudo_targets":targets
        }
    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None
    #  input_query_label -> bs,pad_size,256
    #  input_query_bbox -> bs,pad_size,4
    #  attn_mask -> tgt_size,tgt_size  
    return input_query_label, input_query_bbox, attn_mask, dn_meta

def targets_preprocess(targets):
    new_target=[]
    for target in targets:
        new_target_i=copy.deepcopy(target)
        pseudo_mask=new_target_i['pseudo_mask'].to(torch.bool)
        new_target_i['boxes']=new_target_i['boxes'][pseudo_mask]
        new_target_i['labels']=new_target_i['labels'][pseudo_mask]
        new_target_i['weight']=new_target_i['weight'][pseudo_mask]
        new_target.append(new_target_i)
    # If there is no pseudo annotation, use a real annotation for training
    if sum([len(target['labels']) for target in new_target])==0:
        for i,target in enumerate(targets):
            new_target_i=copy.deepcopy(target)
            if len(new_target_i['labels'])>0:
                new_target_i['boxes']=new_target_i['boxes'][0].unsqueeze(0)
                new_target_i['labels']=new_target_i['labels'][0].unsqueeze(0)
                new_target_i['weight']=new_target_i['weight'][0].unsqueeze(0)
                new_target[i]=new_target_i
                break
    return new_target
        
def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
        output_known_coord = outputs_coord[:, :, : dn_meta["pad_size"], :]
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
        outputs_coord = outputs_coord[:, :, dn_meta["pad_size"] :, :]
        out = {"pred_logits": output_known_class[-1],
                "pred_boxes": output_known_coord[-1],}
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord
