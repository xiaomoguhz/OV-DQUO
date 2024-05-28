import torch
from models.criterion.ov_criterion import OVSetCriterion
from models.ov_dquo.utils import sigmoid_focal_loss
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import ( get_world_size, 
                       is_dist_avail_and_initialized)
import torch.nn.functional as F

# 计算带有伪标注json训练数据的loss
# 伪标注使用pseudo_mask进行了标注，不计算其bbox的loss
class OVSetCriterion_Pseudo(OVSetCriterion):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.gamma = 0.5

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        device = next(iter(outputs.values())).device
        indices,pseudo_indices,weight = self.ov_matcher(outputs_without_aux, targets)
        num_boxes = sum([index[0].numel() for index in indices])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_pseudo_boxes = sum([index[0].numel() for index in pseudo_indices])
        num_pseudo_boxes = torch.as_tensor([num_pseudo_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pseudo_boxes)
        num_pseudo_boxes = torch.clamp(num_pseudo_boxes / get_world_size(), min=1).item()
        losses = {}


        ######### start for denoise loss ######### 
        dn_meta = outputs["dn_meta"]
        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)
            pseudo_targets=dn_meta['pseudo_targets']
            dn_pos_idx = []
            dn_pos_weight = []
            for i in range(len(pseudo_targets)):
                if len(pseudo_targets[i]['labels']) > 0:
                    t = torch.arange(0, len(pseudo_targets[i]['labels'])).cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                    weight_i=pseudo_targets[i]['weight'][tgt_idx]
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                    weight_i=torch.tensor([]).long().cuda()
                dn_pos_idx.append((output_idx, tgt_idx))
                dn_pos_weight.append(weight_i)
            dn_pos_weight=torch.cat(dn_pos_weight)
            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            l_dict.update(self._loss_labels_denoise(output_known_lbs_bboxes, pseudo_targets, dn_pos_idx,num_boxes*scalar,dn_pos_weight))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).cuda()
            losses.update(l_dict)
        ######### end for denoise loss ######### 


        ######### start bbox&classification loss for decoders ######### 
        for loss in self.losses:
            if loss=="labels":
                losses.update(self._loss_labels_vfl(outputs, targets, indices,num_boxes,pseudo_indices, num_pseudo_boxes,weight))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices, pseudo_indices,weight = self.ov_matcher(aux_outputs, targets[: aux_outputs["pred_logits"].size(0)])
                for loss in self.losses:
                    kwargs = {}
                    if loss == "masks":
                        continue
                    elif loss == "labels":
                        kwargs = {"log": False}
                        l_dict =self._loss_labels_vfl(aux_outputs, 
                                                targets[: aux_outputs["pred_logits"].size(0)], 
                                                indices,
                                                num_boxes,
                                                pseudo_indices, 
                                                num_pseudo_boxes,
                                                weight,
                                                **kwargs)
                    else:
                        l_dict = self.get_loss(
                            loss,
                            aux_outputs,
                            targets[: aux_outputs["pred_logits"].size(0)],
                            indices,
                            num_boxes,
                            **kwargs,
                        )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][i]
                    l_dict = {}
                    l_dict.update(self._loss_labels_denoise(aux_outputs_known, pseudo_targets, dn_pos_idx,num_boxes*scalar,dn_pos_weight))
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to(device)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        ######### start bbox & classification loss for decoders ######### 


        ######### start bbox & classification loss for encoders ######### 
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            if "split_class" in outputs.keys():
                target_len = int(len(targets) / 2)
                targets = targets[:target_len]
            indices,pseudo_indices,weight = self.vanilla_matcher(interm_outputs, targets)
            for loss in self.losses:
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs = {"log": False}
                    l_dict = self._loss_labels_vfl(interm_outputs, targets, indices,num_boxes,pseudo_indices, num_pseudo_boxes,weight)
                else:
                    l_dict = self.get_loss(
                        loss, interm_outputs, targets, indices, num_boxes, **kwargs
                    )
                l_dict = {k + f"_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)
        ######### start bbox & classification loss for encoders ######### 

        
        return losses
    
    def _loss_labels(self, outputs, targets, indices, num_boxes, pseudo_indices, num_pseudo_boxes,pseudo_weight, log=True, dn=False):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] 
        idx = self._get_src_permutation_idx(indices)
        pseudo_idx= self._get_src_permutation_idx(pseudo_indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            src_logits.size(-1),
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o 
        target_classes[pseudo_idx] = 0
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        ) 
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
            reduce=False,
        )
        weight_mask = torch.ones_like(loss_ce)
        pseudo_weight=pseudo_weight.view(-1, 1)**self.gamma
        weight_mask[pseudo_idx]=pseudo_weight
        loss_ce = loss_ce * weight_mask
        loss_ce = loss_ce.mean(1).sum() / (num_boxes+num_pseudo_boxes) * src_logits.shape[1]
        losses = {"loss_ce": loss_ce}
        return losses
    
    def _loss_labels_denoise(self,outputs, targets, indices, num_boxes,weight):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] 
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            src_logits.size(-1),
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = 0 
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        ) 
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
            reduce=False,
        )
        weight_mask = torch.ones_like(loss_ce)
        weight=weight.view(-1, 1)**self.gamma
        weight_mask[idx]=weight
        loss_ce = loss_ce * weight_mask
        loss_ce = loss_ce.mean(1).sum() / (num_boxes) * src_logits.shape[1]
        losses = {"loss_ce": loss_ce}
        return losses
    
    def _loss_labels_vfl(self, outputs, targets, indices, num_boxes, pseudo_indices, num_pseudo_boxes, pseudo_weight, log=True, dn=False):
        focal_alpha=0.75
        focal_gamma=2.0
        assert 'pred_boxes' in outputs
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        src_boxes = outputs['pred_boxes']
        idx = self._get_src_permutation_idx(indices)
        pseudo_idx= self._get_src_permutation_idx(pseudo_indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        pseudo_target_boxes=torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, pseudo_indices)], dim=0)
        ground_ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes[idx]), box_cxcywh_to_xyxy(target_boxes))
        pseudo_ious, _ =box_iou(box_cxcywh_to_xyxy(src_boxes[pseudo_idx]), box_cxcywh_to_xyxy(pseudo_target_boxes))
        ground_ious = torch.diag(ground_ious).detach()
        pseudo_ious = torch.diag(pseudo_ious).detach()
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            src_logits.size(-1),
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o 
        target_classes[pseudo_idx] = 0
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        ) 
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ground_ious.to(target_score_o.dtype)
        target_score_o[pseudo_idx] = pseudo_ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target_classes_onehot

        pred_score = F.sigmoid(src_logits).detach()
        weight = focal_alpha * pred_score.pow(focal_gamma) * (1 - target_classes_onehot) + target_score
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        weight_mask = torch.ones_like(loss_ce)
        pseudo_weight=pseudo_weight.view(-1, 1)**self.gamma
        weight_mask[pseudo_idx]=pseudo_weight
        loss_ce = loss_ce * weight_mask
        loss_ce = loss_ce.mean(1).sum() / (num_boxes+num_pseudo_boxes) * src_logits.shape[1]
        losses = {"loss_ce": loss_ce}
        return losses


