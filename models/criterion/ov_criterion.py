import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ov_dquo.utils import sigmoid_focal_loss
from util import box_ops
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized


class OVSetCriterion(nn.Module):
    """
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        ov_matcher,
        vanilla_matcher,
        weight_dict,
        focal_alpha,
        losses,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        # self.num_classes = num_classes
        self.ov_matcher = ov_matcher
        self.vanilla_matcher = vanilla_matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, dn=False):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # ! bs , num_query, 1
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            src_logits.size(-1),
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o  # ! bs, num_query
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )  # ! bs, num_query, 2
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
        if not dn and "ignore" in outputs.keys():
            for i, ig in enumerate(outputs["ignore"]):
                weight_mask[i, ig] = 0.0
        loss_ce = loss_ce * weight_mask
        loss_ce = loss_ce.mean(1).sum() / num_boxes * src_logits.shape[1]
        losses = {"loss_ce": loss_ce}
        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups
        return output_known_lbs_bboxes, single_pad, num_dn_groups

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        device = next(iter(outputs.values())).device
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.ov_matcher(outputs_without_aux, targets)

        if "ignore" in outputs_without_aux:
            outputs["ignore"] = outputs_without_aux["ignore"]
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = sum([index[0].numel() for index in indices])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # ! for denoise loss
        dn_meta = outputs["dn_meta"]
        if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
            # label_noise_idx = dn_meta["label_noise_idx"].flatten()
            input_query_mask = dn_meta["input_query_mask"]
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"])).long().cuda(device)
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda(
                        device
                    ).unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda(device)
                # ! remove noise label idx from pos
                pos_output_idx, pos_tgt_idx = self.label_noise_post_process(
                    output_idx,
                    tgt_idx,
                    input_query_mask[i],
                )

                # invalid_mask = torch.isin(
                #     output_idx, label_noise_idx
                # )  # for label noise
                # output_idx = output_idx[~invalid_mask]
                # tgt_idx = tgt_idx[~invalid_mask]
                output_idx = pos_output_idx
                tgt_idx = pos_tgt_idx
                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]

            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False, "dn": True}
                l_dict.update(
                    self.get_loss(
                        loss,
                        output_known_lbs_bboxes,
                        targets,
                        dn_pos_idx,
                        num_boxes * scalar,
                        **kwargs,
                    )
                )  # ! denoise没有用匈牙利匹配
            l_dict = {k + f"_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)

        else:
            l_dict = dict()
            l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to(device)
            l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to(device)
            l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to(device)

            losses.update(l_dict)
        # !
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.ov_matcher(
                    aux_outputs, targets[: aux_outputs["pred_logits"].size(0)]
                )
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
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
                    for loss in self.losses:
                        kwargs = {}
                        if "labels" in loss:
                            kwargs = {"log": False, "dn": True}
                        l_dict.update(
                            self.get_loss(
                                loss,
                                aux_outputs_known,
                                targets,
                                dn_pos_idx,
                                num_boxes * scalar,
                                **kwargs,
                            )
                        )

                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to(device)
                    l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to(device)
                    l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to(device)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            if "split_class" in outputs.keys():
                target_len = int(len(targets) / 2)
                targets = targets[:target_len]
            indices = self.vanilla_matcher(interm_outputs, targets)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs = {"log": False}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

    def label_noise_post_process(self, output_idx, tgt_idx, input_query_mask):
        # label noise后不在target中的pos idx也改为neg
        # valid_mask = torch.isin(after_noise_label[output_idx], gt_labels)
        # pos_output_idx = output_idx[valid_mask]
        # pos_tgt_idx = tgt_idx[valid_mask]
        # * label noise之后 query class和gt class不匹配的为neg
        mask = input_query_mask[output_idx]
        return output_idx[mask], tgt_idx[mask]
