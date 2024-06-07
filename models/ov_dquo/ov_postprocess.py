import torch
import torch.nn as nn
from util import box_ops
from detectron2.layers import batched_nms


class OVPostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, args):
        super().__init__()
        if args.dataset_file == "ovlvis":
            self.max_det = 300
        else:
            self.max_det = 100
        self.nms_thres=args.nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        score_threshold = 0.001
        boxes__ = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scores = []
        labels = []
        boxes = []
        out_logits = out_logits.sigmoid()
        for class_logit, coords in zip(out_logits, boxes__):
            valid_mask = torch.isfinite(coords).all(dim=1) & torch.isfinite(
                class_logit
            ).all(dim=1)
            if not valid_mask.all():
                coords = coords[valid_mask]
                class_logit = class_logit[valid_mask]
            coords = coords.unsqueeze(1)
            filter_mask = class_logit > score_threshold
            filter_inds = filter_mask.nonzero()
            coords = coords[filter_inds[:, 0], 0]
            scores_ = class_logit[filter_mask]
            keep = batched_nms(coords, scores_, filter_inds[:, 1], self.nms_thres)
            keep = keep[: self.max_det]
            coords, scores_, filter_inds = (
                coords[keep],
                scores_[keep],
                filter_inds[keep],
            )
            scores.append(scores_)
            labels.append(filter_inds[:, 1])
            boxes.append(coords)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = [box * fct[None] for box, fct in zip(boxes, scale_fct)]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]
        return results
