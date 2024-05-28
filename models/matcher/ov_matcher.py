import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

# 带有伪标注的匈牙利匹配
# 伪标注用pseudo_mask进行了标注, 
# 修改来源于class-aware，对伪标注IoU>0.5的box进行了query feature的替换，class变成48
class OVHungarianMatcher(nn.Module):
    def __init__(
        self,
        num_sample,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_sample=num_sample
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        alpha = 0.25
        gamma = 2.0
        obj_key = "pred_logits"
        box_key = "pred_boxes"
        bs, num_queries = outputs[obj_key].shape[:2]
        out_prob = (outputs[obj_key].flatten(0, 1).sigmoid())  # [batch_size * num_queries, num_classes]
        out_bbox = outputs[box_key].flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        neg_cost_class = ((1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-7).log()))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-7).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        # Compute the L1 cost between boxes
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1)
        sizes = [len(v["boxes"]) for v in targets]
        if "proposal_classes" in outputs:
            ori_tgt_ids = torch.cat([v["ori_labels"] for v in targets])
            ori_tgt_ids[ori_tgt_ids==-1]=self.num_sample # override the pseudo labels to class+1
            batch_idx = torch.cat(
                [torch.zeros_like(v["ori_labels"]) + i for i, v in enumerate(targets)]
            )
            batched_ori_tgt_ids = (
                torch.zeros_like(ori_tgt_ids).unsqueeze(0).repeat((len(targets), 1)) - 1
            )
            batched_ori_tgt_ids.scatter_(
                0, batch_idx.unsqueeze(0), ori_tgt_ids.unsqueeze(0)
            )
            if outputs["proposal_classes"].dim() == 3:
                batched_ori_tgt_ids = batched_ori_tgt_ids.unsqueeze(1)
                valid_mask = (
                    outputs["proposal_classes"].unsqueeze(-1)
                    == batched_ori_tgt_ids.unsqueeze(1)
                ).any(-2)
            else:
                valid_mask = outputs["proposal_classes"].unsqueeze(
                    -1
                ) == batched_ori_tgt_ids.unsqueeze(
                    1
                )  # ! bs, num_query, num_gt_boxes
            giou = -cost_giou.view(bs, num_queries, -1)
            # only consider the correctly classified subset
            giou[~valid_mask] = -1
            valid_box = (giou > 0).any(1)
            valid_mask[~valid_box.unsqueeze(1).expand_as(valid_mask)] = False
            C[~valid_mask] = 99

        C = C.cpu()  # bs,num_query,num_boxes
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        C = [c[i] for i, c in enumerate(C.split(sizes, -1))]
        new_indices = []
        for i, c in enumerate(C):
            mask = (
                c[indices[i]] < 99
            ).numpy()  
            new_indices.append((indices[i][0][mask], indices[i][1][mask]))
        indices = new_indices
        new_ignore = []
        outputs["ignore"] = new_ignore
        true_indices,pseudo_indices,pseudo_weight=_split_true_pseudo(indices,targets)
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in true_indices
        ] , [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in pseudo_indices
        ], pseudo_weight
    
def _split_true_pseudo(indices,targets):
    device=targets[0]["labels"]
    true_indices=[]
    pseudo_indices=[]
    pseudo_weight=[]
    tgt_pseudo_mask=[v["pseudo_mask"].to(torch.bool) for v in targets]
    tgt_pseudo_weight=[v["weight"] for v in targets]
    for i, indice in enumerate(indices):
        src_i, tgt_i= torch.from_numpy(indice[0]).to(device),torch.from_numpy(indice[1]).to(device)
        pseudo_idx=tgt_pseudo_mask[i].nonzero(as_tuple=True)[0]
        pseudo_weight_i=tgt_pseudo_weight[i]
        p_mask=torch.isin(tgt_i,pseudo_idx)
        pseudo_weight_i=pseudo_weight_i[tgt_i[p_mask]]
        pseudo_weight.append(pseudo_weight_i)
        true_indices.append((src_i[~p_mask],tgt_i[~p_mask]))
        pseudo_indices.append((src_i[p_mask],tgt_i[p_mask]))
    pseudo_weight=torch.cat(pseudo_weight)

    return true_indices,pseudo_indices,pseudo_weight

class HungarianMatcher(nn.Module):

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(
                0, 1
            )  # [batch_size * num_queries, 4]
            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            tgt_pseudo_mask =[v["pseudo_mask"].to(torch.bool) for v in targets]
            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )

            # Final cost matrix
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            C = C.view(bs, num_queries, -1).cpu()
            sizes = [len(v["boxes"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]
            true_indices,pseudo_indices,pseudo_weight=_split_true_pseudo(indices,targets)
            pseudo_indices = [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in pseudo_indices
                ]
            true_indices = [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in true_indices
                ]
            
            return true_indices,pseudo_indices,pseudo_weight


def build_ov_matcher(args):
    return OVHungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        num_sample=args.num_label_sampled,

    ), HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    )
