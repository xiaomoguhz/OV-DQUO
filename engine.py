# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import copy
import json
import math
import os
import random
import sys
from typing import Iterable
from util import box_ops
from util.utils import to_device
import torch
import torchvision
from copy import deepcopy
import util.misc as utils
from datasets.coco_eval import CocoEvaluator, convert_to_xywh
from lvis import LVISEval, LVISResults
# for eval visualize only


def generate_deterministic_rand(num):
    prev_state = random.getstate()
    random.seed(num)
    rand = random.random()
    random.setstate(prev_state)
    return rand

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    lr_scheduler=None,
    args=None,
    ema_m=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    if utils.get_world_size() == 1:
        print_freq = 20
    else:
        print_freq = 200
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [
            {
                k: v if isinstance(v, (list, dict)) else v.to(device)
                for k, v in t.items()
            }
            for t in targets
        ]
        categories = copy.deepcopy(data_loader.dataset.category_list)
        if args.num_label_sampled > 0 and args.dataset_file!="ovcoco":
            if args.pseudo_box:
                gt = torch.cat([target['labels'][~target['pseudo_mask'].to(torch.bool)] for target in targets]).unique()
            else:
                gt = torch.cat([target['labels'] for target in targets]).unique()
            if gt.numel() >= args.num_label_sampled:
                sampled = gt[torch.randperm(gt.numel(), device=gt.device)][:args.num_label_sampled]
            else:
                all_class = torch.arange(len(categories), device=gt.device)
                neg_class = all_class[~(all_class.unsqueeze(1) == gt.unsqueeze(0)).any(-1)]
                num_sample = args.num_label_sampled - gt.numel()
                sampled = neg_class[torch.randperm(neg_class.numel(), device=gt.device)][:num_sample]
                sampled = torch.cat([gt, sampled])
            used_categories = sampled.tolist()
            # reorder
            for target in targets:
                label = target['labels']
                sampled_mask = (label.unsqueeze(-1) == sampled.unsqueeze(0)).any(-1)
                if args.pseudo_box:
                    sampled_mask[target['pseudo_mask'].to(torch.bool)]=True
                target['boxes'] = target['boxes'][sampled_mask]
                label = label[sampled_mask]
                new_label = (label.unsqueeze(-1) == sampled.unsqueeze(0)).int().argmax(-1)
                # reassign pseudo box label
                if args.pseudo_box:
                    new_label[target['pseudo_mask'].to(torch.bool)]=-1
                target['labels'] = new_label
        else:
            used_categories = categories
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples,categories=used_categories,targets=targets)
            for target in targets:
                target["ori_labels"] = target["labels"]
                target["labels"] = target["labels"] - target["labels"]
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        del samples
        del targets
        del outputs
        del loss_dict
        del loss_dict_reduced
        del weight_dict
        del losses
        del losses_reduced_scaled
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
    return resstat

@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    args=None,
    epoch=None,
):
    model.eval()
    criterion.eval()
    if epoch and utils.get_rank()==0 and not os.path.exists(os.path.join(output_dir,f"epoch_{epoch}")):
        os.mkdir(os.path.join(output_dir,f"epoch_{epoch}"))
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    if args.dataset_file == "ovlvis":
        cat2label = data_loader.dataset.cat2label
        label2cat = {v: k for k, v in cat2label.items()}
        lvis_results = []
        label_map = args.label_map
        iou_types = ["bbox"]
    elif args.dataset_file == "ovcoco":
        iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
        coco_evaluator = CocoEvaluator(
            base_ds, iou_types, label2cat=data_loader.dataset.label2catid
        )

    else:
        raise ValueError
    if args.debug or utils.get_world_size() == 1:
        print_freq = 10
    else:
        print_freq = 100
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [
            {
                k: v if isinstance(v, (list, dict)) else v.to(device)
                for k, v in t.items()
            }
            for t in targets
        ]
        outputs = model(
            samples,
            categories=data_loader.dataset.category_list,
            targets=targets,
        )
        # for loss only
        training_target = []
        for target in targets:
            new_target = target.copy()
            new_target["ori_labels"] = target["labels"]
            new_target["labels"] = target["labels"] - target["labels"]
            training_target.append(new_target)
        loss_dict = criterion(outputs, training_target)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
        )
        
        if "class_error" in loss_dict_reduced.keys():
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        else:
            metric_logger.update(class_error=0.0)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
    
        if args.dataset_file == "ovlvis":
            for target, output in zip(targets, results):
                image_id = target["image_id"].item()
                boxes = convert_to_xywh(output["boxes"])
                for ind in range(len(output["scores"])):
                    temp = {
                        "image_id": image_id,
                        "score": output["scores"][ind].item(),
                        "category_id": output["labels"][ind].item(),
                        "bbox": boxes[ind].tolist(),
                    }
                    if label_map:
                        temp["category_id"] = label2cat[temp["category_id"]]
                    lvis_results.append(temp)
        else:
            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, results)
            }
            if coco_evaluator is not None:
                coco_evaluator.update(res)
    
        _cnt += 1
        if args.debug:
            if _cnt % (15 * 5) == 0:
                print("BREAK!" * 5)
                break
    metric_logger.synchronize_between_processes()
    if args.dataset_file == "ovlvis":
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        rank = utils.get_rank()
        if epoch is not None: # 训练期间只保存，不验证
            torch.save(lvis_results, os.path.join(output_dir,f"epoch_{epoch}",f"pred_{rank}.pth"))
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        else:
            torch.save(lvis_results, os.path.join(output_dir,f"pred_{rank}.pth"))
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if rank == 0:
                world_size = utils.get_world_size()
                for i in range(1, world_size):
                    temp = torch.load(output_dir + f"/pred_{i}.pth")
                    lvis_results += temp
                lvis_results = LVISResults(base_ds, lvis_results, max_dets=300)
                for iou_type in iou_types:
                    lvis_eval = LVISEval(base_ds, lvis_results, iou_type)
                    lvis_eval.run()
                    lvis_eval.print_results()
            if rank == 0:
                stats.update(lvis_eval.get_results())
        return stats, None
    
    else:
        print("Averaged stats:", metric_logger)
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
        # accumulate predictions from all images
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if coco_evaluator is not None:
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
        return stats, coco_evaluator