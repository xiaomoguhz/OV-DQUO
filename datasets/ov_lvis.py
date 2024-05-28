"""
LVIS dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import math
import datasets.transforms as T
from .lvis import LvisDetection as TvLvisDetection
import json

class LvisDetection(TvLvisDetection):
    def __init__(self, img_folder, 
                 ann_file, 
                 transforms, 
                 label_map, 
                 debug, 
                 repeat_factor_sampling=False, 
                 repeat_threshold=0.001, 
                 pseudo_box=''):
        super(LvisDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.debug = debug
        self.prepare = ConvertCocoPolysToMask(self.cat2label, label_map)
        self.all_categories = {k['id']: k['name'].replace('_', ' ') for k in self.lvis.dataset['categories']}
        self.category_list = [self.all_categories[k] for k in sorted(self.all_categories.keys())]
        self.category_ids = {v: k for k, v in self.all_categories.items()}
        self.label2catid = {k: self.category_ids[v] for k, v in enumerate(self.category_list)}
        # self.catid2label = {v: k for k, v in self.label2catid.items()}
        if repeat_factor_sampling:
            # 1. For each category c, compute the fraction of images that contain it: f(c)
            counter = {k: 1e-3 for k in self.category_ids.values()}
            for id in self.ids:
                ann_ids = self.lvis.get_ann_ids(img_ids=[id])
                target = self.lvis.load_anns(ann_ids)
                cats = [t['category_id'] for t in target]
                cats = set(cats)
                for c in cats:
                    counter[c] += 1
            num_images = len(self.ids)
            for k, v in counter.items():
                counter[k] = v / num_images
            # 2. For each category c, compute the category-level repeat factor:
            #    r(c) = max(1, sqrt(t / f(c)))
            category_rep = {
                cat_id: max(1.0, math.sqrt(repeat_threshold / cat_freq))
                for cat_id, cat_freq in counter.items()
            }
            # 3. For each image I, compute the image-level repeat factor:
            #    r(I) = max_{c in I} r(c)
            rep_factors = []
            for id in self.ids:
                ann_ids = self.lvis.get_ann_ids(img_ids=[id])
                target = self.lvis.load_anns(ann_ids)
                cats = [t['category_id'] for t in target]
                cats = set(cats)
                rep_factor = max({category_rep[cat_id] for cat_id in cats}, default=1.0)
                rep_factors.append(rep_factor)
            self.rep_factors = rep_factors
        self.use_pseudo_box = pseudo_box != ""
        if self.use_pseudo_box:
            with open(pseudo_box, "r") as f:
                pseudo_annotations = json.load(f)
            self.pseudo_annotations = dict()
            for annotation in pseudo_annotations:
                if annotation["image_id"] not in self.pseudo_annotations:
                    self.pseudo_annotations[annotation["image_id"]] = []
                self.pseudo_annotations[annotation["image_id"]].append(annotation)

    def __getitem__(self, idx):
        img, target = super(LvisDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        if self.use_pseudo_box:
            pseudo_annotations = (
                self.pseudo_annotations[image_id]
                if image_id in self.pseudo_annotations
                else [])
            target.extend(pseudo_annotations)
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if len(target["labels"]) == 0:
            return self[(idx + 1) % len(self)]
        else:
            return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self,cat2label=None, label_map=False):
        self.cat2label = cat2label
        self.label_map = label_map

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.label_map:
            classes = [self.cat2label[obj["category_id"]] if obj["category_id"]!=-1 else -1 for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        pseudo_mask=[]
        weight=[]
        for i, obj in enumerate(anno):
            if "pseudo" in obj:
                pseudo_mask.append(obj["pseudo"])
            else:
                pseudo_mask.append(0) # 0 for non-pseudo labels

            if "weight" in obj:
                weight.append(obj["weight"]) #  FE score for pseudo labels
            else:
                weight.append(1.0)
        pseudo_mask= torch.tensor(pseudo_mask, dtype=torch.int64)
        weight= torch.tensor(weight, dtype=torch.float32)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        pseudo_mask=pseudo_mask[keep]
        weight=weight[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["pseudo_mask"] = pseudo_mask
        target["weight"] = weight
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] if "area" in obj else 0 for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize([args.resolution[0]], max_size=args.resolution[1]),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize([args.resolution[0]], max_size=args.resolution[1]),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                 T.RandomResize([args.resolution[0]], max_size=args.resolution[1]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.lvis_path)
    assert root.exists(), f"provided LVIS path {root} does not exist"
    PATHS = {"train": (root / "Images", 
                  root / "Annotations/lvis_v1_train_norare.json"),
                "val": (root / "Images",
                root / "Annotations/lvis_v1_val.json"),}
    if args.label_version == 'lvis_relabel':
        PATHS['train'] = (root/"Images",root / "Annotations/lvis_train_base_relabel.json")
    img_folder, ann_file = PATHS[image_set]
    dataset = LvisDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        label_map=args.label_map,
        debug=args.debug,
        repeat_factor_sampling=args.repeat_factor_sampling and image_set == 'train',
        repeat_threshold=args.repeat_threshold and image_set == 'train',
        pseudo_box=args.pseudo_box if image_set == 'train' else '',
    )
    return dataset