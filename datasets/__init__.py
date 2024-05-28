import torch.utils.data
import torchvision

from datasets.lvis import LvisDetection
from .coco import build as build_coco
from .ov_coco import build as build_ov_coco
from .ov_lvis import build as build_lvis
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    elif isinstance(dataset, LvisDetection):
        return dataset.lvis

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'ovcoco':
        return build_ov_coco(image_set, args)
    if args.dataset_file == "ovlvis":
        return build_lvis(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
