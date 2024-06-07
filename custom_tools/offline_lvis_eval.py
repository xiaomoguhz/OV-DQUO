import argparse
import json
import os
import torch
from lvis import LVISEval, LVISResults
from multiprocessing import Pool
import sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from datasets import build_dataset
from util.slconfig import SLConfig, DictAction

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument('-f','--folder', required=True, type=str)
    parser.add_argument('-n', '--num_files', nargs='+', type=int, required=True)
    parser.add_argument('-m','--max_processes',default=5, type=int)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser

def evaluate_epoch(epoch, args, dataset_val):
    cur_folder = os.path.join(args.folder, f"epoch_{epoch}")
    pred_files = os.listdir(cur_folder)
    cur_lvis_results = []
    for pred_file in pred_files:
        if "pred_" in pred_file:
            temp = torch.load(os.path.join(cur_folder, pred_file), map_location="cpu")
            cur_lvis_results += temp
    lvis_results = LVISResults(dataset_val.lvis, cur_lvis_results, max_dets=300)
    lvis_eval = LVISEval(dataset_val.lvis, lvis_results, "bbox")
    lvis_eval.run()
    lvis_eval.print_results()
    result = lvis_eval.get_results()
    with open(os.path.join(cur_folder, "result.txt"), "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("lvis offline evaluation", parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    dataset_val = build_dataset(image_set="val", args=args)
    max_processes=args.max_processes
    print("dataset loaded...")
    print(f"max_processes is {max_processes}")
    print(f"evaluate epoch interval [{args.num_files[0]},{args.num_files[1]})")
    with Pool(processes=max_processes) as pool:
            pool.starmap(evaluate_epoch, [(i, args, dataset_val) for i in range(args.num_files[0],args.num_files[1])])
    print(f"epoch interval [{args.num_files[0]},{args.num_files[1]}) have been evaluated and saved")
