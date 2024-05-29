import argparse
import json
import os
import torch
from lvis import LVISEval, LVISResults
from datasets import build_dataset
from multiprocessing import Pool
def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('-f','--folder', required=True, type=str)
    parser.add_argument('-n', '--num_files', nargs='+', type=int, required=True)
    parser.add_argument('-m','--max_processes',default=3, type=int)
    parser.add_argument('--lvis_path', default="", type=str)
    parser.add_argument('--dataset_file', default="ov_lvis", type=str)
    parser.add_argument('--label_version', default="", type=str)
    parser.add_argument('--backbone', default="clip_R50x4", type=str)
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--repeat_threshold', default=0.001, type=int)
    parser.add_argument('--class_group', default="", type=str)
    parser.add_argument('--resolution', default=[896,896], type=str)
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
    args.label_map = True
    args.repeat_factor_sampling = True
    dataset_val = build_dataset(image_set="val", args=args)
    print("dataset loaded...")
    max_processes=args.max_processes
    print(f"max_processes is {max_processes}")
    print(f"evaluate epoch interval [{args.num_files[0]},{args.num_files[1]})")
    with Pool(processes=max_processes) as pool:
            pool.starmap(evaluate_epoch, [(i, args, dataset_val) for i in range(args.num_files[0],args.num_files[1])])
    print(f"epoch interval [{args.num_files[0]},{args.num_files[1]}) have been evaluated and saved")