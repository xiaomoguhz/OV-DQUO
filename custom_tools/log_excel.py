import os
import pandas as pd

ROI_KEYS = [
    "epoch",
    "train_lr",
    "train_loss",
    "train_loss_bbox",
    "train_loss_ce",
    "train_loss_giou",
    "test_loss",
    "test_APc",
    "test_APf",
    "test_APr",
    "test_AP",
    "epoch_time",
]


def Log_excel(log_stats, output_dir):
    new_log = {}
    for k in ROI_KEYS:
        new_log[k]=0.0
    for k in log_stats.keys():
        if k in ROI_KEYS:
            if k=="test_APc" or k=="test_APf" or k=="test_APr"or k=="test_AP":
                new_log[k] = log_stats[k]*100
            else:
                new_log[k] = log_stats[k]
    target_file = os.path.join(output_dir, "exp_res.xlsx")
    if os.path.exists(target_file):
        old_res = pd.read_excel(target_file)
        for k in old_res.keys():
            new_list = []
            new_list.extend(old_res[k].tolist())
            new_list.append(new_log[k])
            new_log[k] = new_list
        df = pd.DataFrame(new_log)
    else:
        df = pd.DataFrame([new_log])
    df.to_excel(
        target_file,
        index=False,
    )


if __name__ == "__main__":
    import json
    out = "logs/OVDINO/lvis_exp1/log.txt"
    with open(out, mode="r") as fp:
        for line in fp:
            try:
                # 解析每行的 JSON 对象
                res = json.loads(line)
                Log_excel(res, os.path.dirname(out))
            except json.JSONDecodeError as e:
                print(f"解析错误：{str(e)}")
