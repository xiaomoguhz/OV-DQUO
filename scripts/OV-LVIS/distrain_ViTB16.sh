#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
coco_path="/mnt/SSD8T/home/wjj/dataset/coco2017/raw"
work_dir="logs/coco_r50x4"

# 默认的主节点地址和端口
master_addr="127.0.0.1"
master_port=29501

python -m torch.distributed.launch --nproc_per_node=8 --master_addr=$master_addr --master_port=$master_port --use_env main.py \
        -c config/OV_LVIS/OVDQUO_ViTL14.py  --output_dir $output_dir \
        --amp \
        --options batch_size=4
