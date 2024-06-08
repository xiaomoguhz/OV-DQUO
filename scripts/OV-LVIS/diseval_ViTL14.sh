#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
output_dir=$1


master_addr="127.0.0.1"
master_port=29501

python -m torch.distributed.launch --nproc_per_node=8 --master_addr=$master_addr --master_port=$master_port main.py \
        -c config/OV_LVIS/OVDQUO_ViTL14.py \
        --output_dir $output_dir \
        --amp \
        --eval \
        --resume ckpt/OVDQUO_ViT-L/14_LVIS.pth \
