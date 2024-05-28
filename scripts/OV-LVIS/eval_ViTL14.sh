#!/bin/bash

output_dir=$1
device=${2:-cuda:0}
python main.py \
	--output_dir $output_dir -c config/OV_LVIS/OVDQUO_ViTL14.py  \
	--amp --device $device \
	--resume /mnt/SSD8T/home/wjj/code/OV-DINO_lvis/logs/OVDINO/lvis_exp13/checkpoint0031.pth --eval 