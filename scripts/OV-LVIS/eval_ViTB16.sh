#!/bin/bash

output_dir=$1
device=${2:-cuda:0}
python main.py \
	--output_dir $output_dir -c config/OV_LVIS/OVDQUO_ViTB16.py  \
	--amp --device $device \
	--resume /mnt/SSD8T/home/wjj/code/OV-DINO-LVIS2/logs/OVDINO/lvis_exp17/checkpoint0037.pth --eval 