#!/bin/bash

output_dir=$1
device=${2:-cuda:0}
python main.py \
	--output_dir $output_dir -c config/OV_COCO/OVDQUO_RN50.py  \
	--amp --device $device\
	--resume /mnt/SSD8T/home/wjj/code/my_DINO/logs/OVDINO/exp115/checkpoint0028.pth --eval