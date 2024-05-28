#!/bin/bash

output_dir=$1
device=${2:-cuda:0}
python main.py \
	--output_dir $output_dir -c config/OV_LVIS/OVDQUO_ViTL14.py  \
	--amp --device $device\
    --options batch_size=4