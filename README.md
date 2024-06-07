# OV-DQUO: Open-Vocabulary DETR with Denoising Text Query Training and Open-World Unknown Objects Supervision
This repository is the official PyTorch implementation of [OV-DQUO](https://arxiv.org/abs/2405.17913). 


![Alt text](docs/method.png)
## Overview
OV-DQUO is an open-vocabulary detection framework that learns from open-world unknown objects through wildcard matching and contrastive denoising training methods, mitigating performance degradation in novel category detection caused by confidence bias.
## TODO
- [x] Release the Model Code
- [x] Release the Training and Evaluation Code
- [ ] Updating the RoQIs selection Code

## Environment
- Linux with Python == 3.9.0
- CUDA 11.7
- The provided environment is suggested for reproducing our results, similar configurations may also work.

## Quick Start

### Create conda environment
```
conda create -n OV-DQUO python=3.9.0
conda activate OV-DQUO
pip install torch==2.0.0 torchvision==0.15.1

# other dependencies
pip install -r requirements.txt

# install detectron2
Please install detectron2==0.6 as instructed in the official tutorial
 (https://detectron2.readthedocs.io/en/latest/tutorials/install.html). 
```
### Install OpenCLIP 
`pip install -e . -v`
### Build for DeformableAttention 
```
cd ./models/ops
sh ./make.sh
```
### Download backbone weights
Download the __ResNet CLIP__ pretrained region prompt weights for __OV-COCO__ experiments from [CORA](https://drive.google.com/drive/folders/17mi8O1YW6dl8TRkwectHRoC8xbK5sLMw) , and place them in the `pretrained` directory. 

Download the [ViT-B/16](https://drive.google.com/file/d/1-yfrMVaS4aN5uZSYCTalhJ_Pq3j_2aT4/view) and [ViT-L/14](https://drive.google.com/file/d/1_bQMw-R0tBgvFWAAJFi7RbAHN4-OYIz0/view) pretrained weights for __OV-LVIS__ experiments from CLIPself, and place them in the `pretrained` directory.
### Download text embedding & precomputed wildcard embeddings(optional)
For the OV-LVIS experiment, you need to download the category name list and pre-computed text embedding and wildcard embedding from this [Link](https://drive.google.com/drive/folders/1xtMPvWfhAc3udfskw4wLVZy3zR_KUvgQ?usp=sharing). Similarly, placing them in the `pretrained` directory. 
### Prepare the datasets
Please download the [COCO dataset](https://cocodataset.org/#download), unzip it, place them in the `data` directory, and make sure it is in the following structure:
```
data/
  Annotations/
    instances_train2017.json
    instances_val2017.json
  Images/
    train2017
    val2017
```
Please download the [OV-COCO](https://drive.google.com/drive/folders/1Jgkpoz_ILJRI4xRJydi7dQfFjwtAFbef?usp=sharing) and [OV-LVIS](https://drive.google.com/drive/folders/1ID3TqDzDMm8VBaY-pPS4WRjoio-rePpO?usp=sharing) dataset annotations, and place them in the `data/Annotations` folder.
### Prepare the open-world unknwon objects
<!-- Please download the [open-world pseudo labels](https://drive.google.com/drive/folders/1j-i6BkbsHvD_pNXVZRQ6fmAYOWnF4Ao4?usp=sharing), and place them in the `ow_labels` directory.  -->
Download the [open-world pseudo labels](https://drive.google.com/drive/folders/1j-i6BkbsHvD_pNXVZRQ6fmAYOWnF4Ao4?usp=sharing) and place them in the `ow_labels` folder.
## Script for training OV-DQUO
To train the OV-DQUO on the OV-COCO dataset, please run the following script:
``` 
# dist training based on RN50 backbone, 8 GPU
bash scripts/OV-COCO/distrain_RN50.sh logs/r50_ovcoco
```
``` 
# dist training based on RN50x4 backbone, 8 GPU
bash scripts/OV-COCO/distrain_RN50x4.sh logs/r50x4_ovcoco
```
To train the OV-DQUO on the OV-LVIS dataset, please run the following script:
``` 
# dist training based on ViT-B/16 backbone, 8 GPU
bash scripts/OV-LVIS/distrain_ViTB16.sh logs/vitb_ovlvis
```
``` 
# dist training based on ViT-L/14 backbone, 8 GPU
bash scripts/OV-LVIS/distrain_ViTL14.sh logs/vitl_ovlvis
```
Our code can also run on a single GPU. You can find the corresponding run script in the `script` folder. However, we have not tested it due to the long training time.

Since the evaluation process of OV-LVIS is very time-consuming and thus significantly prolongs the training time, we adopted an offline evaluation method. After training, please run the following script to evaluate the results of each epoch:
``` 
# offline evaluation
python custom_tools/offline_lvis_eval.py -f logs/vitl_ovlvis -n 15 34 -c config/OV_LVIS/OVDQUO_ViTL14.py
```
## Results & Checkpoints  
### OV-COCO
| Model name    | __AP50_Novel__  |  Checkpoint |
| ------------  | :------------:  | :------------: |
| OVDQUO_RN50_COCO   | __39.2__ | [model](https://drive.google.com/file/d/17Nlo0V4jrJz0bNvivfFXcOcaYZq-Up3x/view?usp=sharing)  |
| OVDQUO_RN50x4_COCO | __45.6__ |  [model](https://drive.google.com/file/d/1bDxIj1spUmqrMRNHGzK5TZd9uhL9T1KG/view?usp=sharing) |

### OV-LVIS
| Model name    | mAP_rare     | Checkpoint |
| ------------  | :------------: | :------------: |
| OVDQUO_ViT-B/16_LVIS | __29.7__ | Please wait for further updates  |   
| OVDQUO_ViT-L/14_LVIS | __39.3__ | Please wait for further updates |   
## Evaluation
To evaluate our pretrained checkpoint on the OV-COCO dataset, please download the checkpoint from above links, and run:
```
# R50
bash scripts\OV-COCO\diseval_RN50.sh logs/r50_ovcoco_eval
# R50x4
bash scripts\OV-COCO\diseval_RN50x4.sh logs/r50x4_ovcoco_eval
```
To evaluate our pretrained checkpoint on the OV-LVIS dataset, please download the checkpoint from above links, and run:
```
# vit-b
bash scripts\OV-LVIS\diseval_ViTB16.sh logs/vitb_ovlvis_eval
# vit-l
bash scripts\OV-LVIS\diseval_ViTL14.sh logs/vitl_ovlvis_eval
```
## Citation and Acknowledgement

### Citation

If you find this repo useful, please consider citing our paper:
```
@misc{wang2024ovdquo,
      title={OV-DQUO: Open-Vocabulary DETR with Denoising Text Query Training and Open-World Unknown Objects Supervision}, 
      author={Junjie Wang and Bin Chen and Bin Kang and Yulin Li and YiChi Chen and Weizhi Xian and Huifeng Chang},
      year={2024},
      eprint={2405.17913},
      archivePrefix={arXiv},
      primaryClass={cs.CV}}
```
### Acknowledgement

This repository was built on top of [DINO](https://github.com/IDEA-Research/DINO), [CORA](https://github.com/tgxs002/CORA/tree/master), [MEPU](https://github.com/frh23333/mepu-owod), and [CLIPself](https://github.com/wusize/CLIPSelf/). We thank the effort from the community.