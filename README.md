# OV-DQUO: Open-Vocabulary DETR with Denoising Text Query Training and Open-World Unknown Objects Supervision
This repository is the official PyTorch implementation of [OV-DQUO](https://arxiv.org/abs/2405.17913). 


![Alt text](docs/method.png)
## Overview
OV-DQUO is an open-vocabulary detection framework that learns from open-world unknown objects through wildcard matching and contrastive denoising training strategies, mitigating performance degradation in novel category detection caused by confidence bias.
## TODO
- [x] Release the Model Code
- [ ] Release the Training and Evaluation Code
- [ ] Updating the RoQIs selection Code

## Environment
- Linux with Python == 3.9.0
- CUDA 11.7
- The provided environment is suggested for reproducing our results, similar configurations may also work.

### Quick Start

#### Create conda environment
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
#### Install OpenCLIP 
`pip install -e . -v`
#### Build for DeformableAttention 
```
cd ./models/ops
sh ./make.sh
```
#### Download backbone weights
Download the __ResNet CLIP__ pretrained region prompt weights for __OV-COCO__ experiments from [CORA](https://drive.google.com/drive/folders/17mi8O1YW6dl8TRkwectHRoC8xbK5sLMw) , and place them in the `pretrained` directory. 

Download the __ViT CLIP__ pretrained weights for __OV-LVIS__ experiments from [ViT-B/16](https://drive.google.com/file/d/1-yfrMVaS4aN5uZSYCTalhJ_Pq3j_2aT4/view) and [ViT-L/14](https://drive.google.com/file/d/1_bQMw-R0tBgvFWAAJFi7RbAHN4-OYIz0/view), and place them in the `pretrained` directory.
#### Prepare the datasets
Please download the [COCO dataset](https://cocodataset.org/#download), unzip it, and make sure it is in the following structure:
```
{COCO dataset folder}/
  Annotations/
    instances_train2017.json
    instances_val2017.json
  Images/
    train2017
    val2017
```
Please download the [OV-COCO](https://drive.google.com/drive/folders/1Jgkpoz_ILJRI4xRJydi7dQfFjwtAFbef?usp=sharing) and [OV-LVIS](https://cocodataset.org/#download) dataset annotations, and place them in the `{COCO dataset folder}/Annotations`
#### Prepare the open-world unknwon objects
<!-- Please download the [open-world pseudo labels](https://drive.google.com/drive/folders/1j-i6BkbsHvD_pNXVZRQ6fmAYOWnF4Ao4?usp=sharing), and place them in the `ow_labels` directory.  -->
Download the [open-world pseudo labels](https://drive.google.com/drive/folders/1j-i6BkbsHvD_pNXVZRQ6fmAYOWnF4Ao4?usp=sharing) and place them in the `ow_labels` folder.
## Results & Checkpoints  
Our model achieves the following performance on :
### OV-COCO

| Model name    | __AP50_Novel__  |  Checkpoint |
| ------------  | :------------:  | :------------: |
| OVDQUO_RN50   | __39.2__ | [model](https://drive.google.com/file/d/1scwpSUYzFH-AtzskFSCcOSpM_6dcD3MY/view?usp=sharing)             |
| OVDQUO_RN50x4 | __45.6__ |  [model](https://drive.google.com/file/d/1O7Gu1hWFewo7FD260rHNnUygw7nYAcLF/view?usp=sharing) |

### OV-LVIS
| Model name    | mAP_rare     | Checkpoint |
| ------------  | :------------: | ------------ |
| OVDQUO_ViT-B/16 | __29.7__ |   |                |
| OVDQUO_ViT-L/14 | __39.3__ |   |                 |
## Evaluation

To evaluate my model on OV-COCO, run:

```eval
Please wait for further updates
```
To evaluate my model on OV-LVIS, run:

```eval
Please wait for further updates
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