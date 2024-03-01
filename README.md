## [CVPR2024] PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization
[Link to our paper](www.baidu.com)
![image](https://github.com/znchen666/HDG/assets/95161725/327a2f38-a96f-4019-ad2f-2a570c8c6ea8)


## Requirements
```
Python 3.7.11+
Pytorch 1.8.0+
```

## Data Preparation
Download the dataset PACS, OfficeHome and DomainNet.

Arrange data with the following structure:
```
Path/To/Dataset
├── Domain1
      ├── cat
      ├── ......
├── Domain2
      ├── cat
      ├── ......
├── Domain3
      ├── cat
      ├── ......
├── Domain4
      ├── cat
      ├── ......
├── image_list
      ├── train.txt
      ├── val.txt
```
Modify the file path in the scripts.

## Train and inference
For the training and inference process, please simply execute:
```
bash scripts/run.sh
```
Change `test_envs` to different values (e.g., 0,1,2,3) to conduct leave-one-domain-out protocol.

## Acknowledgment
We thank the authors from OpenDG-Eval for referencing on [OpenDG-Eval](https://github.com/shiralab/OpenDG-Eval). We modify their code to implement Hybrid Domain Generalization.

## Citation
