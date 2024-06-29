## [CVPR2024] PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization
[Link to our paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_PracticalDG_Perturbation_Distillation_on_Vision-Language_Models_for_Hybrid_Domain_Generalization_CVPR_2024_paper.pdf)
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
      List for each dataset is provided in ./image_list 
```
Modify the file path in the scripts.

## Train and inference
For the training and inference process, please simply execute:
```
bash scripts/run.sh
```
Change `test_envs` to different values (e.g., 0,1,2,3) to conduct leave-one-domain-out protocol.

## Acknowledgment
We thank the authors from [OpenDG-Eval](https://github.com/shiralab/OpenDG-Eval) for reference. We modify their code to implement Hybrid Domain Generalization.

## Citation
```
@inproceedings{chen2024practicaldg,
  title={PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization},
  author={Chen, Zining and Wang, Weiqiu and Zhao, Zhicheng and Su, Fei and Men, Aidong and Meng, Hongying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23501--23511},
  year={2024}
}
```
