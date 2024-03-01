# HDG

## Paper
```
"PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization", CVPR 2024
```

## Directory
```
.
├── alg
├── dalib
├── datautil
├── eval.py
├── eval_hscore.py
├── network
├── scripts
├── train.py
└── utils
```
- __alg__
  - Implements for each algorithm, including basic DG algorithms and our SCI-PD.
- __dalib__
  -  Implements for datasets such as PACS, Office-Home.
- __datautil__
  - Implements loading and preprocessing datasets.
- __eval_hscore.py__
  - Implements H-score of the algorithm after training.
- __eval.py__
  - Implements Accuracy of the algorithm after training.
- __network__
  - Implements algorithm networks.
- __scripts__
  - Scripts to conduct experiments.
- __train.py__
  - Implements training for each algorithm. This code is used in common with the algorithms.
- __utils__
  - Implementation of evaluation metrics and functions used for getter function and DAML.

## Requirement
```
Python 3.7.11
Pytorch 1.8.0
```


## Data Preparation
Prepare the dataset (PACS, Office-Home and DomainNet) and modify the file path in the scripts.

## Train and inference
The main script files are `train.py`, `eval.py`, and `eval_hscore.py`, which can be runned by using `bash scripts/run.sh`.

## Acknowledgment
Thanks to OpenDG-Eval. We modify their code to implement Hybrid Domain Generalization.

## Citation
