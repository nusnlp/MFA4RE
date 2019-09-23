# EAM4RE #

This repository contains the source code of the paper "Effective Attention Modeling for Neural Relation Extraction" published in CoNLL 2019.

### Datasets ###

NYT10 and NYT11 datasets used for experiments in the paper can be downloaded from the following link:

https://drive.google.com/drive/folders/1xWoN8zfK3IA1WZqxBQ1-Nw-y275YE628?usp=sharing

Each dataset has a sub-directory named 'Best' which contains our final model which gives the best result mentioned in Table 2 of our paper.

Each line in the '.json' files is one instance. It containes the sentence text, relation mentions and entity mentions. Fields are self explanatory.

Each line in the '.dep' files containes the dependency distance information of the entities to corresponding line in '.json' file.

### Requirements ###

1) python3.5
2) pytorch 0.2
3) CUDA 7.5

### How to run ###

python re_models.py gpu_id source_dir target_dir model_id train/test multi_factor_count

Use model_id as 1 for CNN, 2 for PCNN, 3 for EA, 4 for BGWA, and 5 for our models.

Use multi_factor_count=0 to run our own baseline model BiLSTM-CNN

Example:

Training command to train our model with multi factor count of 5

python re_models.py 0 NYT10/ NYT10/MFA_5/ 5 train 5

Inference command to test our model with multi factor count of 5

python re_models.py 0 NYT10/ NYT10/MFA_5/ 5 test 5

### Publication ###

If you use the source code or models from this work, please cite our paper:

```
@inproceedings{nayak2019effective,
  author    = {Nayak, Tapas and Ng, Hwee Tou},
  title     = {Effective Attention Modeling for Neural Relation Extraction},
  booktitle = {Proceedings of The SIGNLL Conference on Computational Natural Language Learning (CoNLL)},
  year      = {2019},
}
```


