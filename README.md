# Ranking Distillation

A PyTorch implementation of Ranking Distillation:

*[Ranking Distillation: Learning Compact Ranking Models With High Performance for Recommender System](http://www.sfu.ca/~jiaxit/resources/kdd18ranking.pdf), Jiaxi Tang and Ke Wang , KDD '18*

# Requirements

- Python 2 or 3
- [PyTorch v0.4+](https://github.com/pytorch/pytorch)
- Numpy
- SciPy

# Usage

#### Training student models (Model-S)

1. Run <code>python train_caser.py</code> with <code>d=50</code> to get the performance of *student* model.

   When finished, we will have the model saved in folder *checkpoints/*

#### Training teacher models (Model-T)

1. Run <code>python train_caser.py</code> with <code>d=100</code> to get the performance of *teacher* model.

   When finished, we will have the model saved in folder *checkpoints/*

#### Training student models with *ranking distillation* (Model-RD)

1. Run <code>python train_caser.py</code> with <code>d=100</code> to get the well-trained teacher model.

   When finished, we will have the teacher model saved in folder *checkpoints/*

   (you can also skip this step, as there is one in the *checkpoint/gowalla-caser-dim=100.pth.tar*)

2. Run <code>python distill_caser.py</code> with <code>d=50</code> and <code>teacher_model_path</code> pointed to the teacher model.

# Configurations

#### Model Args (in train_caser.py)

- <code>d</code> is set to 50 for student model and 100 for teacher model.

- All other the hyper-parameters (e.g., <code>nh</code>, <code>nv</code>, <code>ac_conv</code>, <code>ac_fc</code>) are set by grid-search.

  Please check [this repo](https://github.com/graytowne/caser_pytorch) for more information and definations of these hyper-parameters.

#### Model Args (in distill_caser.py)

- <code>teacher_model_path</code>: path to teacher's model checkpoint.
- <code>teacher_topk_path</code>: (optional) path to teacher's top-K ranking cache for each training query.
- <code>teach_alpha</code>:  hyperparameter for balancing ranking loss and distillation loss.
- <code>K</code>: length of teacher's exemplary ranking.
- <code>lamda</code>: hyperparameter for tuning the sharpness of position importance weight.
- <code>mu</code>: hyperparameter for tuning the sharpness of ranking discrepancy weight.
- <code>dynamic_samples</code>: number of samples used for estimating student's rank.
- <code>dynamic_start_epoch</code>: number of iteration to start using hybrid of two different weights.

# Citation

If you use this Caser in your paper, please cite the paper:

```
@inproceedings{tang2018ranking,
  title={Ranking Distillation: Learning Compact Ranking Models With High Performance for Recommender System},
  author={Tang, Jiaxi and Wang, Ke},
  booktitle={ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  year={2018}
}
```

# Acknowledgment

This project (utils.py, interactions.py, etc.) is heavily built on [Spotlight](https://github.com/maciejkula/spotlight). Thanks [Maciej Kula](https://github.com/maciejkula) for his great work.
