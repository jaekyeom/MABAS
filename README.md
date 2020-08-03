## Model-Agnostic Boundary-Adversarial Sampling for Test-Time Generalization in Few-Shot Learning  
#### In ECCV 2020 (Oral)
*[Jaekyeom Kim](https://jaekyeom.github.io/), Hyoungseok Kim, and [Gunhee Kim](http://vision.snu.ac.kr/~gunhee/)*  
[[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460579.pdf) [[appx]](https://drive.google.com/uc?id=1LhdzmuHBxOOoxrJYf9nR4pVOTOhyX_K4) [[code]](https://github.com/jaekyeom/MABAS)

This repository provides the source code for the application of our method to [Dynamic Few-Shot Visual Learning without Forgetting](https://arxiv.org/abs/1804.09458).

### Environment setup
You can create a conda environment by  
```
conda env create -f environment.yml
```  
and activate it with
```
conda activate mabas-fswf
```

### Downloading datasets

* Create a subdirectory named `datasets`.
* Download and decompress [**miniImageNet**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (from [FSwF](https://github.com/gidariss/FewShotWithoutForgetting) and [MetaOptNet](https://github.com/kjunelee/MetaOptNet)) into `datasets/MiniImagenet/`.
* Download and decompress [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing) (from [MetaOptNet](https://github.com/kjunelee/MetaOptNet)) into `datasets/CIFAR_FS/`.
* Download and decompress [**FC100**](https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing) (from [MetaOptNet](https://github.com/kjunelee/MetaOptNet)) into `datasets/FC100/`.

### Meta-training
| Command                                                                                                                                | Dataset      | Type   |
|:-------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:------:|
|`python train.py --config miniImageNet_ResNetLikeCosineClassifier`                                                                      | miniImageNet | Base   |
|`python train.py --config miniImageNet_ResNetLikeCosineClassifierGenWeightAttN5 --parent_exp "miniImageNet_ResNetLikeCosineClassifier"` | miniImageNet | 5-shot |
|`python train.py --config miniImageNet_ResNetLikeCosineClassifierGenWeightAttN1 --parent_exp "miniImageNet_ResNetLikeCosineClassifier"` | miniImageNet | 1-shot |
|`python train.py --config CIFARFS_ResNetLikeCosineClassifier`                                                                           | CIFAR-FS     | Base   |
|`python train.py --config CIFARFS_ResNetLikeCosineClassifierGenWeightAttN5 --parent_exp "CIFARFS_ResNetLikeCosineClassifier"`           | CIFAR-FS     | 5-shot |
|`python train.py --config CIFARFS_ResNetLikeCosineClassifierGenWeightAttN1 --parent_exp "CIFARFS_ResNetLikeCosineClassifier"`           | CIFAR-FS     | 1-shot |
|`python train.py --config FC100_ResNetLikeCosineClassifier`                                                                             | FC100        | Base   |
|`python train.py --config FC100_ResNetLikeCosineClassifierGenWeightAttN5 --parent_exp "FC100_ResNetLikeCosineClassifier"`               | FC100        | 5-shot |
|`python train.py --config FC100_ResNetLikeCosineClassifierGenWeightAttN1 --parent_exp "FC100_ResNetLikeCosineClassifier"`               | FC100        | 1-shot |

### Test-time fine-tuning with MABAS
| Command                                                                                                                                                                                                  | Dataset      | Type   |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------:|:------:|
|`python evaluate_finetune.py --config miniImageNet_ResNetLikeCosineClassifierGenWeightAttN5 --parent_exp "miniImageNet_ResNetLikeCosineClassifier/miniImageNet_ResNetLikeCosineClassifierGenWeightAttN5"` | miniImageNet | 5-shot |
|`python evaluate_finetune.py --config miniImageNet_ResNetLikeCosineClassifierGenWeightAttN1 --parent_exp "miniImageNet_ResNetLikeCosineClassifier/miniImageNet_ResNetLikeCosineClassifierGenWeightAttN1"` | miniImageNet | 1-shot |
|`python evaluate_finetune.py --config CIFARFS_ResNetLikeCosineClassifierGenWeightAttN5 --parent_exp "CIFARFS_ResNetLikeCosineClassifier/CIFARFS_ResNetLikeCosineClassifierGenWeightAttN5"`                | CIFAR-FS     | 5-shot |
|`python evaluate_finetune.py --config CIFARFS_ResNetLikeCosineClassifierGenWeightAttN1 --parent_exp "CIFARFS_ResNetLikeCosineClassifier/CIFARFS_ResNetLikeCosineClassifierGenWeightAttN1"`                | CIFAR-FS     | 1-shot |
|`python evaluate_finetune.py --config FC100_ResNetLikeCosineClassifierGenWeightAttN5 --parent_exp "FC100_ResNetLikeCosineClassifier/FC100_ResNetLikeCosineClassifierGenWeightAttN5"`                      | FC100        | 5-shot |
|`python evaluate_finetune.py --config FC100_ResNetLikeCosineClassifierGenWeightAttN1 --parent_exp "FC100_ResNetLikeCosineClassifier/FC100_ResNetLikeCosineClassifierGenWeightAttN1"`                      | FC100        | 1-shot |


### Acknowledgments

This source code is based on the implementations for [Dynamic Few-Shot Visual Learning without Forgetting](https://github.com/gidariss/FewShotWithoutForgetting) and [Meta-learning with differentiable convex optimization](https://github.com/kjunelee/MetaOptNet).

