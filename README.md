Train Cifar10 with Pytorch
========================

# Setup

1. python: 3.8+
2. Pytorch: 2.2.1+

```
pip install -r requirements.txt
```

![](images/requirements.png)

# Training

## help
```
python main.py  --help
```
> usage: main.py [-h] --model MODEL --optimizer OPTIMIZER --lr LR
> Deep Learning with Cifar-10 Datasets
> optional arguments:
>  -h, --help            show this help message and exit
>  --model MODEL         choose model from mlp and cnn
>  --optimizer OPTIMIZER
>                        choose optimizer from adam and sgd
>  --lr LR               specify learning rate

## train
```
python main.py --model mlp --optimizer sgd --lr 0.001
```

## modify default value
Please update config.ini to modify default value.
