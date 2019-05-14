# CCN
A PyTorch implementation of Convolutional Capsule Network based on the paper [Convolutional Capsule Network for Object Detection]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```

## Datasets
TODO

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'CIFAR10'](choices=['CIFAR10'])
--batch_size                  train batch size [default value is 32]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser.

## Results
The train/val/test loss、accuracy and confusion matrix are showed on visdom.

- CIFAR10

![result](results/cifar10.png)
