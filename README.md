# SCN
A PyTorch implementation of Separable Convolutional Network based on the paper [Separable Convolutional Network for Acoustic Scene Classification]().

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
- librosa
```
pip install librosa
```

## Datasets
The datasets are coming from [DCASE2018](http://dcase.community/challenge2018/) and 
[DCASE2019](http://dcase.community/challenge2019/)
Download these datasets and extract them into `data` directory. Make sure the dir names 
are same with dataset names.

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'DCASE2018A'](choices=['DCASE2018A', 'DCASE2018B','DCASE2019A', 'DCASE2019B'])
--batch_size                  train batch size [default value is 32]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser.

## Results
The train/val/test loss、accuracy and confusion matrix are showed on visdom.

- DCASE2018 Task1 Subtask A

![result](results/dcase2018_1a.png)

- DCASE2018 Task1 Subtask B

![result](results/dcase2018_1b.png)

- DCASE2019 Task1 Subtask A

![result](results/dcase2019_1a.png)

- DCASE2019 Task1 Subtask B

![result](results/dcase2019_1b.png)
