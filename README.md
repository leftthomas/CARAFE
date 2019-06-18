# PCN
A PyTorch implementation of ProbAM-guided Capsule Network based on the paper [ProbAM-guided Capsule Network for Object Detection]().

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
- capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```
- opencv
```
conda install opencv
```

## Datasets
TODO

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 200
optional arguments:
--data_name                   dataset name [default value is 'voc'](choices=['voc', 'coco', 'cityscapes'])
--batch_size                  train batch size [default value is 32]
--num_iterations              routing iterations number [default value is 3]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/$data_type` in your browser.

### Visualization
```
python vis.py --data_type coco
optional arguments:
--data_name                   dataset name [default value is 'voc'](choices=['voc', 'coco', 'cityscapes'])
--batch_size                  vis batch size [default value is 64]
--num_iterations              routing iterations number [default value is 3]
```
Generated results are on `results` directory.

## Results
The train/test loss, accuracy and confusion matrix are showed on visdom.

- VOC
![result](results/voc.png)

- COCO
![result](results/coco.png)

- CityScapes
![result](results/scityscapses.png)
