# RepPoints
A PyTorch implementation of RepPoints based on ICCV 2019 paper [RepPoints: Point Set Representation for Object Detection](https://arxiv.org/abs/1904.11490). 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
- opencv
```
pip install opencv-python
```
- tensorboard
```
pip install tensorboard
```
- pycocotools
```
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```
- fvcore
```
pip install git+https://github.com/facebookresearch/fvcore
```
- detectron2
```
pip install git+https://github.com/facebookresearch/detectron2.git@master
```

## Dataset
The dataset is assumed to exist in a directory called `datasets/`, under the directory where you launch the program.

## Training
To train a model, run
```bash
python train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end RPDet training with `ResNet-50` backbone for `coco` dataset on 8 GPUs, one should execute:
```bash
python train_net.py --config-file configs/r50_coco.yaml --num-gpus 8
```

## Evaluation
Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/r50_coco.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS epochs/model.pth
```

## Results
There are some difference between this implementation and official implementation:
1. The image sizes of `Multi-Scale Training` are (640, 672, 704, 736, 768, 800) for `coco` dataset;
2. The image sizes of `Multi-Scale Training` are (800, 832, 864, 896, 928, 960, 992, 1024) for `cityscapes` dataset;
3. No `RandomCrop` used;
4. Learning rate policy is `WarmupCosineLR`;

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Name</th>
		<th>train time (s/iter)</th>
		<th>inference time (s/im)</th>
		<th>train mem (GB)</th>
		<th>PA</br>%</th>
		<th>mean PA %</th>
		<th>mean IoU %</th>
		<th>FW IoU %</th>
		<th>download link</th>
		<!-- TABLE BODY -->
		<!-- ROW: r50 -->
		<tr>
			<td align="center"><a href="configs/r50.yaml">R50</a></td>
			<td align="center">1.04</td>
			<td align="center">0.11</td>
			<td align="center">11.14</td>
			<td align="center">80.49</td>
			<td align="center">53.92</td>
			<td align="center">42.71</td>
			<td align="center">68.69</td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">model</a>&nbsp;|&nbsp;xxi8</td>
		</tr>
		<!-- ROW: r101 -->
		<tr>
			<td align="center"><a href="configs/r101_coco.yaml">R101</a></td>
			<td align="center">1.55</td>
			<td align="center">0.18</td>
			<td align="center">17.92</td>
			<td align="center">81.16</td>
			<td align="center">54.54</td>
			<td align="center">43.61</td>
			<td align="center">69.50</td>
			<td align="center"><a href="https://pan.baidu.com/s/1BeGS7gckGAczd1euB55EFA">model</a>&nbsp;|&nbsp;1jhd</td>
		</tr>
		<!-- ROW: r152 -->
		<tr>
			<td align="center"><a href="configs/r152_coco.yaml">R152</a></td>
			<td align="center">1.95</td>
			<td align="center">0.23</td>
			<td align="center">23.88</td>
			<td align="center">81.73</td>
			<td align="center">56.53</td>
			<td align="center">45.15</td>
			<td align="center">70.40</td>
			<td align="center"><a href="https://pan.baidu.com/s/1c-AWtejmmQs2pk_uNu_kYA">model</a>&nbsp;|&nbsp;wka6</td>
		</tr>
	</tbody>
</table>