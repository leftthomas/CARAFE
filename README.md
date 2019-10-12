# RepPoints
A PyTorch implementation of RepPoints based on CVPR 2019 paper 
[RepPoints: Point Set Representation for Object Detection](https://arxiv.org/abs/1904.11490). 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
- mmdetection
```
python setup.py develop
```
## Datasets
The [COCO2017](http://cocodataset.org/#download) dataset is used.
Download it and set the path in `configs` directory.

## Usage

### Train
```shell
# single-gpu training
python train.py ${CONFIG_FILE} [--work_dir ${WORK_DIR}] [--resume_from ${CHECKPOINT_FILE}] [--validate] [--autoscale-lr]
# python train.py configs/reppoints_moment_x101_dcn_fpn_2x_mt.py --validate --autoscale-lr

# multi-gpu training
./train.sh ${GPU_NUM} ${PORT} ${CONFIG_FILE} [--work_dir ${WORK_DIR}] [--resume_from ${CHECKPOINT_FILE}] [--validate] [--autoscale-lr]
# ./train.sh 8 29500 configs/reppoints_moment_x101_dcn_fpn_2x_mt.py --validate --autoscale-lr
```

Optional arguments are:
- `WORK_DIR`: Override the working directory specified in the config file.
- `CHECKPOINT_FILE`: Resume from a previous checkpoint file.
- `--validate`: Perform evaluation at every k (default value is 1) epochs during the training.
- `--autoscale-lr`: Automatically scale lr with the number of gpus.

### Test
```shell
# single-gpu testing
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--json_out ${RESULT_JSON_FILE}] [--eval ${EVAL_METRICS}] [--show]
# python test.py configs/reppoints_moment_x101_dcn_fpn_2x_mt.py checkpoints/reppoints_moment_x101_dcn_fpn_2x_mt.pth --json_out results/results

# multi-gpu testing
./test.sh ${GPU_NUM} ${PORT} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--json_out ${RESULT_JSON_FILE}] [--eval ${EVAL_METRICS}]
# ./test.sh 8 29501 configs/reppoints_moment_x101_dcn_fpn_2x_mt.py checkpoints/reppoints_moment_x101_dcn_fpn_2x_mt.pth --out results/results.pkl  --eval bbox
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `RESULT_JSON_FILE`: Filename of the output results without extension in json format. If not specified, the results will 
not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values are: `proposal_fast`, `proposal`, `bbox`, `segm`, `keypoints`.
- `--show`: If specified, detection results will be ploted on the images and shown in a new window. It is only applicable 
to single GPU testing. Please make sure that GUI is available in your environment, otherwise you may encounter the error 
like `cannot connect to X server`.

## Results

The results on COCO 2017val are shown in the table below.

| Method | Backbone | convert func | Lr schd | box AP | Download |
| :----: | :------: | :-------: | :-----: | :----: | :------: |
| RepPoints | X-101-FPN-DCN | moment | 2x (ms train)   | 45.6| [model](https://drive.google.com/open?id=1nr9gcVWxzeakbfPC6ON9yvKOuLzj_RrJ) |
| RepPoints | X-101-FPN-DCN | moment | 2x (ms train&ms test)   | 46.8|          |

**Notes:**

- `R-xx`, `X-xx` denote the ResNet and ResNeXt architectures, respectively. 
- `DCN` denotes replacing 3x3 conv with the 3x3 deformable convolution in `c3-c5` stages of backbone.
- `moment`, `partial MinMax`, `MinMax` in the `convert func` column are three functions to convert a point set to a 
pseudo box.
- `ms` denotes multi-scale training or multi-scale test.
