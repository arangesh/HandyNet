# HandyNet: A One-stop Solution to Detect, Segment, Localize & Analyze Driver Hands

Keras implementation for training and testing the models described in [HandyNet: A One-stop Solution to Detect, Segment, Localize & Analyze Driver Hands](http://cvrr.ucsd.edu/publications/2018/handynet.pdf).
This repository was created by modifying the pre-existing Mask R-CNN implementation found [here](https://github.com/matterport/Mask_RCNN). 

## Installation
1) Clone this repository.
2) Ensure `keras` and `tensorflow` are installed. This code has been tested with Keras 2.1.4 and Tensorflow 1.4.1.

## Dataset preparation
### 1. Download the HandyNet dataset using [this link](https://drive.google.com/open?id=1wV8gmTgap24MTFxCqno4_TLiB-3YPcc-). 

### 2. Split the dataset into separate training and validation folders as below:
```plain
└── DATASET_ROOT
    ├── train
        ├── seq...
        └── seq...
        ...
        └── seq...
        └── objects.txt
    └── val
        ├── seq...
        └── seq...
        ...
        └── seq...
        └── objects.txt
```
Each `seq...` folder above is a from a separate capture sequence. You can split the sequences into `train` and `val` as per your requirement.
Ensure that the original `objects.txt` is split into two corresponding files, one for each split. 

### 3. Create train-val split using [this MATLAB script](https://github.com/arangesh/HandyNet/blob/master/prepare_data.mat).
Make sure you replace `root` in this script with the actual path to the dataset.

## Training
HandyNet can be trained using [this](https://github.com/arangesh/HandyNet/blob/master/scripts/handynet.py) script as follows:

```shell
python3 handynet.py train --dataset=/path/to/dataset/ --model=imagenet
```

## Testing
An example of using the HandyNet network for inference can be seen in [this script](https://github.com/arangesh/HandyNet/blob/master/scripts/demo_inference.py).

This script can be used to generate results on the KITTI test set as follows:
```shell
python3 demo_inference.py /path/to/inference/model /path/to/smooth/depth/mat/file
```
You can download our trained model using [this link](https://drive.google.com/open?id=1VU7F4r8Wwi2gDnym41WtyGm9MbkNNnSY).