"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified by Akshay Rangesh, UC San Diego

------------------------------------------------------------

Usage: Run from the command line as such:

    # Train a new model starting from ImageNet weights
    python3 handynet.py train --dataset=/path/to/dataset/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 handynet.py train --dataset=/path/to/dataset/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 handynet.py train --dataset=/path/to/dataset/ --model=last
"""

import os
import time
import numpy as np
import scipy.io
import zipfile
import shutil

from config import Config
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class HandyNetConfig(Config):
    """Configuration for training on HandyNet Dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "HandyNet"  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 6 # originally 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 1257 # originally 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64] # originally [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1+5  # Originally 1

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128) # originally (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64 # originally 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 100 # originally 2000
    POST_NMS_ROIS_INFERENCE = 100 # originally 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False # orignally True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 423 # orignally 800
    IMAGE_MAX_DIM = 512 # orignally 1024
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([0.0, 0.0, 0.0])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 20 # orignally 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.1 # orignally 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 10 # orignally 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 10 # orignally 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Choose negative ROIs that are closer to the steering wheel in 3D
    USE_SMART_TRAIN_ROIS = False
    
    # Factor with which to increase ROI for object classification
    ROI_EXPANSION_FACTOR = 0.5


############################################################
#  Dataset
############################################################

class HandSegDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
        """Load a subset of the HandSeg dataset.
        dataset_dir: The root directory of the HandSeg dataset.
        subset: What to load (train, val, test)
        """
        
        filename  = '{}/{}.txt'.format(dataset_dir, subset)
        info = open(filename, 'r').read().splitlines()

        image_ids = len(info)
        class_ids = [1, 2, 3, 4, 5]
        class_names = ['no_object', 'smartphone', 'tablet', 'drink', 'book/newspaper']

        # Add classes
        for i, class_id in enumerate(class_ids):
            self.add_class("HandSeg", class_id, class_names[i])

        # Add images
        for i in range(image_ids):
            identifier = info[i].split(",")
            self.add_image(
                "HandSeg", image_id=i,
                path=os.path.join(dataset_dir, subset, identifier[0], 'smooth_depth', identifier[1]+'.mat'),
                mask_path=os.path.join(dataset_dir, subset, identifier[0], 'instance_label', identifier[1]+'.mat'),
                info=info[i],
                width=512,
                height=424)


    def load_image(self, image_id):
        """Load depth image corresponding to given image_id.

        Returns:
        image: A float array of shape [height, width, 3] with
            the following channels: depth, row_indices, col_indices.
        """
        depth = scipy.io.loadmat(self.image_info[image_id]['path'])['currdata']
        depth = np.log(depth) - 6.756
        depth = 144.*depth[..., np.newaxis]
        output = np.concatenate((depth, depth, depth), axis = 2)
        return output


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        mat_file = scipy.io.loadmat(self.image_info[image_id]['mask_path'])
        masks = mat_file['L'].astype(bool)
        if masks.ndim == 2:
            masks = masks[..., np.newaxis]
        class_ids = mat_file['object_class'].astype(np.int32)
        class_ids = np.squeeze(class_ids) + 1
        if len(class_ids.shape) == 0:
            class_ids = class_ids[np.newaxis, ...]
        return masks, class_ids

    def image_reference(self, image_id):
        """Return some information realted to current inputs"""
        return self.image_info[image_id]['info']
        

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on HandSeg Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on dataset (no evaluate for now)")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--model', required=False,
                        default="imagenet",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HandyNetConfig()
    else:
        class InferenceConfig(HandyNetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    if args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    else:
        model_path = args.model

    # Load weights
    if model_path == "random":
        print("Initializing model with random weights...")
    else:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as in the Mask RCNN paper.
        dataset_train = HandSegDataset()
        dataset_train.load_dataset(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = HandSegDataset()
        dataset_val.load_dataset(args.dataset, "val")
        dataset_val.prepare()

        # Training - Stage 1
        print("Training network backbone")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='backbone')

        # Training - Stage 2
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=80,
                    layers='4+')

        # Training - Stage 4
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=120,
                    layers='all')

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
