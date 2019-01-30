import os
import sys
import numpy as np
import timeit

import handynet
import utils
import model as modellib
import argparse
import scipy.io

class InferenceConfig(handynet.HandyNetConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    ROI_EXPANSION_FACTOR=0.5

config = InferenceConfig()
config.display()

# Local path to trained weights file
MODEL_PATH = sys.argv[1]

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='', config=config)

# Load trained weights
model.load_weights(MODEL_PATH, by_name=True)

# Load and preprocess depth file
smooth_depth = scipy.io.loadmat(sys.argv[2])
smooth_depth = np.log(smooth_depth) - 6.756
smooth_depth = 144.*smooth_depth[..., np.newaxis]
image = np.concatenate((smooth_depth, smooth_depth, smooth_depth), axis = 2)

# Run network
start_time = timeit.default_timer()
results = model.detect([image], verbose=0)
elapsed = timeit.default_timer() - start_time
print("Image: %.6d/%.6d, Frame rate: %fHz" % (image_id+1, len(TS_list), 1./elapsed))

# Obtain results
boxes = results[0]['rois']
masks = results[0]['masks']
object_class = results[0]['class_ids']
scores = results[0]['scores']