import os
import sys
import numpy as np
import timeit
import argparse
import scipy.io

import handynet
import utils
import model as modellib
import visualize

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
smooth_depth = scipy.io.loadmat(sys.argv[2])['currdata']
smooth_depth = np.log(smooth_depth) - 6.756
smooth_depth = 144.*smooth_depth[..., np.newaxis]
image = np.concatenate((smooth_depth, smooth_depth, smooth_depth), axis = 2)

# Run network
start_time = timeit.default_timer()
results = model.detect([image], verbose=0)
elapsed = timeit.default_timer() - start_time
print("Frame rate: %fHz" % (1./elapsed,))

# Obtain results
class_names = ['BG', 'no_object', 'smartphone', 'tablet', 'drink', 'book/newspaper']
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], display_time=5)
