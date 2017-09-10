from _init_paths import *
import os 
print(os.getcwd())

import numpy as np
weights_path = "../pretrained_weights/VGG_imagenet.npy"

weights_dict= np.load(weights_path).item()
