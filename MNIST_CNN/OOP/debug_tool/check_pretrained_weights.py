from _init_paths import *
import os 
print(os.getcwd())

import numpy as np
weights_path = "../pretrained_weights/VGG_imagenet.npy"

weights_dict= np.load(weights_path).item()
print("layers in VGGnet:",weights_dict.keys())
keys = weights_dict.keys()
print("weights in the one layer:", weights_dict[keys[0]].keys())
