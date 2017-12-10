from _init_paths import *
from helper.for_training import *

import numpy as np

num = 3
cls_prob = np.random.rand(num,2)
labels = np.random.choice(2,num)
print(get_accuracy(cls_prob,labels))
