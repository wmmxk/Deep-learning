from _init_paths import *
from models.CNN import *

model = get_CNN(input_shape = (128,128,3), num_classes = 10)
