from _init_paths import *
from data.data_generator import *

debug = True

if debug:
    tr_generator = generator(images_tr,labels_tr,3)
    images,labels = next(tr_generator)
    print("images shape:",images.shape)
    print("generator works")

