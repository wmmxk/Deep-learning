from _init_paths import *
from data_gen.data_generator import *
tr_generator = generator(images_tr, labels_tr,10)
te_generator = generator(images_te, labels_te, 10)
print("X shape: ",images_tr.shape[1:])

print("Y shape: ",labels_tr.shape[1])
