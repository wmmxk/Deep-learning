from _init_paths import *
print(src_dir)
from helper.data_generator import train_generator

num_classes = 10
w=28
h= 28
batch_size =10

tr_generator = train_generator(num_classes,w,h,batch_size)

imgs, labels = next(tr_generator)
print(labels)
