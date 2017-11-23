'''

this code is to 
     test the train_generator
          the CNN model
          model.fit_generator
'''

from _init_paths import *
print(src_dir)
from helper.data_generator import train_generator
from helper.model import CNN_classify

num_classes = 10
w=28
h= 28
batch_size = 500
every_n =20

tr_generator = train_generator(num_classes,w,h,batch_size,every_n)

model=CNN_classify(w,h,num_classes)

model.fit_generator(tr_generator, steps_per_epoch = 40, nb_epoch =2, verbose = 2)
# steps_per_epoch defines the number of batches in an epoch
#print(help(model.fit_generator))
