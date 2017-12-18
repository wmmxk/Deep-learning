from _init_paths import *
import os
from keras.utils import to_categorical
from data_gen.data_generator import *
tr_generator = data_generator(os.path.join(project_dir,"data"))
for img, label in tr_generator:
    print(img.shape)
    print(label.shape)




from data_gen.data_generatorc import *
tr_generator_c = data_generator_c(os.path.join(project_dir,"data"))
for img, label in tr_generator_c:
    print(img.shape)
    print(label.shape)
