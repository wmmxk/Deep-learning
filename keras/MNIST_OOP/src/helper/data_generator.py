'''
This code is 
      to build a train_generator
      to do test_generator
'''

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

def train_generator(num_classes,w,h,batch_size,every_n=10):
    (X_train, Y_train), (X_test,Y_test) = mnist.load_data()
    
    Y_train = np_utils.to_categorical(Y_train,num_classes)
    X_train = X_train.reshape(X_train.shape[0], w,h,1)

    X_train = X_train.astype('float32')

    X_train /= 255

    while 1:
        for start in range(0,len(X_train),batch_size):  # 1875*32 = 60000 -> # of the training samples
            end = min(start + batch_size, len(X_train))
            if start%(batch_size*every_n)==0:
                print("sample index from " +str(start))
            yield X_train[start:end], Y_train[start:end]
