import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#args = sys.argv
#debug = args[1]
debug = False

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

images_tr = np.reshape(mnist.train.images,(-1,28,28,1))
labels_tr = mnist.train.labels

images_te = np.reshape(mnist.test.images,(-1,28,28,1))
labels_te = mnist.test.labels

print("the max grayscale:", np.max(images_te))

def generator(images,labels,batch_size):
    while True:
        for start in range(0,len(labels),batch_size):
            end = min(start + batch_size, len(labels))
            yield images[start:end,:,:,:], labels[start:end,:]



if debug:
    tr_generator = generator(images_tr,labels_tr,3)
    images,labels = next(tr_generator)
    print("images shape:",images.shape)
    print("generator works")


