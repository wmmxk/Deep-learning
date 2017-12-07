'''
signature: tf.boolean_mask(tensor, mask)
          mask's shape must match the first K dimensions of tensor's shape.

The shape of return value: e.g tensor is of shape (19,19,5,4) and the mask is of shape (19, 19, 5)
                     the shape of the return tensor is (None,4)


The shape of the maks should be statically determined.
'''
import tensorflow as tf
import numpy as np
tensor = tf.constant([[1,2],[3,4],[5,6]])
mask = np.array([False,True, False]) #[False, True] does not work
print(tf.boolean_mask(tensor,mask))

# create a mask with the same shape as the input
mask2 = tensor > 1
# the shape of the survive is (none,)
survive = tf.boolean_mask(tensor,mask2)

with tf.Session() as sess:
    print(survive.eval())

