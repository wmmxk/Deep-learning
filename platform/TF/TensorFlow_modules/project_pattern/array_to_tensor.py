# this code is show how to pass data to place holders, do some operations and fetch the values of resulted tensor; no training is involved
import tensorflow as tf
import numpy as np
#create data
np.random.seed(1)
A = np.random.randn(3,3,5,1)
B = np.random.randn(3,3,5,4)

# define placeholders in which the data will pass
tf.reset_default_graph()
A_p = tf.placeholder("float",[3,3,5,1])
B_p = tf.placeholder("float",[3,3,5,4])

# define operations
C_t = A_p * B_p
res_t = tf.argmax(C_t, axis=3)

# define what to feed and what to fetch
feed_dict = {A_p:A, B_p:B}
fetches = [C_t,res_t]

# run the operations with the data passed
with tf.Session() as test:
    C_v,res_v = test.run(fetches, feed_dict)
print(res_v)


#variation it is OK to use a dictionary to fecth the value of multiple tensors

"""
fetches = {'C_v':C_t, 'res_v': res_t}
results = test.run(fetches, feed_dict)
print(results['res_v'])

"""
