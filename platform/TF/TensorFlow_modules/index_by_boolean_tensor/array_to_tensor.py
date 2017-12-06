# this code is show how to pass data to place holders, do some operations and fetch the values of resulted tensor; no training is involved
import tensorflow as tf
import numpy as np
#create data
np.random.seed(1)
A = np.random.randn(3,3,5,1)
B = np.random.randn(3,3,5,4)

# define placeholders in which the data will pass
tf.reset_default_graph()
box_confidence_p = tf.placeholder("float",[3,3,5,1])
box_class_probs_p = tf.placeholder("float",[3,3,5,4])

# define operations
box_scores_t = box_confidence_p * box_class_probs_p

box_class_score_t = tf.reduce_max(box_scores_t, axis=tf.rank(box_scores_t)-1)
mask = box_class_score_t > 1

print(box_confidence_p.get_shape())
#scores_t = tf.boolean_mask(box_class_score_t, mask)
#
## define what to feed and what to fetch
#feed_dict = {A_p:A, B_p:B}
#fetches = [box_class_score_t, scores_t]
#
## run the operations with the data passed
#with tf.Session() as test:
#    box_class_score_v, scores_v = test.run(fetches, feed_dict)
#print(scores_v)
#
#
