import tensorflow as tf

x = tf.constant([1, 2, 0, 4])
y = tf.Variable([1, 2, 0, 4])
mask = x > 1
slice_y_greater_than_one = tf.boolean_mask(y, mask)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(slice_y_greater_than_one)) # [2 4]
