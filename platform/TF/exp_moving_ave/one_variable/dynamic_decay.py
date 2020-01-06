"""
In this example, we show the ema of v1 = decay * ema + (1-decay) * v1.
The decay is dynamic. decay = min(decay, (1+step)/(1+10))
In the early steps, the decay is small meaning the current value have higher weights when updating ema.

source: https://blog.csdn.net/zhaojianting/article/details/80593189
"""
import tensorflow as tf
v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99, num_updates=step)
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # initialize
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("The initial value of v1 and ema_v1----------", sess.run([v1, ema.average(v1)]))

    # update the value of variable 1. the decay = 0.1
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print("The value of v1 after it is changed and the ema of v1----------", sess.run([v1, ema.average(v1)]))

    # update step and the value of v1. Now the decay = 0.99
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # update the emv of v1
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

