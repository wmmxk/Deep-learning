from _init_paths import *
from data.data_generator import *
from model.VGGnet import *
from config.config import cfg
from helper.for_training import get_accuracy

import tensorflow as tf

net = VGGnet()
loss = net.build_loss()

opt = tf.train.AdamOptimizer(0.01)
global_step = tf.Variable(0, trainable=False)

train_op = opt.minimize(loss, global_step = global_step)

sess = 
        
