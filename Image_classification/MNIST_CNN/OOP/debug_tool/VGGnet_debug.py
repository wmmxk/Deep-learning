from _init_paths import *
from model.VGGnet import *
from config.config import cfg
import tensorflow as tf

net = VGGnet(input_dim = 28)
if net.layers['cls_prob'].get_shape().as_list()[1]== cfg.DATA.NUM_CLASS:
    print("--------network is created successfully------")

loss = net.build_loss()

sess = tf.Session()
weights_path = "../pretrained_weights/VGG_imagenet.npy"
net.load_variable(weights_path,sess)
