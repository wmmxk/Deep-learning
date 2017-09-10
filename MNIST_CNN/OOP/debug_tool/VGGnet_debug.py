from _init_paths import *
from model.VGGnet import *
from config.config import cfg

net = VGGnet()
if net.layers['cls_prob'].get_shape().as_list()[1]== cfg.DATA.NUM_CLASS:
    print("--------network is created successfully------")

loss = net.build_loss()


