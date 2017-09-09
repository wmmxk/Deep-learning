import tensorflow as tf
from .libs.network import Network
from config.config import cfg

class VGGnet(Network):
    def __init__(self,trainable=True,num_class = cfg.DATA.NUM_CLASS)
        self.inputs=[]
        self.data = tf.placeholder(tf.float32, shape = [None, None, None, 1], name = 'data')
        self.labels = tf.placeholder(tf.float32,shape = [None, num_class], name ='labels')

