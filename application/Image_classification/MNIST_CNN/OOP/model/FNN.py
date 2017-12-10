import tensorflow as tf
from .libs.network import Network
from config.config import cfg

class FNN(Network):
    def __init__(self,trainable=True,input_dim=cfg.NETWORK.DIM,num_class = cfg.DATA.NUM_CLASS):
        self.inputs=[]
        self.data = tf.placeholder(tf.float32, shape = [None, input_dim,input_dim, 1], name = 'data')
        self.labels = tf.placeholder(tf.int32,shape = [None], name ='labels')
        self.layers = dict({'data':self.data})
        self.num_class = num_class
        self.setup()

    def setup(self):

        (self.feed('data').conv(3,3,64,1,1,name='conv1_1',trainable = True)
                .conv(3,3,64,1,1,name = 'conv1_2', trainable = True)
                .max_pool(2,2,2,2,name='pool1', padding='VALID')
                .fc(1024,name='fc1')
                .dropout(0.5,name='drop1')
                .fc(self.num_class,name='cls_score',relu=False)
                .softmax(name='cls_prob'))
