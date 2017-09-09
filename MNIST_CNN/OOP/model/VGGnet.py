import tensorflow as tf
from .libs.network import Network
from config.config import cfg

class VGGnet(Network):
    def __init__(self,trainable=True,num_class = cfg.DATA.NUM_CLASS):
        self.inputs=[]
        self.data = tf.placeholder(tf.float32, shape = [None, None, None, 1], name = 'data')
        self.labels = tf.placeholder(tf.float32,shape = [None, num_class], name ='labels')
        self.layers = dict({'data':self.data})
        self.setup()
        self.num_class = num_class

    def setup(self):

        (self.feed('data').conv(3,3,64,1,1,name = 'conv1_1',trainable = False)
                .conv(3,3,64,1,1,name = 'conv1_2', trainable = False)
                .max_pool(2,2,2,2,name='pool1', padding='VALID')
                .conv(3,3,128,1,1,name = 'conv2_1',trainable = False)
                .conv(3,3,128,1,1,name = 'conv2_2', trainable = False)
                .max_pool(2,2,2,2, name= 'pool2', padding = 'VALID')
                .conv(3,3,256,1,1,name= 'conv3_1')
                .conv(3,3,256,1,1,name='conv3_2')
                .conv(3,3,256,1,1,name='conv3_3')
                .max_pool(2,2,2,2,name='pool3',padding = 'VALID')
                .conv(3,3,512,1,1,name='conv4_1')
                .conv(3,3,512,1,1,name='conv4_2')
                .conv(3,3,512,1,1,name='conv4_3')
                .max_pool(2,2,2,2,name='pool4', padding = 'VALID')
                .conv(3,3,512,1,1,name='conv5_1')
                .conv(3,3,512,1,1,name='conv5_2')
                .conv(3,3,512,1,1,name='conv5_3'))

        (self.feed('conv5_3').fc(1024,name='fc1')
                .dropout(0.5,name='drop1')
                .fc(self.num_class,name='cls_score',relu=False)
                .softmax(name='cls_prob'))
