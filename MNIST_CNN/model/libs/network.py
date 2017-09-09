import numpy as np
import tensorflow as tf
from config.config import cfg

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name','default_layer_name')
        layer_input = self.inputs[0]
        layer_output = op(self,layer_input, *args, **kwargs)

        self.layers[name] = layer_output
        self.feed(layer_output)
        return self
    return layer_decorated

class Network(object):
    
    '''
    interface: feed, get_output
    layer component: conv, max_pool, fc, dropout, softmax
    initilization and setup:  __init__, setup

    '''
    def __init__(self,inputs,trainable = True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed')

    def feed(self,*args)
        self.inputs = []
            for layer in args:
                layer = self.layers[layer]
                self.inputs.append(layer)

        return self

    def get_output(self,layer):
        return self.layers[layer]

    
    # layer components:
    @layer
    def conv(self,input,k_h,k_w,c_o,s_h,s_w,name,trainable = True):

        c_i = input.get_shape()[-1]

        convolve = lambda i,k : tf.nn.conv2d(i,k,[1,s_h,s_w,1], padding='SAME')

        with tf.variable_scope(name) as scope:
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor = 0.1,mode ='FAN_AVG', uniform = False)
            init_biases = tf.constant_initializer(0.0)
            kernel = tf.get_variable('weights',[k_h,k_w,c_i,c_o], init_weights,trainable,regularizer = self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = tf.get_variable('biases',[c_o],init_biases, trainable)

            conv = convolve(input,kernel)
            bias = tf.nn.bias_add(conv,biases)

            return tf.nn.relu(bias)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding='SAME'):
        return tf.nn.max_pool(input,ksize= [1,k_h,k_w,1], strides = [1,s_h,s_w,1], padding=padding,name=name)


    @layer
    def fc(self, input, num_out, name, relu = True, trainable = True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims==4:
                dim = reduce(lambda x,y: x*y, input_shape[1:].as_list(),1)
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]),[-1,dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            init_weights = tf.truncated_normal_initializer(0,0,stddev = 0.001)
            init_biases = tf.constant_initializer(0.0)

            weights = tf.get_variable('weights',[dim,num_out], init_weights, trainable, regularizer= self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = tf.get_variable('biases',[num_out],init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            return op(feed_in,weights, biases,name=scope.name)

    @layer
    def dropout(self, input,keep_prob, name):
        return tf.nn.dropout(input,keep_prob,name = name)

    @layer
    def softmax(self, input, name):
        return tf.nn.softmax(input,name=name)

   # build loss
   def build_loss(self):
       cls_prob = self.get_output('cls_prob')
       labels = self.labels
       cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_prob, labels = labels)
       return tf.reduce_mean(cross_entropy_n)
