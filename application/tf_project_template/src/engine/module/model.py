import tensorflow as tf
from src.engine.module.model_module.model_setting import res1_3, res2_3, res2_5, res3_3, res3_5, res3_7, res3_9, res4_3
from .model_module.loss import original_softmax_loss


class Model(object):
    def __init__(self, embedding_dim, config):
        self.images = tf.placeholder(tf.float32, shape=[config.TRAIN.BATCH_SIZE, 28, 28, 1])
        self.labels = tf.placeholder(tf.int64, shape=[config.TRAIN.BATCH_SIZE, ])
        self.global_step = tf.Variable(0, trainable=False)
        self.config = config
        # TODO add a placeholder for is_train
        self.embedding_dim = embedding_dim
        self.embedding = self.__forward()
        self.pred_prob, self.loss = self.__get_loss()
        self.accuracy = self.__get_accuracy()
        self.train_op, self.add_step_op = self.__train_op()

    @staticmethod
    def network(inputs, embedding_dim=2):

        def prelu(inputs, name=''):
            alpha = tf.get_variable(name, shape=inputs.get_shape(), initializer=tf.constant_initializer(0.0),
                                    dtype=inputs.dtype)
            return tf.maximum(alpha*inputs, inputs)

        def conv(inputs, filters, kernel_size, strides, padding='same', suffix='', scope=None):
            conv_name = 'conv' + suffix
            relu_name = 'relu' + suffix

            with tf.name_scope(name=scope):
                w_init = tf.contrib.layers.xavier_initializer(uniform=True)
                input_shape = inputs.get_shape().as_list()
                x = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=padding,
                                     kernel_initializer=w_init, name=conv_name)
                output_shape = x.get_shape().as_list()

                print("=================================================================================")
                print("layer:%8s    input shape:%8s   output shape:%8s" % (conv_name, str(input_shape), str(output_shape)))
                print("---------------------------------------------------------------------------------")

                x = prelu(x, name=relu_name)
                return x

        def resnet_block(x, layers, suffix=''):
            n = len(layers)
            for i in range(n):
                if n == 2 and i == 0:
                    identity = x
                x = conv(inputs=x,
                         filters=layers[i]['filters'],
                         kernel_size=layers[i]['kernel_size'],
                         strides=layers[i]['strides'],
                         padding=layers[i]['padding'],
                         suffix=suffix + '_' + layers[i]['suffix'],
                         scope='conv' + suffix + "_" + layers[i]['suffix'])
                if n == 3 and i == 0:
                    identity = x
            return identity + x

        x = inputs
        suffxes = ('1', '2', '2', '3', '3', '3', '3', '4')
        blocks = [res1_3, res2_3, res2_5, res3_3, res3_5, res3_7, res3_9, res4_3]
        for suffix, layers in zip(suffxes, blocks):
            x = resnet_block(x, layers, suffix=suffix)

        x = tf.layers.flatten(x)
        embedding = tf.layers.dense(x, units=embedding_dim,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return embedding

    def __forward(self):
        return self.network(inputs=self.images, embedding_dim=self.embedding_dim)

    def __get_loss(self):
        return original_softmax_loss(self.embedding, self.labels)

    def __train_op(self):
        decay_lr = tf.train.exponential_decay(self.config.TRAIN.LR, self.global_step, 500, 0.9)
        optimizer = tf.train.AdamOptimizer(decay_lr)
        train_op = optimizer.minimize(self.loss)
        add_step_op = tf.assign_add(self.global_step, tf.constant(1))
        return train_op, add_step_op

    def __get_accuracy(self):
        pred_label = tf.argmax(self.pred_prob, axis=1)
        correct_pred = tf.equal(pred_label, self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        return accuracy

