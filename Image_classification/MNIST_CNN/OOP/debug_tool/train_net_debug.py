from _init_paths import *
from data.data_generator import *
from model.VGGnet import *
from model.FNN import *
from config.config import cfg
from helper.for_training import get_accuracy

import tensorflow as tf

# create a net
#net = VGGnet(input_dim = 28)
net = FNN(input_dim = 28)
loss = net.build_loss()

# define train step
opt = tf.train.AdamOptimizer(0.01)
#opt = tf.train.GradientDescentOptimizer(0.1)
global_step = tf.Variable(0, trainable=False)
train_op = opt.minimize(loss, global_step = global_step)


weights_path = "../pretrained_weights/VGG_imagenet.npy"
#sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    net.load_variable(weights_path, sess)

    res_fetches = [loss, net.get_output('cls_prob')]
    fetch_list = [train_op] + res_fetches

    tr_generator = generator(images_tr,labels_tr,100)

    for i in range(300):
        ims,labels = next(tr_generator)
        feed_dict = {net.data: ims, net.labels: labels}

        _, loss_v,cls_prob = sess.run([train_op,loss,net.get_output('cls_prob')], feed_dict = feed_dict)

        accuracy = get_accuracy(cls_prob,labels)
        if i%10==0:
            print("loss:, ", loss_v)
           # print("prediction:",cls_prob)
           # print("labels:", labels)
            print(i," batch:", "accuracy: ", accuracy)
            

    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(i) 
