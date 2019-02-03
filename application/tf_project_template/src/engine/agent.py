import tensorflow as tf


class Agent:
    def __init__(self, sess, model, data_gen_train, data_gen_test, config):
        self.sess = sess
        self.model = model
        self.data_gen_train = data_gen_train
        self.data_gen_test = data_gen_test
        self.config = config
        self.saver = self._init()

    def _init(self):
        self.init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init_ops)
        return tf.train.Saver(max_to_keep=1000)

    def train(self, n, resume=True):
        if resume:
            self.load_checkpoint()
        for i in range(n):
            loss, acc = self.train_k_iterations()
            print("loss: %s;   acc: %s---------\n" % (loss, acc))
        self.save_checkpoint()

    def train_k_iterations(self):
        for i in range(self.config.TRAIN.NUM_ITE_PER_EPOCH):
            loss, acc = self.train_one_iteration()
        return loss, acc

    def train_one_iteration(self):
        x, y = self.data_gen_train.next_batch()
        feed_dict = {self.model.images: x, self.model.labels: y}
        operations = [self.model.train_op, self.model.add_step_op, self.model.loss, self.model.accuracy,
                      self.model.pred_prob]
        _, _, loss, acc, pred_prob = self.sess.run(operations, feed_dict)
        return loss, acc

    def validate(self):
        self.load_checkpoint()
        for i in range(10):
            x, y = self.data_gen_test.next_batch()
            feed_dict = {self.model.images: x, self.model.labels: y}
            operations = [self.model.accuracy, self.model.pred_prob]
            acc, pred_prob = self.sess.run(operations, feed_dict)
            print("validation acc: %s---------\n" % acc)

    def save_checkpoint(self):
        print("Save model----------------\n")
        print(self.config.SAVE.MODEL_DIR)
        self.saver.save(self.sess, self.config.SAVE.MODEL_DIR, self.model.global_step)
        print("Model saved\n")

    def load_checkpoint(self):
        latest_checkpoint = tf.train.latest_checkpoint("/home/wxk/tmp/sphere_mnist/")
        print(self.config.SAVE.MODEL_DIR)
        print(latest_checkpoint)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded")
