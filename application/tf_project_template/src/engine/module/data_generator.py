from tensorflow.examples.tutorials.mnist import input_data


class DataGenerator:
    def __init__(self, is_train=True, batch_size=32):
        self.mnist = input_data.read_data_sets("./MNIST_data", one_hot=False, reshape=False)
        self.is_train = is_train
        self.batch_size = batch_size

    def next_batch(self):
        if self.is_train:
            # TODO yield does not work
            return self.mnist.train.next_batch(self.batch_size)
        else:
            return self.mnist.test.next_batch(self.batch_size)
