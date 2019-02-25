import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np


class SimpleIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = list(zip(data_names, data_shapes))
        self._provide_label = list(zip(label_names, label_shapes))
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration


def test_iter():
    n = 32
    num_classes = 10
    data_iter = SimpleIter(['data'], [(n, 100)],
                           [lambda s: np.random.uniform(-1, 1, s)],
                           ['softmax_label'], [(n,)],
                           [lambda s: np.random.randint(0, num_classes, s)])

    one_batch = data_iter.next()
    data = one_batch.data[0].asnumpy()
    plt.imshow(data + 1)
    plt.show()
    pass
