from src.engine.module.model import Model
import tensorflow as tf
import numpy as np
from src.config.config import C as cfg


def test_model():

    cfg_file = "/home/wxk/Data_Science/MNIST_sphere/src/config/config.yaml"
    cfg.merge_from_file(cfg_file)

    batch = cfg.TRAIN.BATCH_SIZE
    images = np.random.rand(batch, 28, 28, 1)
    labels = np.random.randint(low=0, high=9, size=(batch,), dtype=np.int64)
    embedding_dim = 2

    model = Model(embedding_dim, cfg)

    embedding = model.embedding
    loss = model.loss

    feed_dict = {model.images: images, model.labels: labels}

    sess = tf.Session()
    embedding_v, loss_v = sess.run([embedding, loss], feed_dict)
    print(embedding_v.shape)
