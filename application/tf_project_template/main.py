from src.engine.agent import Agent
from src.config.config import C as cfg
import tensorflow as tf
from src.engine.module.data_generator import DataGenerator
from src.engine.module.model import Model


def main():
    cfg_file = "/home/wxk/Data_Science/MNIST_sphere/src/config/config.yaml"
    cfg.merge_from_file(cfg_file)
    sess = tf.Session()
    data_gen_train = DataGenerator(batch_size=cfg.TRAIN.BATCH_SIZE)
    data_gen_test = DataGenerator(batch_size=cfg.TRAIN.BATCH_SIZE, is_train=False)
    model = Model(embedding_dim=2, config=cfg)
    agent = Agent(sess, model, data_gen_train, data_gen_test, cfg)
    agent.train(10, resume=True)
    agent.validate()


main()

