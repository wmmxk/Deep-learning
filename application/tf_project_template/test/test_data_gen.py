from src.engine.module.data_generator import DataGenerator


def test_data_gen():
    data_gen = DataGenerator()
    images, labels = data_gen.next_batch()
    images, labels = data_gen.next_batch()
    print(images.shape)
