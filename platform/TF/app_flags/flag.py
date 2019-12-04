import tensorflow as tf
import pickle

flags = tf.app.flags
flags.DEFINE_string(flag_name='color',
                    default_value='green',
                    docstring='the color to make a flower')
# you can not dump a module
# pickle.dump(flags, open("flags.pickle", "wb"))

def main(args):
    print('a {} flower'.format(flags.FLAGS.color))


if __name__ == '__main__':
    tf.app.run()
