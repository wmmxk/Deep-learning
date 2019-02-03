import tensorflow as tf


def test_save_model():
    v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    with tf.Session() as sess:
      sess.run(init_op)
      # Do some work with the model.
      inc_v1.op.run()
      dec_v2.op.run()
      # Save the variables to disk.
      step = 2
      save_path = saver.save(sess, "/home/wxk/tmp/model.ckpt", step)
      print("Model saved in path: %s" % save_path)

    with tf.Session() as sess:
        # Restore variables from disk.
        last_checkpoint = tf.train.latest_checkpoint("/home/wxk/tmp")
        print("lastest checkpoint:  %s" % last_checkpoint)
        if last_checkpoint:
            saver.restore(sess, last_checkpoint)
        print("Model restored.")
        # Check the values of the variables
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())
