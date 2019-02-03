import tensorflow as tf


def original_softmax_loss(embeddings, labels):
    with tf.variable_scope("softmax"):
        weights = tf.get_variable(name="embedding_ws",
                                  shape=[embeddings.get_shape().as_list()[-1], 10],
                                  initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.matmul(embeddings, weights)
        pred_prob = tf.nn.softmax(logits=logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)
        return pred_prob, loss


def modified_softmax_loss(embeddings, labels):
    with tf.variable_scope("softmax"):
        weights = tf.get_variable(name='embedding_weights',
                                  shape=[embeddings.get_shape().as_list()[-1], 10],
                                  initializer=tf.contrib.layers.xavier_initializer())
        weights_norm = tf.norm(weights, axis=0, keepdims=True)
        weights = tf.div(weights, weights_norm, name="nor")