import tensorflow as tf


def sequence_loss(network, outputs, prefix=None, graphs="evaluate"):
    # get variables
    context = network.context
    batch_size = context.config.get("batch_size", 1)
    timesteps = context.config.get("timesteps", 1)
    # timesteps = context.input.shape[0]
    vocabulary_size = context.dataset.vocabulary.size  # FIXME - ...better way of exposing
    predict = context.get_fetch("predict", graphs)
    logits = network.get_output_layer().references["Z"]

    # calculate loss
    logits_shape = [batch_size, timesteps, vocabulary_size]
    logits = tf.reshape(logits, logits_shape)
    weights = tf.ones([batch_size, timesteps])  # , dtype=self.input.dtype)
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits, outputs, weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=True)
    # context.set_fetch(f"sequence_{name}", sequence_loss, graphs)
    loss = tf.reduce_sum(sequence_loss)

    # calculate accuracy
    correct_prediction = tf.equal(tf.reshape(predict, [-1]), tf.reshape(outputs, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy
