import tensorflow as tf


def simple_loss(network, outputs):
    # get variables
    context = network.context
    output_interface = context.output
    predict = context.get_fetch("predict")
    logits = network.get_output_layer().references["Z"]

    if output_interface.discrete:
        # evaluate discrete loss
        neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=outputs)
        loss = tf.reduce_mean(neg_log_p)

        # evaluate accuracy
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    else:
        # evaluate continuous loss
        loss = tf.reduce_mean(tf.square(outputs - predict))

        # evaluate accuracy
        accuracy = tf.exp(-loss)

    return loss, accuracy
