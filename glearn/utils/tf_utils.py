import numpy as np
import tensorflow as tf


def nan_to_num(x, inf=1.0e10):
    if isinstance(x, tf.Tensor):
        if x.dtype.is_floating:
            with tf.variable_scope("nan_to_num"):
                x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)
                x = tf.where(tf.is_inf(x), tf.fill(tf.shape(x), inf), x)
    else:
        x = np.nan_to_num(x)
    return x


def flatten(x, axis=None):
    if len(x.shape) <= 1:
        return x
    elif axis is None:
        return tf.reshape(x, (-1))
    else:
        size = np.prod(x.shape[axis:])
        return tf.reshape(x, (-1, size))


def huber_loss(x, delta=1.0):
    # Huber loss (https://en.wikipedia.org/wiki/Huber_loss)
    with tf.variable_scope("huber_loss"):
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
        )
