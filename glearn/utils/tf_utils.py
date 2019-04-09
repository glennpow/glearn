import tensorflow as tf


def nan_to_num(x, inf=1.0e10):
    x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)
    x = tf.where(tf.is_inf(x), tf.fill(tf.shape(x), inf), x)
    return x
