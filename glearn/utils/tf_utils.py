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
