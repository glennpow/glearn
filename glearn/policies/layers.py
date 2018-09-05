import tensorflow as tf
from glearn.policies.policy import Policy


def add_conv2d(policy, inputs, input_size, input_channels, filter, strides=1, max_pool_k=2,
               padding="SAME", reshape=False):
    # TODO - initializer?

    # create variables and operations
    # TODO - convert this to create object with __enter__/__exit__ to define name_scope
    #        and add layer, etc.     (...huh?)
    index = policy.get_layer_count("conv2d")
    with tf.name_scope(f"conv2d_{index}"):
        info = {}

        with tf.variable_scope(f"conv2d_{index}"):
            height, width, output_channels = filter
            # TODO - random_norm init?
            W = tf.get_variable("W", (height, width, input_channels, output_channels))
            b = tf.get_variable("b", (output_channels))

        x = inputs
        # if reshape:
        #     x = tf.reshape(x, shape=[-1, 28, 28, 1])  # TODO - should pass/infer dims, or trust?

        # Conv2D wrapper, with bias and relu activation
        Z = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        Z = tf.nn.bias_add(Z, b)

        info["Z"] = Z
        A = tf.nn.relu(Z)

        # MaxPool2D wrapper
        info["unpooled"] = A
        A = tf.nn.max_pool(Z, ksize=[1, max_pool_k, max_pool_k, 1],
                           strides=[1, max_pool_k, max_pool_k, 1], padding=padding)

    policy.add_layer("conv2d", A)
    return (A, info)


Policy.add_conv2d = add_conv2d


def add_fc(policy, inputs, input_size, output_size, reshape=False, keep_prob=None,
           initializer=None, activation=tf.nn.relu):
    # initializer
    initializer_seed = 1
    if isinstance(initializer, int):
        initializer_seed = initializer
        initializer = None
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer(seed=initializer_seed)

    # create variables and operations
    index = policy.get_layer_count("fc")
    with tf.name_scope(f"fc_{index}"):
        info = {}

        with tf.variable_scope(f"fc_{index}"):
            W = tf.get_variable("W", (input_size, output_size), initializer=initializer)
            b = tf.get_variable("b", (output_size, ), initializer=initializer)

        x = inputs
        if reshape:
            x = tf.reshape(x, (-1, input_size))

        Z = tf.matmul(x, W)
        Z = tf.add(Z, b)

        if activation is not None:
            info["Z"] = Z
            A = activation(Z)
        else:
            A = Z

        # dropout
        if keep_prob is not None:
            info["undropped"] = A
            A = tf.nn.dropout(A, keep_prob)

        policy.add_layer("fc", A)
        return (A, info)


Policy.add_fc = add_fc
