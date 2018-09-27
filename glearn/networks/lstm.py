import tensorflow as tf
from .layer import NetworkLayer


class LSTMLayer(NetworkLayer):
    def __init__(self, network, index, hidden_sizes=[128], cell_args={"forget_bias": 1},
                 embedding_lookup=False, activation=tf.nn.relu, initializer=None):
        super().__init__(network, index)

        self.hidden_sizes = hidden_sizes
        self.cell_args = cell_args
        self.embedding_lookup = embedding_lookup
        self.activation = activation
        self.initializer = initializer

    def build(self, inputs, outputs=None):
        # get variables
        dropout = self.context.get_feed("dropout")
        if dropout is None:
            dropout = tf.placeholder(tf.float32, (), name="dropout")
            self.context.set_feed("dropout", dropout)

        # get configs
        batch_size = self.context.config.get("batch_size", 1)
        timesteps = self.context.config.get("timesteps", 1)
        # timesteps = self.context.input.shape[0]
        vocabulary_size = self.context.dataset.vocabulary.size  # FIXME - ...better way of exposing

        # initializer
        initializer_seed = 1
        if isinstance(self.initializer, int):
            initializer_seed = self.initializer
            self.initializer = None
        if self.initializer is None:
            init_scale = 0.05  # TODO - expose
            self.initializer = tf.random_uniform_initializer(-init_scale, init_scale,
                                                             seed=initializer_seed)

        # process inputs into embeddings
        x = inputs
        if self.embedding_lookup:
            with tf.device("/cpu:0"):
                # default initializer
                embedding = tf.get_variable("embedding", [vocabulary_size, self.hidden_sizes[0]],
                                            initializer=self.initializer)
                x = tf.nn.embedding_lookup(embedding, x)

                # debugging fetches
                visualize_embedded = False  # HACK - expose
                visualize_embeddings = False
                if visualize_embedded:
                    self.context.set_fetch("embedded", x, "debug")
                if visualize_embeddings:
                    self.context.set_fetch("embedding", embedding, "debug")

        # first dropout here
        x = tf.nn.dropout(x, dropout)

        # define lstm cell(s)
        if len(self.hidden_sizes) == 1:
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_sizes[0], **self.cell_args)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        else:
            cells = []
            for hidden_size in self.hidden_sizes:
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, **self.cell_args)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        # prepare lstm state
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        # build unrolled layers
        x = tf.unstack(x, timesteps, 1)
        x, state = tf.nn.static_rnn(cell, x, initial_state=initial_state)

        # reshape for output
        x = tf.reshape(tf.concat(x, 1), [-1, self.hidden_sizes[-1]])

        if outputs is None:
            return x

        # create output layer
        y = self.fully_connected(x, self.hidden_sizes[-1], vocabulary_size,
                                 dropout, tf.nn.softmax)

        # calculate prediction and accuracy
        with tf.name_scope('predict'):
            predict = tf.cast(tf.argmax(y, axis=1), tf.int32)
            batched_predict = tf.reshape(predict, [batch_size, timesteps])
            self.context.set_fetch("predict", batched_predict, ["predict", "debug"])

            correct_prediction = tf.equal(predict, tf.reshape(outputs, [-1]))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.context.set_fetch("accuracy", accuracy, "evaluate")

        # calculate loss and cost
        with tf.name_scope('loss'):
            logits_shape = [batch_size, timesteps, vocabulary_size]
            logits = tf.reshape(self.references["Z"], logits_shape)
            weights = tf.ones([batch_size, timesteps])  # , dtype=self.input.dtype)
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits, outputs, weights,
                                                             average_across_timesteps=False,
                                                             average_across_batch=True)
            self.context.set_fetch("sequence_loss", sequence_loss, "evaluate")

            loss = tf.reduce_sum(sequence_loss)
            self.context.set_fetch("loss", loss, "evaluate")
        return y

    def fully_connected(self, x, index, hidden_size, dropout, activation):
        scope = f"fc_{self.index}_{index}"
        with tf.name_scope(scope):
            # create variables
            with tf.variable_scope(scope):
                W = tf.get_variable("W", (x.shape[1], hidden_size),
                                    initializer=self.initializer)
                b = tf.get_variable("b", (hidden_size, ), initializer=self.initializer)

            # weights and biases
            Z = tf.matmul(x, W)
            Z = tf.add(Z, b)

            # activation
            if activation is not None:
                self.references["Z"] = Z
                A = activation(Z)
            else:
                A = Z

            # dropout
            if dropout is not None:
                self.references["undropped"] = A
                A = tf.nn.dropout(A, dropout)
        return A
