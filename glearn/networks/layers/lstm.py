import tensorflow as tf
from .layer import NetworkLayer


class LSTMLayer(NetworkLayer):
    def __init__(self, network, index, hidden_sizes=[128], cell_args={"forget_bias": 1},
                 embedding_lookup=False, activation=tf.nn.relu, embedding_initializer=None):
        super().__init__(network, index)

        self.hidden_sizes = hidden_sizes
        self.cell_args = cell_args
        self.embedding_lookup = embedding_lookup
        self.activation = activation
        self.embedding_initializer = embedding_initializer

    def build(self, inputs):
        # get variables
        self.dropout = self.context.get_or_create_feed("dropout")

        # get configs
        batch_size = self.context.config.get("batch_size", 1)
        timesteps = self.context.config.get("timesteps", 1)
        # timesteps = self.context.input.shape[0]
        vocabulary_size = self.context.dataset.vocabulary.size  # FIXME - ...better way of exposing

        # initializer
        if self.embedding_initializer is None:
            SCALE = 0.05  # TODO - expose
            embedding_initializer = tf.random_uniform_initializer(-SCALE, SCALE, seed=self.seed)
        else:
            embedding_initializer = self.load_initializer(self.embedding_initializer)

        # process inputs into embeddings
        x = inputs
        if self.embedding_lookup:
            with tf.device("/cpu:0"):
                # default initializer
                embedding = tf.get_variable("embedding", [vocabulary_size, self.hidden_sizes[0]],
                                            initializer=embedding_initializer,
                                            trainable=self.trainable)
                x = tf.nn.embedding_lookup(embedding, x)

                # debugging fetches
                visualize_embedded = False  # HACK - expose
                visualize_embeddings = False
                if visualize_embedded:
                    self.context.set_fetch("embedded", x, "debug")
                if visualize_embeddings:
                    self.context.set_fetch("embedding", embedding, "debug")

        # first dropout here
        x = tf.nn.dropout(x, self.dropout)

        # define lstm cell(s)
        if len(self.hidden_sizes) == 1:
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_sizes[0], **self.cell_args)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
        else:
            cells = []
            for hidden_size in self.hidden_sizes:
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, **self.cell_args)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        # prepare lstm state
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        # build unrolled layers
        x = tf.unstack(x, timesteps, 1)
        x, state = tf.nn.static_rnn(cell, x, initial_state=initial_state)

        # reshape for output
        y = tf.reshape(tf.concat(x, 1), [-1, self.hidden_sizes[-1]])
        self.references["hidden"] = y

        return y

    def build_predict(self, y):
        # get configs
        batch_size = self.context.config.get("batch_size", 1)
        timesteps = self.context.config.get("timesteps", 1)
        vocabulary_size = self.context.dataset.vocabulary.size

        # create output layer and convert to batched sequences
        y = self.dense(y, vocabulary_size, self.dropout, tf.nn.softmax)
        y = tf.cast(tf.argmax(y, axis=1), tf.int32)
        y = tf.reshape(y, [batch_size, timesteps])

        return y
