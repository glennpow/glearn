import tensorflow as tf
from .layer import NetworkLayer


class LSTMLayer(NetworkLayer):
    def __init__(self, network, index, hidden_sizes=[128],
                 cell_type="basic", cell_args={"forget_bias": 1},
                 embedding_lookup=False, activation=tf.nn.relu, embedding_initializer=None):
        super().__init__(network, index)

        self.hidden_sizes = hidden_sizes
        self.cell_type = cell_type
        self.cell_args = cell_args
        self.embedding_lookup = embedding_lookup
        self.activation = activation
        self.embedding_initializer = embedding_initializer

        self.debug_embeddings = self.context.config.is_debugging("debug_embeddings")
        self.debug_embedded = self.context.config.is_debugging("debug_embedded")

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
            embedding_initializer = tf.random_uniform_initializer(-SCALE, SCALE)
        else:
            embedding_initializer = self.load_initializer(self.embedding_initializer)

        # process inputs into embeddings
        x = inputs
        if self.embedding_lookup:
            with tf.device("/cpu:0"):
                # default initializer
                embedding = self.get_variable("embedding", [vocabulary_size, self.hidden_sizes[0]],
                                              initializer=embedding_initializer,
                                              trainable=self.trainable)
                x = tf.nn.embedding_lookup(embedding, x)

                # debugging fetches
                if self.debug_embeddings:
                    self.context.set_fetch("embedding", embedding, "evaluate")
                elif self.debug_embedded:
                    self.context.set_fetch("embedded", x, "evaluate")

        # first dropout here
        x = tf.nn.dropout(x, self.dropout)

        # define lstm cell(s)
        cells = []
        for hidden_size in self.hidden_sizes:
            if self.cell_type == "basic":  # TODO - deprecate
                is_training = True  # HACK FIXME
                cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, **self.cell_args,
                                                    state_is_tuple=True, reuse=not is_training)
            else:
                cell = tf.contrib.rnn.LSTMBlockCell(hidden_size, **self.cell_args)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
            cells.append(cell)
        if len(cells) == 1:
            cell = cells[0]
        else:
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

    # def build_predict(self, y):
    #     # get configs
    #     batch_size = self.context.config.get("batch_size", 1)
    #     timesteps = self.context.config.get("timesteps", 1)
    #     vocabulary_size = self.context.dataset.vocabulary.size

    #     # create output layer and convert to batched sequences
    #     y = self.dense(y, vocabulary_size, self.dropout, tf.nn.softmax)
    #     y = tf.cast(tf.argmax(y, axis=1), tf.int32)
    #     y = tf.reshape(y, [batch_size, timesteps])

    #     return y

    def build_loss(self, outputs):
        # get variables
        context = self.network.context
        batch_size = context.config.get("batch_size", 1)
        timesteps = context.config.get("timesteps", 1)
        # timesteps = context.input.shape[0]
        vocabulary_size = context.dataset.vocabulary.size  # FIXME - ...better way of exposing
        predict = self.network.outputs
        logits = self.references["Z"]

        # calculate loss
        logits_shape = [batch_size, timesteps, vocabulary_size]
        logits = tf.reshape(logits, logits_shape)
        weights = tf.ones([batch_size, timesteps])  # , dtype=self.input.dtype)
        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits, outputs, weights,
                                                         average_across_timesteps=False,
                                                         average_across_batch=True)
        loss = tf.reduce_sum(sequence_loss)

        # calculate accuracy
        correct_prediction = tf.equal(tf.reshape(predict, [-1]), tf.reshape(outputs, [-1]))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return loss, accuracy
