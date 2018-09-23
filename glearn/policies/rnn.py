import math
import tensorflow as tf
from glearn.policies.policy import Policy


class RNNPolicy(Policy):
    def __init__(self, config,
                 hidden_sizes=[128], keep_prob=1, cell_args={"forget_bias": 1}, **kwargs):
        self.hidden_sizes = hidden_sizes
        self.keep_prob = keep_prob
        self.cell_args = cell_args

        self.batch_size = config.get("batch_size", 1)  # TODO - from dataset/env?

        self.visualize_embeddings = False  # HACK - expose these?
        self.visualize_embedded = False  # HACK

        super().__init__(config, **kwargs)

        self.init_visualize()

    def init_model(self):
        # infer values from dataset
        self.timesteps = self.input.shape[0]  # sequence_length
        self.vocabulary = self.dataset.vocabulary

        # default initializer
        init_scale = 0.05  # TODO - expose
        scaled_init = tf.random_uniform_initializer(-init_scale, init_scale)

        # create placeholders
        with tf.name_scope('feeds'):
            inputs, outputs = self.create_default_feeds()

            self.set_fetch("X", inputs, ["evaluate", "debug"])
            self.set_fetch("Y", outputs, ["evaluate", "debug"])

            dropout = tf.placeholder(tf.float32, (), name="dropout")
            self.set_feed("dropout", dropout)

        # process inputs into embeddings
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.vocabulary.size, self.hidden_sizes[0]],
                                        # dtype=tf.float32,
                                        initializer=scaled_init)
            inputs = tf.nn.embedding_lookup(embedding, inputs)

            # debugging fetches
            if self.visualize_embedded:
                self.set_fetch("embedded", inputs, "debug")
            if self.visualize_embeddings:
                self.set_fetch("embedding", embedding, "debug")

        # first dropout here
        if self.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # define lstm cell(s)
        cell_args = {}
        if self.cell_args is not None:
            cell_args.update(self.cell_args)
        if len(self.hidden_sizes) == 1:
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_sizes[0], **cell_args)
            if self.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        else:
            cells = []
            for hidden_size in self.hidden_sizes:
                cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, **cell_args)
                if self.keep_prob < 1:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        # prepare lstm state
        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # build unrolled layers
        inputs = tf.unstack(inputs, self.timesteps, 1)
        layer, state = tf.nn.static_rnn(cell, inputs, initial_state=initial_state)

        # NOTE - alternatively, manually build unrolled layers...
        # state = initial_state
        # outputs = []
        # for time_step in range(self.timesteps):
        #     if time_step > 0:
        #         tf.get_variable_scope().reuse_variables()
        #     (layer, state) = cell(inputs[:, time_step, :], state)
        #     outputs.append(layer)
        # layer = outputs

        # create output layer
        layer = tf.reshape(tf.concat(layer, 1), [-1, self.hidden_sizes[-1]])
        layer, info = self.add_fc(layer, self.hidden_sizes[-1], self.vocabulary.size,
                                  activation=tf.nn.softmax, initializer=scaled_init)

        # calculate prediction and accuracy
        with tf.name_scope('predict'):
            predict = tf.cast(tf.argmax(layer, axis=1), tf.int32)
            batched_predict = tf.reshape(predict, [self.batch_size, self.timesteps])
            self.set_fetch("predict", batched_predict, ["predict", "debug"])

            correct_prediction = tf.equal(predict, tf.reshape(outputs, [-1]))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.set_fetch("accuracy", accuracy, "evaluate")

        # calculate loss and cost
        with tf.name_scope('loss'):
            logits = tf.reshape(info["Z"], [self.batch_size, self.timesteps, self.vocabulary.size])
            weights = tf.ones([self.batch_size, self.timesteps])  # , dtype=self.input.dtype)
            sequence_loss = tf.contrib.seq2seq.sequence_loss(logits, outputs, weights,
                                                             average_across_timesteps=False,
                                                             average_across_batch=True)
            self.set_fetch("sequence_loss", sequence_loss, "evaluate")

            loss = tf.reduce_sum(sequence_loss)
            self.set_fetch("loss", loss, "evaluate")

        # remember final state
        self.final_state = state

    def prepare_default_feeds(self, graph, feed_map):
        feed_map["dropout"] = 1
        return feed_map

    def run(self, sess, graph, feed_map, **kwargs):
        results = super().run(sess, graph, feed_map, **kwargs)

        if graph == "debug":
            # visualize evaluated dataset results
            if self.supervised and self.rendering:
                self.update_visualize(feed_map, results)

        return results

    def init_visualize(self):
        if self.rendering:
            # cache the desired dims here
            if self.visualize_embeddings:
                self.max_embeddings = 40
                size = self.max_embeddings * self.hidden_size
                stride = self.hidden_size
            elif self.visualize_embedded:
                size = self.hidden_size * self.timesteps
                stride = self.timesteps
            else:
                self.viewer.set_size(512, 512)
                return
            cols = math.ceil(math.sqrt(size) / stride) * stride
            rows = math.ceil(size / cols)
            self.viewer.set_size(cols, rows)

    def update_visualize(self, feed_map, results):
        cols, rows = self.viewer.get_size()

        if self.visualize_embeddings:
            # render embeddings params
            values = results["embedding"][:self.max_embeddings]
            values = self.viewer.process_image(values, rows=rows, cols=cols)
            self.viewer.set_main_image(values)
        elif self.visualize_embedded:
            # render embedded representation of input
            values = results["embedded"]
            batch = 0
            values = values[batch]
            values = self.viewer.process_image(values, rows=rows, cols=cols)
            self.viewer.set_main_image(values)

        # show labels with targets/predictions
        num_labels = 1
        input = self.output.decode(results["X"])
        input_batch = self.vocabulary.decode(input[:num_labels])
        target = self.output.decode(results["Y"])
        target_batch = self.vocabulary.decode(target[:num_labels])
        predict = self.output.decode(results["predict"])
        predict_batch = self.vocabulary.decode(predict[:num_labels])
        # predict_batch = np.reshape(predict_batch, [num_labels, self.timesteps])
        for i in range(num_labels):
            input_seq = " ".join([str(x) for x in input_batch[i]])
            target_seq = " ".join([str(x) for x in target_batch[i]])
            predict_seq = " ".join([str(x) for x in predict_batch[i]])
            prediction_message = (f"INPUT:  {input_seq}\n"
                                  f"TARGET:  {target_seq}"
                                  f"\nPREDICT: {predict_seq}")
            self.viewer.add_label(f"prediction_{i}", prediction_message, width=cols,
                                  multiline=True, font_name="Courier New", font_size=12)
