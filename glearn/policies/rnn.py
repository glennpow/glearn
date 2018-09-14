import math
import tensorflow as tf
from glearn.policies.policy import Policy


class RNNPolicy(Policy):
    def __init__(self, config, version=None):
        self.learning_rate = config.get("learning_rate", 1)
        self.lr_decay = config.get("lr_decay", .95)
        self.keep_prob = config.get("keep_prob", None)
        self.max_grad_norm = config.get("max_grad_norm", None)

        self.hidden_size = config.get("hidden_size", 128)
        self.hidden_depth = config.get("hidden_depth", 1)
        self.cell_args = config.get("cell_args", None)

        self.visualize_embeddings = False  # HACK - expose these?
        self.visualize_embedded = False  # HACK

        super().__init__(config, version=version)

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

            # TODO - add new graph "render" or "debug" for all this crap
            self.set_fetch("X", inputs, "evaluate")
            self.set_fetch("Y", outputs, "evaluate")

            learning_rate = tf.placeholder(tf.float32, (), name="lambda")
            self.set_feed("lambda", learning_rate, ["optimize", "evaluate"])

            dropout = tf.placeholder(tf.float32, (), name="dropout")
            self.set_feed("dropout", dropout)

        # process inputs into embeddings
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.vocabulary.size, self.hidden_size],
                                        # dtype=tf.float32,
                                        initializer=scaled_init)
            inputs = tf.nn.embedding_lookup(embedding, inputs)

            # TODO - add new graph "render" or "debug" for all this crap
            if self.visualize_embedded:
                self.set_fetch("embedded", inputs, "evaluate")
            if self.visualize_embeddings:
                self.set_fetch("embedding", embedding, "evaluate")

        # first dropout here
        if self.keep_prob is not None:
            inputs = tf.nn.dropout(inputs, dropout)

        # define lstm cell(s)
        cell_args = {}
        # cell_args.update(DEFAULT_LSTM_CELL_ARGS)
        if self.cell_args is not None:
            cell_args.update(self.cell_args)
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, **cell_args)
        if self.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if self.hidden_depth > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.hidden_depth)

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
        layer = tf.reshape(tf.concat(layer, 1), [-1, self.hidden_size])
        layer, info = self.add_fc(layer, self.hidden_size, self.vocabulary.size,
                                  activation=tf.nn.softmax, initializer=scaled_init)

        # calculate prediction and accuracy
        with tf.name_scope('predict'):
            predict = tf.cast(tf.argmax(layer, axis=1), tf.int32)
            batched_predict = tf.reshape(predict, [self.batch_size, self.timesteps])
            self.set_fetch("predict", batched_predict, ["predict", "evaluate"])

            correct_prediction = tf.equal(predict, tf.reshape(outputs, [-1]))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.set_fetch("accuracy", accuracy, "evaluate")

        # calculate loss and cost
        with tf.name_scope('loss'):
            logits = tf.reshape(info["Z"], [self.batch_size, self.timesteps, self.vocabulary.size])
            weights = tf.ones([self.batch_size, self.timesteps])  # , dtype=self.input.dtype)
            loss = tf.contrib.seq2seq.sequence_loss(logits, outputs, weights,
                                                    average_across_timesteps=False,
                                                    average_across_batch=True)
            self.set_fetch("loss", loss, "evaluate")

            cost = tf.reduce_sum(loss)
            self.set_fetch("cost", cost, "evaluate")

        # remember final state
        self.final_state = state

        # minimize loss
        with tf.name_scope('optimize'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            global_step = tf.train.get_or_create_global_step()

            if self.max_grad_norm is None:
                # apply unclipped gradients
                optimize = optimizer.minimize(cost, global_step=global_step)
            else:
                # apply gradients with clipping
                tvars = tf.trainable_variables()
                grads = tf.gradients(cost, tvars)
                grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                optimize = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            self.set_fetch("optimize", optimize, "optimize")

            self.set_fetch("lambda", learning_rate, "evaluate")

    def prepare_feed_map(self, graph, data, feed_map):
        if graph == "optimize" or graph == "evaluate":
            max_lr_step = 10
            lr_decay = self.lr_decay ** max(self.step + 1 - max_lr_step, 0.0)
            learning_rate = self.learning_rate * lr_decay
            feed_map["lambda"] = learning_rate
            feed_map["dropout"] = self.keep_prob
        else:
            feed_map["dropout"] = 1
        return feed_map

    def optimize(self, step):
        data = super().optimize(step)

        # visualize evaluated dataset results
        if self.supervised and self.evaluating:
            self.update_visualize(data)

    def init_viewer(self):
        super().init_viewer()

        self.viewer.set_label_spacing(20)

    def init_visualize(self):
        if self.viewer is not None:
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

    def update_visualize(self, data):
        cols, rows = self.get_viewer_size()

        if self.visualize_embeddings:
            # render embeddings params
            values = self.results["evaluate"]["embedding"][:self.max_embeddings]
            values = self.process_image(values, rows=rows, cols=cols)
            self.viewer.set_main_image(values)
        elif self.visualize_embedded:
            # render embedded representation of input
            values = self.results["evaluate"]["embedded"]
            batch = 0
            values = values[batch]
            values = self.process_image(values, rows=rows, cols=cols)
            self.viewer.set_main_image(values)

        # show labels with targets/predictions
        num_labels = 5
        input = self.output.decode(self.results["evaluate"]["X"])
        input_batch = self.vocabulary.decode(input[:num_labels])
        target = self.output.decode(self.results["evaluate"]["Y"])
        target_batch = self.vocabulary.decode(target[:num_labels])
        predict = self.output.decode(self.results["evaluate"]["predict"])
        predict_batch = self.vocabulary.decode(predict[:num_labels])
        # predict_batch = np.reshape(predict_batch, [num_labels, self.timesteps])
        for i in range(num_labels):
            input_seq = " ".join([str(x) for x in input_batch[i]])
            target_seq = " ".join([str(x) for x in target_batch[i]])
            predict_seq = " ".join([str(x) for x in predict_batch[i]])
            prediction_message = (f"INPUT:  {input_seq}\n"
                                  f"TARGET:  {target_seq}"
                                  f"\nPREDICT: {predict_seq}")
            self.add_label(f"prediction_{i}", prediction_message, width=cols, multiline=True,
                           font_name="Courier New", font_size=12)
