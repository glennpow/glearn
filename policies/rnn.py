import math
import numpy as np
import tensorflow as tf
from policies.policy import Policy
from policies.layers import add_fc


DEFAULT_LSTM_CELL_ARGS = {
    "forget_bias": 1,
}


class RNN(Policy):
    def __init__(self, data_type=tf.float32, keep_prob=0.5, max_grad_norm=5,
                 learning_rate=1, lr_decay=0.93, hidden_size=650, hidden_depth=1, cell_args=None,
                 **kwargs):
        self.data_type = data_type
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate  # lamdba λ
        self.lr_decay = lr_decay
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.cell_args = cell_args

        self.visualize_embeddings = False  # HACK - expose this?

        kwargs["multithreaded"] = True  # TODO - figure this out from the dataset

        super().__init__(**kwargs)

        self.init_visualize()

    def init_model(self):
        # infer values from dataset
        self.timesteps = self.input.shape[0]  # sequence_length
        self.vocabulary = self.dataset.info["vocabulary"]

        # default initializer
        init_scale = 0.05  # TODO - expose
        scaled_init = tf.random_uniform_initializer(-init_scale, init_scale)

        # create placeholders
        with tf.name_scope('inputs'):
            inputs, outputs = self.create_inputs()

            self.feeds["lambda"] = tf.placeholder(tf.float32, (), name="lambda")
            self.evaluate_graph["lambda"] = self.feeds["lambda"]
            self.feeds["dropout"] = tf.placeholder(tf.float32, (), name="dropout")

        # process inputs into embeddings
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.vocabulary.size, self.hidden_size],
                                        dtype=self.data_type,
                                        initializer=scaled_init)
            inputs = tf.nn.embedding_lookup(embedding, inputs)
            self.evaluate_graph["embedding"] = embedding

        # first dropout here
        if self.keep_prob is not None:
            inputs = tf.nn.dropout(inputs, self.feeds["dropout"])

        # define lstm cell(s)
        cell_args = {}
        cell_args.update(DEFAULT_LSTM_CELL_ARGS)
        if self.cell_args is not None:
            cell_args.update(self.cell_args)
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, **cell_args)
        if self.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.feeds["dropout"])
        if self.hidden_depth > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.hidden_depth)

        # prepare lstm state
        initial_state = cell.zero_state(self.batch_size, dtype=self.data_type)

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

        # create output layer
        layer = tf.reshape(tf.concat(layer, 1), [-1, self.hidden_size])
        layer, info = add_fc(self, layer, self.hidden_size, self.vocabulary.size,
                             activation=tf.nn.softmax, initializer=scaled_init)

        # calculate prediction and accuracy
        with tf.name_scope('predict'):
            predict = tf.cast(tf.argmax(layer, axis=1), tf.int32)
            self.act_graph["predict"] = predict
            self.evaluate_graph["predict"] = predict
            self.evaluate_graph["target"] = outputs

            correct_prediction = tf.equal(predict, tf.reshape(outputs, [-1]))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.evaluate_graph["accuracy"] = accuracy

        # calculate loss and cost
        with tf.name_scope('loss'):
            logits = tf.reshape(info["Z"], [self.batch_size, self.timesteps, self.vocabulary.size])
            weights = tf.ones([self.batch_size, self.timesteps], dtype=self.data_type)
            loss = tf.contrib.seq2seq.sequence_loss(logits, outputs, weights,
                                                    average_across_timesteps=False,
                                                    average_across_batch=True)
            self.evaluate_graph["loss"] = loss

            cost = tf.reduce_sum(loss)
            self.evaluate_graph["cost"] = cost

        # remember final state
        self.final_state = state

        # minimize loss
        with tf.name_scope('optimize'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.feeds["lambda"])
            # self.optimize_graph["optimize"] = optimizer.minimize(loss)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)
            global_step = tf.train.get_or_create_global_step()
            optimize = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            self.optimize_graph["optimize"] = optimize

    def init_viewer(self):
        super().init_viewer()

        self.viewer.set_zoom(1)
        self.viewer.set_label_spacing(20)

    def init_visualize(self):
        if self.viewer is not None:
            # cache the desired dims here
            size = self.vocabulary.size * self.hidden_size
            cols = math.ceil(math.sqrt(size) / 128.0) * 128
            rows = math.ceil(size / cols)
            self.viewer.set_size(cols, rows)

    def update_visualize(self, data):
        cols, rows = self.get_viewer_size()

        # render embeddings
        if self.visualize_embeddings:
            values = self.process_image(self.evaluate_result["embedding"], rows=rows, cols=cols)
            self.viewer.set_main_image(values)

        # show labels with targets/predictions
        num_labels = 5
        target = self.output.decode(self.evaluate_result["target"])
        target_batch = self.vocabulary.decode(target[:num_labels])
        predict = self.output.decode(self.evaluate_result["predict"])
        predict_batch = self.vocabulary.decode(predict[:self.timesteps * num_labels])
        predict_batch = np.reshape(predict_batch, [num_labels, self.timesteps])
        for i in range(num_labels):
            target_seq = " ".join(target_batch[i])
            predict_seq = " ".join(predict_batch[i])
            prediction_message = f"TARGET:  {target_seq}\n\nPREDICT: {predict_seq}"
            self.add_label(f"prediction_{i}", prediction_message, width=cols, multiline=True,
                           font_name="Courier New", font_size=12)

    def act_feed(self, observation, feed_dict):
        # no dropout for inference
        feed_dict[self.feeds["dropout"]] = 1

        return feed_dict

    def optimize_feed(self, epoch, data, feed_dict):
        max_lr_epoch = 10
        lr_decay = self.lr_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
        learning_rate = self.learning_rate * lr_decay
        feed_dict[self.feeds["lambda"]] = learning_rate
        feed_dict[self.feeds["dropout"]] = self.keep_prob

        return feed_dict

    def optimize(self, epoch, evaluating=False, saving=True):
        data = super().optimize(epoch, evaluating=evaluating, saving=saving)

        # visualize evaluated dataset results
        if self.supervised and evaluating:
            self.update_visualize(data)
