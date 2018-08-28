import math
import tensorflow as tf
from policies.policy import Policy
from policies.layers import add_fc


DEFAULT_LSTM_CELL_ARGS = {
    "forget_bias": 1,
}


class RNN(Policy):
    def __init__(self, data_type=tf.float32, keep_prob=0.8, max_grad_norm=5,
                 learning_rate=2e-4, hidden_size=128, hidden_depth=1, cell_args=None,
                 **kwargs):
        self.data_type = data_type
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm
        self.learning_rate = .98  # learning_rate  # lamdba λ
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.cell_args = cell_args

        kwargs["multithreaded"] = True  # TODO - figure this out from the dataset

        super().__init__(**kwargs)

        self.init_visualize()

    def init_model(self):
        # infer values from dataset
        self.timesteps = self.input.shape[0]  # sequence_length
        self.vocabulary = self.dataset.info["vocabulary"]

        # create placeholders
        with tf.name_scope('inputs'):
            inputs, outputs = self.create_inputs()
            self.dropout = tf.placeholder(tf.float32, (), name="dropout")

        # process inputs into embeddings
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.vocabulary, self.hidden_size],
                                        dtype=self.data_type)
            inputs = tf.nn.embedding_lookup(embedding, inputs)
            self.evaluate_graph["embedding"] = embedding

        # first dropout here
        if self.keep_prob is not None:
            inputs = tf.nn.dropout(inputs, self.dropout)

        # define lstm cell(s)
        cell_args = self.cell_args or DEFAULT_LSTM_CELL_ARGS
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, **cell_args)
        if self.hidden_depth <= 1:
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
        # layer, info = add_fc(self, layer, self.hidden_size, self.output.size,
        layer, info = add_fc(self, layer, self.hidden_size, self.vocabulary,
                             activation=tf.nn.softmax)

        # store prediction
        with tf.name_scope('predict'):
            predict = tf.cast(tf.argmax(layer, axis=1), tf.int32)
            self.act_graph["predict"] = predict

            correct_prediction = tf.equal(predict, tf.reshape(outputs, [-1]))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.evaluate_graph["accuracy"] = accuracy

        # calculate loss and cost
        with tf.name_scope('loss'):
            logits = tf.reshape(info["Z"], [self.batch_size, self.timesteps, self.vocabulary])
            weights = tf.ones([self.batch_size, self.timesteps], dtype=self.data_type)
            loss = tf.contrib.seq2seq.sequence_loss(logits, outputs, weights,
                                                    average_across_timesteps=False,
                                                    average_across_batch=True)
            self.evaluate_graph["loss"] = loss

            cost = tf.reduce_sum(loss)
            self.evaluate_graph["cost"] = cost

        # record final state
        self.final_state = state

        # minimize loss
        with tf.name_scope('optimize'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            # self.optimize_graph["optimize"] = optimizer.minimize(loss)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)
            global_step = tf.train.get_or_create_global_step()
            optimize = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            self.optimize_graph["optimize"] = optimize

    def init_visualize(self):
        # TODO cache the desired dims here
        pass

    def update_visualize(self, data):
        # TODO cache the desired dims above
        # 1024x1250
        size = self.vocabulary * self.hidden_size
        cols = (math.ceil(math.sqrt(size)) // 128) * 128
        rows = math.ceil(size / cols)

        # image = data.inputs[index] * 255

        values = self.process_image(self.evaluate_result["embedding"], rows=rows, cols=cols)
        self.viewer.set_main_image(values)

        # action = self.output.decode(self.evaluate_result["act"][index])
        # action_message = f"{action}"
        # self.add_label("action", action_message)

    def act_feed(self, observation, feed_dict):
        # no dropout for inference
        feed_dict[self.dropout] = 1

        return feed_dict

    def optimize_feed(self, data, feed_dict):
        feed_dict[self.dropout] = self.keep_prob

        return feed_dict

    def optimize(self, evaluating=False, saving=True):
        data = super().optimize(evaluating=evaluating, saving=saving)

        # visualize evaluated dataset results
        if self.supervised and evaluating:
            self.update_visualize(data)
