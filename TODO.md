# for some inputs/vars/operations use --> with tf.device('/cpu:0'):
# fix everaging of evaluation summary results
# figure out TD and PolicyGradient inheritance
X replace GaussianLayer with CategoricalLayer
# gather multiple episodes (5+)
# empirical rewards (gamma ...)
X L2 penalty on tanh preactivation (HACK)
# how many minibatches per epoch
# fix save/load?  doesn't seem to work right
# rename Batch to Data, and subclass Dataset from it.  Also load datasets using definitions paradigm?
# transition buffer for RL should use tf, and return components as slices
# common logger functions
# prepare_feeds should just look at fetches, not graphs (requires callback from policy.run to trainer)

---
LSTM
       # create output layer  (TODO - just get this out of here...)
        y = self.dense(x, 1, vocabulary_size, dropout, tf.nn.softmax)

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

---
Dense
        # create output layer
        output_interface = outputs.interface
        i = len(self.hidden_sizes)
        if output_interface.discrete:
            y = self.dense(x, i, output_interface.size, dropout, tf.nn.softmax,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
        else:
            y = self.dense(x, i, output_interface.size, dropout, None,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)

        # loss evaluations
        with tf.name_scope('evaluate'):
            if output_interface.discrete:
                # evaluate discrete loss
                logits = self.references["Z"]
                neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=outputs)
                loss = tf.reduce_mean(neg_log_p)
                self.context.set_fetch("loss", loss, "evaluate")

                # evaluate accuracy
                correct = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                self.context.set_fetch("accuracy", accuracy, "evaluate")
            else:
                # evaluate continuous loss
                loss = tf.reduce_mean(tf.square(outputs - y))
                self.context.set_fetch("loss", loss, "evaluate")

                # evaluate accuracy
                accuracy = tf.exp(-loss)
                self.context.set_fetch("accuracy", accuracy, "evaluate")
        return y
