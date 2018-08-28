import tensorflow as tf
from policies.interface import Interface


class Transition(object):
    def __init__(self, observation, action, reward, new_observation, done, info):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.new_observation = new_observation
        self.done = done
        self.info = info


class Batch(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.inputs = []
        self.outputs = []
        self.info = {}


def transition_batch(transitions):
    batch = Batch()
    batch.inputs = [e.observation for e in transitions]
    batch.outputs = [e.action for e in transitions]
    batch.info["transitions"] = transitions
    return batch


class Dataset(object):
    def __init__(self, name, inputs, outputs, input_space, output_space, batch_size,
                 optimize_batch=False, info={}):
        # self.epoch_size = len(inputs) // batch_size  # this is dependent on algorithm?
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.input = Interface(input_space)
        self.output = Interface(output_space)
        self.batch_size = batch_size
        self.optimize_batch = optimize_batch
        self.info = info

        self.reset()

    def reset(self):
        self.head = 0

    def get_inputs(self):
        if self.optimize_batch:
            # have to store this for later (HACK)
            self.inputs_feed = tf.placeholder(self.input.dtype, (None,) + self.input.shape,
                                              name="X")
            return self.inputs_feed
        else:
            return self.inputs

    def get_outputs(self):
        if self.optimize_batch:
            # have to store this for later (HACK)
            self.outputs_feed = tf.placeholder(self.output.dtype, (None,) + self.output.shape,
                                               name="Y")
            return self.outputs_feed
        else:
            return self.outputs

    def get_epoch(self):
        if self.optimize_batch:
            # return individual batches instead (HACK)
            # should iterate over batch count here, but rather just remove this param entirely.
            batch = self.get_batch()
            feed_dict = {
                self.inputs_feed: batch.inputs,
                self.outputs_feed: batch.outputs,
            }
            # dataset = tf.data.Dataset.from_tensor_slices(batch)
            return batch, feed_dict
        else:
            # optimize on entire epoch
            return self, {}

    def get_batch(self):
        batch = Batch(self)
        batch.inputs = self.inputs[self.head:self.head + self.batch_size]
        batch.outputs = self.outputs[self.head:self.head + self.batch_size]

        batch.inputs = [self.input.encode(o) for o in batch.inputs]
        batch.outputs = [self.output.encode(o) for o in batch.outputs]

        self.head = (self.head + self.batch_size) % len(self.inputs)
        return batch
