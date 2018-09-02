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
    def __init__(self, dataset=None):
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
            return tf.placeholder(self.input.dtype, (None,) + self.input.shape, name="X")
        else:
            return self.inputs

    def get_outputs(self):
        if self.optimize_batch:
            return tf.placeholder(self.output.dtype, (None,) + self.output.shape, name="Y")
        else:
            return self.outputs

    def get_step_data(self):
        if self.optimize_batch:
            # return individual batches instead (HACK)
            # should iterate over batch count here, but rather just remove this param entirely.
            batch = self.get_batch()
            feed_map = {
                "X": batch.inputs,
                "Y": batch.outputs,
            }
            # dataset = tf.data.Dataset.from_tensor_slices(batch)
            return batch, feed_map
        else:
            # optimize on entire epoch
            return self, {}

    def get_batch(self):
        batch = Batch(dataset=self)
        batch.inputs = self.inputs[self.head:self.head + self.batch_size]
        batch.outputs = self.outputs[self.head:self.head + self.batch_size]

        batch.inputs = [self.input.encode(o) for o in batch.inputs]
        batch.outputs = [self.output.encode(o) for o in batch.outputs]

        self.head = (self.head + self.batch_size) % len(self.inputs)
        import ipdb; ipdb.set_trace()  # HACK DEBUGGING !!!
        return batch
