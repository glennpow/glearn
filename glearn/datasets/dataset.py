import numpy as np
import tensorflow as tf
import gym
from glearn.policies.interface import Interface


class Transition(object):
    def __init__(self, observation, action, reward, new_observation, done, info):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.new_observation = new_observation
        self.done = done
        self.info = info


class Batch(object):
    def __init__(self, dataset=None, mode="train"):
        self.dataset = dataset
        self.mode = mode
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
    def __init__(self, name, data, batch_size, input_space=None, output_space=None,
                 epoch_size=None, optimize_batch=False):
        self.name = name
        self.data = data
        self.batch_size = batch_size

        if input_space is None:
            inputs = data["train"][0]
            input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.shape(inputs)[1:])
        if output_space is None:
            outputs = data["train"][1]
            output_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.shape(outputs)[1:])
        self.input = Interface(input_space)
        self.output = Interface(output_space)

        # this should be provided, otherwise it is inferred from inputs
        self.epoch_size = {}
        self.total_samples = {}
        for k, v in data.items():
            if epoch_size is None:
                self.epoch_size[k] = len(data[k][0]) // batch_size
            else:
                self.epoch_size[k] = epoch_size
            self.total_samples[k] = self.epoch_size[k] * batch_size

        self.optimize_batch = optimize_batch
        self.heads = {}

        self.reset()

    def __str__(self):
        total_samples = self._format_modes(self.total_samples)
        epoch_size = self._format_modes(self.epoch_size)
        properties = [
            self.name,
            f"total=[{total_samples}]",
            f"batches=[{epoch_size}]x{self.batch_size}",
        ]
        return f"Dataset({', '.join(properties)})"

    def _format_modes(self, modes):
        return ", ".join([f"{k}:{v}" for k, v in modes.items()])

    def reset(self, mode="train"):
        self.heads[mode] = 0
        return self.get_epoch_size(mode=mode)

    def get_inputs(self, mode="train"):
        if self.optimize_batch:
            return tf.placeholder(self.input.dtype, (None,) + self.input.shape, name="X")
        else:
            # FIXME - this doesn't work for test/validation
            return self.data[mode][0]

    def get_outputs(self, mode="train"):
        if self.optimize_batch:
            return tf.placeholder(self.output.dtype, (None,) + self.output.shape, name="Y")
        else:
            # FIXME - this producer method doesn't work for test/validation modes
            return self.data[mode][1]

    def get_epoch_size(self, mode="train"):
        if mode in self.epoch_size:
            return self.epoch_size[mode]
        return 0

    def get_batch(self, mode="train"):
        if self.optimize_batch:
            # return individual batches instead (HACK)
            # should iterate over batch count here, but rather just remove this param entirely.
            batch = self.build_batch(mode=mode)
            feed_map = {
                "X": batch.inputs,
                "Y": batch.outputs,
            }
            # dataset = tf.data.Dataset.from_tensor_slices(batch)
            return batch, feed_map
        else:
            # the tensorflow producer will handle batching itself
            return self, {}

    def build_batch(self, mode="train"):
        inputs = self.data[mode][0]
        outputs = self.data[mode][1]

        # get batch head
        if mode not in self.heads:
            self.heads[mode] = 0
        head = self.heads[mode]

        # get slices of data
        batch = Batch(dataset=self, mode=mode)
        batch.inputs = inputs[head:head + self.batch_size]
        batch.outputs = outputs[head:head + self.batch_size]

        # encode data through the interfaces
        batch.inputs = [self.input.encode(o) for o in batch.inputs]
        batch.outputs = [self.output.encode(o) for o in batch.outputs]

        # move batch head
        head = (head + self.batch_size) % len(inputs)
        self.heads[mode] = head
        return batch

    def encipher(self, value):
        return self.output.encode(value)

    def decipher(self, value):
        return self.output.decode(value)


class LabeledDataset(Dataset):
    def __init__(self, *args, label_names=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_names = label_names

    def encipher(self, value):
        label = self.label_names.index(value)
        return self.output.encode(label)

    def decipher(self, value):
        label = self.output.decode(value)
        return self.label_names[label]
