import numpy as np
import tensorflow as tf
import gym
from glearn.data.batch import Batch
from glearn.data.interface import Interface
from glearn.utils.file_cache import TEMP_DIR


class Dataset(object):
    def __init__(self, name, data, batch_size, input_space=None, output_space=None,
                 epoch_size=None, producer=False):
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
        self.total_samples = {}
        if epoch_size is None:
            self.epoch_size = {}
        else:
            self.epoch_size = epoch_size
        for k, v in data.items():
            if epoch_size is None:
                self.epoch_size[k] = len(data[k][0]) // batch_size
            self.total_samples[k] = self.epoch_size[k] * batch_size

        self.producer = producer
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

    def get_info(self):
        return {
            "Description": str(self),
            "Input": self.input,
            "Output": self.output,
        }

    @staticmethod
    def get_data_path(path):
        # return script_relpath(f"../../data/{path}/")
        return f"{TEMP_DIR}/data/{path}"

    def _format_modes(self, modes):
        return ", ".join([f"{k}:{v}" for k, v in modes.items()])

    def reset(self, mode="train"):
        self.heads[mode] = 0
        return self.get_epoch_size(mode=mode)

    def get_inputs(self, mode="train"):
        if self.producer:
            return self.data[mode][0]
        else:
            return tf.placeholder(self.input.dtype, (None,) + self.input.shape, name="X")

    def get_outputs(self, mode="train"):
        if self.producer:
            return self.data[mode][1]
        else:
            return tf.placeholder(self.output.dtype, (None,) + self.output.shape, name="Y")

    def get_epoch_size(self, mode="train"):
        if mode in self.epoch_size:
            return self.epoch_size[mode]
        return 0

    def get_batch(self, mode="train"):
        if self.producer:
            # the tensorflow producer will handle batching itself
            return self, {}
        else:
            # return individual batches instead of producer
            batch = self.build_batch(mode=mode)
            feed_map = {
                "X": batch.inputs,
                "Y": batch.outputs,
            }
            # dataset = tf.data.Dataset.from_tensor_slices(batch)
            return batch, feed_map

    def build_batch(self, mode="train"):
        inputs = self.data[mode][0]
        outputs = self.data[mode][1]

        # get batch head
        if mode not in self.heads:
            self.heads[mode] = 0
        head = self.heads[mode]

        # get slices of data
        batch = Batch(mode=mode)
        batch.dataset = self
        batch.inputs = inputs[head:head + self.batch_size]
        batch.outputs = outputs[head:head + self.batch_size]

        # move batch head
        head = (head + self.batch_size) % len(inputs)
        self.heads[mode] = head
        return batch

    def encipher(self, value):
        result = value

        # handle discrete values
        if self.output.discrete:
            discretized = np.zeros(self.output.shape)
            discretized[value] = 1
            result = discretized

        return result

    def decipher(self, value):
        result = value

        # handle discrete values
        if self.output.discrete:
            result = np.argmax(value)

        return result
