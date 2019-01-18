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
    def __init__(self, dataset=None, partition="train"):
        self.dataset = dataset
        self.partition = partition
        self.inputs = []
        self.outputs = []
        self.info = {}


def transition_batch(transitions):
    batch = Batch()
    batch.inputs = [e.observation for e in transitions]
    batch.outputs = [e.action for e in transitions]
    batch.info["transitions"] = transitions
    return batch


class DatasetPartition(object):
    def __init__(self, name, inputs, outputs, size, batch_size,
                 input_space=None, output_space=None, shuffle=None):
        self.name = name
        self.size = size  # can I get this automatically?
        self.batch_size = batch_size

        self.epoch_size = size // batch_size
        self.available = self.epoch_size * self.batch_size

        # determine input/output interfaces
        if input_space is None:
            input_shape = inputs.output_shapes
            input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=input_shape)
        self.input = Interface(input_space)
        if output_space is None:
            output_shape = inputs.output_shapes
            output_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=output_shape)
        self.output = Interface(output_space)

        with tf.device('/cpu:0'):
            data = tf.data.Dataset.zip((inputs, outputs))

            if shuffle is not None:
                data = data.shuffle(shuffle)

            data = data.repeat().batch(batch_size)

            # data.prefetch(prefetch_batch_size)  # TODO

            self.data = data

    def __str__(self):
        properties = [
            f"'{self.name}'",
            f"total=[{self.available}]",
            f"batches=[{self.epoch_size}]x{self.batch_size}",
        ]
        return f"[{', '.join(properties)}]"

    def build(self, dataset):
        with tf.device('/cpu:0'):
            partition_iterator = self.data.make_initializable_iterator()
            self.handle = dataset.sess.run(partition_iterator.string_handle())
            self.initializer = partition_iterator.make_initializer(self.data)


class Dataset(object):
    def __init__(self, config, name, partitions):
        self.config = config

        self.name = name

        # prepare partitions
        self.partitions = partitions
        self.partition_names = list(partitions.keys())
        self.primary_partition = self.partitions[self.partition_names[0]]
        self.current_partition = None
        with tf.device('/cpu:0'):
            data = self.primary_partition.data

            self.handle_feed = tf.placeholder(tf.string, shape=[])
            output_types = data.output_types
            output_shapes = data.output_shapes
            self.iterator = tf.data.Iterator.from_string_handle(self.handle_feed,
                                                                output_types,
                                                                output_shapes)
            self.next_element = self.iterator.get_next()

        for partition in partitions.values():
            partition.build(self)

    def __str__(self):
        properties = [
            self.name,
        ] + [str(v) for v in self.partitions.values()]
        return f"Dataset({', '.join(properties)})"

    @property
    def input(self):
        return self.primary_partition.input

    @property
    def output(self):
        return self.primary_partition.output

    @property
    def sess(self):
        return self.config.sess

    def get_info(self):
        return {
            "Description": str(self),
            "Input": self.input,
            "Output": self.output,
        }

<<<<<<< HEAD
    def _get_partition(self, partition):
        if partition in self.partitions:
            return self.partitions[partition]
        raise Exception(f"Unknown dataset partition: '{partition}'")

    def initialize(self, partition="train"):
        self.current_partition = self._get_partition(partition)
        self.sess.run(self.current_partition.initializer)
        return self.get_epoch_size(partition=partition)

    def get_inputs(self):
        return self.next_element[0]

    def get_outputs(self):
        return self.next_element[1]

    def get_epoch_size(self, partition="train"):
        return self._get_partition(partition).epoch_size

    def get_batch(self):
        return self, {self.handle_feed: self.current_partition.handle}

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


class LabeledDataset(Dataset):
    def __init__(self, *args, label_names=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_names = label_names

    def encipher(self, value):
        label = self.label_names.index(value)
        return super().encipher(label)

    def decipher(self, value):
        label = super().decipher(value)
        return self.label_names[label]
