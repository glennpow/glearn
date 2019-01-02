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


class Dataset(object):
    def __init__(self, config, name, data, batch_size, input_space=None, output_space=None,
                 epoch_size=None, shuffle=None):
        self.config = config

        self.name = name
        self.data = data
        self.batch_size = batch_size
        self.current_partition = None

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

        # tf.data.Dataset integration
        partitions = list(data.keys())
        self.datasets = {}
        self.iterator = None
        self.handles = {}
        self.initializers = {}
        with tf.device('/cpu:0'):
            for partition in partitions:
                if partition in data:
                    input_dataset = tf.data.Dataset.from_tensor_slices(data[partition][0])
                    output_dataset = tf.data.Dataset.from_tensor_slices(data[partition][1])
                    zip_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))

                    if shuffle is not None:
                        zip_dataset = zip_dataset.shuffle(shuffle)

                    zip_dataset = zip_dataset.repeat().batch(batch_size)
                    # zip_dataset.prefetch(prefetch_batch_size)  # TODO
                    self.datasets[partition] = zip_dataset

                    if self.iterator is None:
                        self.handle_feed = tf.placeholder(tf.string, shape=[])
                        output_types = zip_dataset.output_types
                        output_shapes = zip_dataset.output_shapes
                        self.iterator = tf.data.Iterator.from_string_handle(self.handle_feed,
                                                                            output_types,
                                                                            output_shapes)
                        self.next_element = self.iterator.get_next()

                    partition_iterator = zip_dataset.make_initializable_iterator()
                    self.handles[partition] = self.sess.run(partition_iterator.string_handle())
                    self.initializers[partition] = partition_iterator.make_initializer(zip_dataset)

    def __str__(self):
        total_samples = self._format_partitions(self.total_samples)
        epoch_size = self._format_partitions(self.epoch_size)
        properties = [
            self.name,
            f"total=[{total_samples}]",
            f"batches=[{epoch_size}]x{self.batch_size}",
        ]
        return f"Dataset({', '.join(properties)})"

    @property
    def sess(self):
        return self.config.sess

    def get_info(self):
        return {
            "Description": str(self),
            "Input": self.input,
            "Output": self.output,
        }

    def _format_partitions(self, partitions):
        return ", ".join([f"{k}:{v}" for k, v in partitions.items()])

    def initialize(self, partition="train"):
        from glearn.utils.memory import print_virtual_memory, print_gpu_memory
        print_virtual_memory(f"before init {partition}")
        print_gpu_memory(f"before init {partition}")

        self.sess.run(self.initializers[partition])

        print_virtual_memory(f"after init {partition}")
        print_gpu_memory(f"after init {partition}")

        self.current_partition = partition
        return self.get_epoch_size(partition=partition)

    def get_inputs(self):
        return self.next_element[0]

    def get_outputs(self):
        return self.next_element[1]

    def get_epoch_size(self, partition="train"):
        if partition in self.epoch_size:
            return self.epoch_size[partition]
        return 0

    def get_batch(self):
        return self, {self.handle_feed: self.handles[self.current_partition]}

        # # encode data through the interfaces (FIXME - needed?)
        # batch.inputs = [self.input.encode(o) for o in batch.inputs]
        # batch.outputs = [self.output.encode(o) for o in batch.outputs]

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
