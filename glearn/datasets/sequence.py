import numpy as np
import tensorflow as tf
from gym.spaces import Box
from glearn.datasets.dataset import Dataset
from glearn.utils.log import log_warning
from glearn.utils.collections import is_collection


class Vocabulary(object):
    def __init__(self, words):
        self.words = words
        self.size = len(self.words)
        self.word_to_ids = dict(zip(words, range(len(words))))

    def encipher(self, value):
        if is_collection(value):
            # TODO omit Nones...
            return [self.encipher(subvalue) for subvalue in value]
        elif isinstance(value, str):
            return self.word_to_ids.get(value, None)
        else:
            log_warning(f"Unknown vocabulary word type: {value} ({type(value)})")
            return None

    def decipher(self, value):
        if is_collection(value):
            # TODO omit Nones...
            return [self.decipher(subvalue) for subvalue in value]
        elif isinstance(value, int) or np.isscalar(value):
            if value < self.size:
                return self.words[value]
        else:
            log_warning(f"Unknown vocabulary word ID type: {value} ({type(value)})")
            return None


class SequenceDataset(Dataset):
    def __init__(self, name, data, batch_size, vocabulary, timesteps):
        self.vocabulary = vocabulary

        tfdtype = tf.int32  # HACK - can we infer?
        npdtype = np.int32  # HACK ?

        producer_data = {}
        for mode, raw_data in data.items():
            # sequence producer
            with tf.variable_scope(name, values=[raw_data, batch_size, timesteps]):
                # tensor of data
                tensor_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tfdtype)
                # length of data tensor
                num_samples = tf.size(tensor_data)
                # number of batches in data
                num_batches = num_samples // batch_size

                # make sure data subdivides into batches evenly
                batched_data = tf.reshape(tensor_data[0: batch_size * num_batches],
                                          [batch_size, num_batches])

                # number of unrolled batches in an epoch
                epoch_size = (num_batches - 1) // timesteps
                assertion = tf.assert_positive(epoch_size,
                                               message="Decrease batch_size or timesteps")
                # assert valid data?
                with tf.control_dependencies([assertion]):
                    epoch_size = tf.identity(epoch_size, name="epoch_size")

                i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

                x = tf.strided_slice(batched_data, [0, i * timesteps],
                                     [batch_size, (i + 1) * timesteps])
                x.set_shape([batch_size, timesteps])

                y = tf.strided_slice(batched_data, [0, i * timesteps + 1],
                                     [batch_size, (i + 1) * timesteps + 1])
                y.set_shape([batch_size, timesteps])

                producer_data[mode] = (x, y)

        input_space = Box(low=0, high=vocabulary.size, shape=(timesteps, ), dtype=npdtype)
        output_space = Box(low=0, high=vocabulary.size, shape=(timesteps, ), dtype=npdtype)

        # calculate this outside TF
        epoch_size = {k: ((len(v) // batch_size) - 1) // timesteps for k, v in data.items()}

        super().__init__(name, producer_data, batch_size, input_space=input_space,
                         output_space=output_space, epoch_size=epoch_size, producer=True)

    def encipher_element(self, value):
        result = self.vocabulary.encipher(value)
        return super().encipher_element(result)

    def decipher_element(self, value):
        result = super().decipher_element(value)
        return self.vocabulary.decipher(result)
