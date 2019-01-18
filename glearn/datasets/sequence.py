from collections import abc
import numpy as np
import tensorflow as tf
from gym.spaces import Box
from glearn.datasets.dataset import Dataset


class Vocabulary(object):
    def __init__(self, words):
        self.words = words
        self.size = len(self.words)
        self.word_to_ids = dict(zip(words, range(len(words))))

    def encipher(self, word):
        if isinstance(word, str):
            return self.word_to_ids.get(word, None)
        elif isinstance(word, abc.Iterable):
            # TODO omit Nones...
            return [self.encipher(w) for w in word]
        else:
            print(f"Unknown vocabulary word type: {word} ({type(word)})")
            return None

    def decipher(self, id):
        if isinstance(id, int) or np.isscalar(id):
            if id < self.size:
                return self.words[id]
        elif isinstance(id, abc.Iterable):
            # TODO omit Nones...
            return [self.decipher(i) for i in id]
        else:
            print(f"Unknown vocabulary word ID type: {id} ({type(id)})")
            return None


class SequenceDataset(Dataset):
    def __init__(self, config, name, data, batch_size, vocabulary, timesteps):
        self.vocabulary = vocabulary

        tfdtype = tf.int32  # HACK - can we infer?
        npdtype = np.int32  # HACK ?

        producer_data = {}
        for mode, raw_data in data.items():
            # sequence producer
            with tf.name_scope(name, values=[raw_data, batch_size, timesteps]):
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

        super().__init__(config, name, producer_data, batch_size, input_space=input_space,
                         output_space=output_space, epoch_size=epoch_size, producer=True)

    def encipher(self, value):
        label = self.vocabulary.encipher(value)
        return super().encipher(label)

    def decipher(self, value):
        label = super.decipher(value)
        return self.vocabulary.decipher(label)
