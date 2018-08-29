import collections
import os
import numpy as np
import tensorflow as tf
from gym.spaces import Box
from datasets.dataset import Dataset
from datasets.vocabulary import Vocabulary


raw_data = None


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    return Vocabulary(words)


def _file_to_word_ids(filename, vocabulary):
    data = _read_words(filename)
    # return [vocabulary.get_word_id(word) for word in data if word in vocabulary.words]
    return vocabulary.encode(data)


def ptb_raw_data(data_path):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    vocabulary = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, vocabulary)
    valid_data = _file_to_word_ids(valid_path, vocabulary)
    test_data = _file_to_word_ids(test_path, vocabulary)
    return train_data, valid_data, test_data, vocabulary


def build_dataset(data_path, index, batch_size, timesteps):
    global raw_data
    if raw_data is None:
        raw_data = ptb_raw_data(data_path)

    inputs, outputs = ptb_producer(raw_data[index], batch_size, timesteps)
    vocabulary = raw_data[3]
    input_space = Box(low=0, high=vocabulary.size, shape=(timesteps, ), dtype=np.int32)
    output_space = Box(low=0, high=vocabulary.size, shape=(timesteps, ), dtype=np.int32)
    info = {"vocabulary": vocabulary}

    return Dataset("PTB", inputs=inputs, outputs=outputs, input_space=input_space,
                   output_space=output_space, batch_size=batch_size, info=info)


def train(data_path, batch_size, timesteps):
    return build_dataset(data_path, 0, batch_size, timesteps)


def validate(data_path, batch_size, timesteps):
    return build_dataset(data_path, 1, batch_size, timesteps)


def test(data_path, batch_size, timesteps):
    return build_dataset(data_path, 2, batch_size, timesteps)


def ptb_producer(raw_data, batch_size, timesteps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, timesteps]):
        # tensor of data
        tensor_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        # length of data tensor
        num_samples = tf.size(tensor_data)
        # number of batches in data
        num_batches = num_samples // batch_size

        # make sure data subdivides into batches evenly
        data = tf.reshape(tensor_data[0: batch_size * num_batches],
                          [batch_size, num_batches])

        # number of unrolled batches in an epoch
        epoch_size = (num_batches - 1) // timesteps
        assertion = tf.assert_positive(epoch_size,
                                       message="epoch_size == 0, decrease batch_size or timesteps")
        # assert valid data?
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * timesteps], [batch_size, (i + 1) * timesteps])
        x.set_shape([batch_size, timesteps])
        y = tf.strided_slice(data, [0, i * timesteps + 1], [batch_size, (i + 1) * timesteps + 1])
        y.set_shape([batch_size, timesteps])
        return x, y
