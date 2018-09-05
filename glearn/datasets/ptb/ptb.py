import collections
import os
import tensorflow as tf
from glearn.datasets.sequence import Vocabulary, SequenceDataset


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


def ptb_raw_data():
    data_path = os.path.dirname(__file__)
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    vocabulary = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, vocabulary)
    valid_data = _file_to_word_ids(valid_path, vocabulary)
    test_data = _file_to_word_ids(test_path, vocabulary)
    return train_data, valid_data, test_data, vocabulary


def build_dataset(index, config):
    global raw_data
    if raw_data is None:
        raw_data = ptb_raw_data()

    batch_size = config.get("batch_size", 20)
    timesteps = config.get("timesteps", 35)

    return SequenceDataset("PTB", raw_data[index], raw_data[3], batch_size, timesteps)


def train(config):
    return build_dataset(0, config)


def validate(config):
    return build_dataset(1, config)


def test(config):
    return build_dataset(2, config)
