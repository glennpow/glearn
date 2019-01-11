import numpy as np
import pickle
import os
import gym
import tensorflow as tf
from glearn.datasets.dataset import DatasetPartition, LabeledDataset
from glearn.utils.download import ensure_download
from glearn.utils.path import script_relpath


DATA_PATH = script_relpath("../../data/cifar10/")
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_PARENT = "cifar-10-batches-py"

IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
IMAGE_SIZE_FLAT = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS
IMAGE_CLASSES = 10

NUM_TRAINING_FILES = 1  # HACK: 5
IMAGES_PER_FILE = 10000
NUM_TRAINING_IMAGES = NUM_TRAINING_FILES * IMAGES_PER_FILE


def _get_file_path(filename=""):
    return os.path.join(DATA_PATH, DATA_PARENT, filename)


def _unpickle(filename):
    file_path = _get_file_path(filename)

    print("Loading CIFAR-10 data: " + file_path)

    # unpickle file
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE])

    # Reorder the indices of the array.  (why?)
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    # load the pickled data-file.
    data = _unpickle(filename)

    # convert the raw images.
    raw_images = data[b'data']
    images = _convert_images(raw_images)

    # convert the class-labels for each image.
    labels = np.array(data[b'labels'])
    labels = np.eye(IMAGE_CLASSES)[labels.reshape(-1)]

    return images, labels


def _ensure_download():
    ensure_download(url=DATA_URL, download_dir=DATA_PATH, extract=True)


def _load_label_names():
    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def cifar10_dataset(config):
    _ensure_download()

    return old_method(config)
    # return new_method(config)


def new_method(config):
    filenames = {
        "train": ["data_batch_" + str(i + 1) for i in range(NUM_TRAINING_FILES)],
        "test": ["test_batch"],
    }

    batch_size = config.get("batch_size", 128)
    label_names = _load_label_names()

    with tf.device('/cpu:0'):
        for name, (input_data, output_data, size) in filenames.items():
            inputs = tf.data.Dataset.from_tensor_slices(input_data)
            outputs = tf.data.Dataset.from_tensor_slices(output_data)

            partitions[name] = DatasetPartition(name, inputs, outputs, size, batch_size,
                                                output_space=output_space, shuffle=1000)

    return LabeledDataset(config, "CIFAR-10", partitions, label_names=label_names)


def old_method(config):
    data = {}

    images, labels = _load_data(filename="test_batch")
    data["test"] = (images, labels, IMAGES_PER_FILE)

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images_shape = [NUM_TRAINING_IMAGES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]
    images = np.zeros(shape=images_shape, dtype=float)
    labels_shape = [NUM_TRAINING_IMAGES, IMAGE_CLASSES]
    labels = np.zeros(shape=labels_shape, dtype=float)
    data["train"] = (images, labels, NUM_TRAINING_IMAGES)

    begin = 0
    for i in range(NUM_TRAINING_FILES):
        # Load the images and class-numbers from the data-file.
        images_batch, labels_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        labels[begin:end, :] = labels_batch
        begin = end

    output_space = gym.spaces.Discrete(IMAGE_CLASSES)

    batch_size = config.get("batch_size", 128)
    label_names = _load_label_names()

    with tf.device('/cpu:0'):
        partitions = {}
        for name, (input_data, output_data, size) in data.items():
            inputs = tf.data.Dataset.from_tensor_slices(input_data)
            outputs = tf.data.Dataset.from_tensor_slices(output_data)

            partitions[name] = DatasetPartition(name, inputs, outputs, size, batch_size,
                                                output_space=output_space, shuffle=1000)

    return LabeledDataset(config, "CIFAR-10", partitions, label_names=label_names)
