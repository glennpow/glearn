import numpy as np
import pickle
import os
import gym
from glearn.datasets.dataset import Dataset
from glearn.utils.download import maybe_download_and_extract
from glearn.utils.path import script_relpath


DATA_PATH = script_relpath("../../data/cifar10/")
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
IMAGE_SIZE_FLAT = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS
IMAGE_CLASSES = 10

NUM_TRAINING_FILES = 5
IMAGES_PER_FILE = 10000
NUM_TRAINING_IMAGES = NUM_TRAINING_FILES * IMAGES_PER_FILE


def _get_file_path(filename=""):
    return os.path.join(DATA_PATH, filename)


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
    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-labels for each image.
    labels = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, labels


def _maybe_download_and_extract():
    maybe_download_and_extract(url=DATA_URL, download_dir=DATA_PATH)


def _load_class_names():
    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def _one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def _build_dataset(config, images, labels, one_hot=False):
    inputs = images
    outputs = labels
    if one_hot:
        outputs = _one_hot_encoded(class_numbers=outputs, num_classes=IMAGE_CLASSES)

    input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.shape(inputs)[1:])
    output_space = gym.spaces.Discrete(IMAGE_CLASSES)

    batch_size = config.get("batch_size", 128)
    names = _load_class_names()

    # TODO fix HACK - Producer like PTB, which iterates all batches in an epoch (optimize_batch)...
    return Dataset("CIFAR-10", inputs, outputs, input_space, output_space, batch_size,
                   epoch_size=None, optimize_batch=True, info={"class_names": names})


def train(config):
    _maybe_download_and_extract()

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images_shape = [NUM_TRAINING_IMAGES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]
    images = np.zeros(shape=images_shape, dtype=float)
    labels_shape = [NUM_TRAINING_IMAGES]
    labels = np.zeros(shape=labels_shape, dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(NUM_TRAINING_FILES):
        # Load the images and class-numbers from the data-file.
        images_batch, labels_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        labels[begin:end] = labels_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return _build_dataset(config, images, labels)


def test(config):
    _maybe_download_and_extract()

    images, labels = _load_data(filename="test_batch")

    return _build_dataset(config, images, labels)
