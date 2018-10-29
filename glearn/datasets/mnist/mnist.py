import gzip
import os
import shutil
import tempfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import gym
from glearn.datasets.dataset import Dataset


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))
        if rows != 28 or cols != 28:
            raise ValueError('Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                             (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
            tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def load_data(path, element_size, max_count=None, header_bytes=0, mapping=None):
    with open(path, "rb") as f:
        data = f.read()[header_bytes:]
        data = [data[i:i + element_size] for i in range(0, len(data), element_size)]
        if max_count is not None:
            data = data[:max_count]
        if mapping is not None:
            # HACK? (to get dtype.  is it fine?)
            # data = list(map(mapping, data))
            data = np.array(list(map(mapping, data)))
        return data


def dataset(images_file, labels_file, config):
    """Download and parse MNIST dataset."""
    directory = os.path.expanduser(os.path.join(os.path.dirname(__file__), "../../../data/mnist"))
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.astype(float)
        image = image.reshape((28, 28, 1))
        image = image / 255.0
        return image

    def decode_label(label):
        label = np.frombuffer(label, dtype=np.uint8)
        label = label.astype(int)
        label = label.flatten()
        return label

    batch_size = config.get("batch_size", 128)
    max_count = config.get("max_count", None)

    images = load_data(images_file, 28 * 28, max_count=max_count, header_bytes=16,
                       mapping=decode_image)
    images = np.reshape(images, [-1, 28, 28, 1])
    labels = load_data(labels_file, 1, max_count=max_count, header_bytes=8, mapping=decode_label)

    input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28, 28, 1), dtype=images.dtype)
    output_space = gym.spaces.Discrete(10)

    # TODO fix HACK - Producer like PTB, which iterates all batches in an epoch...

    return Dataset("MNIST", inputs=images, outputs=labels,
                   optimize_batch=True,  # HACK fix above
                   input_space=input_space, output_space=output_space,
                   batch_size=batch_size)


def train(config):
    """tf.data.Dataset object for MNIST training data."""
    return dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', config)


def test(config):
    """tf.data.Dataset object for MNIST test data."""
    return dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', config)
