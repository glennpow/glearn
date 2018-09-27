from .mnist.mnist import train as mnist_dataset
from .ptb.ptb import train as ptb_dataset
from .digit_repeat import DigitRepeatDataset as digit_repeat_dataset


def load_dataset(config):
    name = config.get("dataset", None)

    if name == "mnist":
        return mnist_dataset(config)
    if name == "ptb":
        return ptb_dataset(config)
    if name == "digit_repeat":
        return digit_repeat_dataset(config)
    return None
