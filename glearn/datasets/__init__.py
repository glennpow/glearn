from .mnist import mnist_dataset
from .cifar10 import cifar10_dataset
from .ptb import ptb_dataset
from .digit_repeat import DigitRepeatDataset as digit_repeat_dataset


def load_dataset(config, mode="train"):
    name = config.get("dataset", None)

    if name == "mnist":
        return mnist_dataset(config, mode=mode)
    if name == "cifar10":
        return cifar10_dataset(config, mode=mode)
    if name == "ptb":
        return ptb_dataset(config, mode=mode)
    if name == "digit_repeat":
        return digit_repeat_dataset(config, mode=mode)
    return None
