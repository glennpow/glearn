import gym
import numpy as np
from glearn.datasets.dataset import Dataset


class LabeledDataset(Dataset):
    def __init__(self, name, data, *args, label_names=None, one_hot=False, **kwargs):
        self.label_names = label_names
        self.one_hot = one_hot
        self.labels = range(self.label_count)

        if self.one_hot:
            eyes = np.eye(self.label_count)
            self.labels = eyes[self.labels]

            for mode, mode_data in data.items():
                targets = np.array(mode_data[1]).reshape(-1)
                data[mode] = (mode_data[0], eyes[targets])

            output_space = gym.spaces.MultiBinary(self.label_count)
        else:
            output_space = gym.spaces.Discrete(self.label_count)

        super().__init__(name, data, *args, output_space=output_space, **kwargs)

    @property
    def label_count(self):
        return len(self.label_names)

    def encipher_element(self, value):
        if self.one_hot:
            raise Exception("TODO")
        else:
            label = self.label_names.index(value)
        return super().encipher_element(label)

    def decipher_element(self, value):
        if self.one_hot:
            raise Exception("TODO")
        else:
            label = super().decipher_element(value)
        return self.label_names[label]
