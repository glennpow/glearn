import numpy as np
from glearn.datasets.sequence import Vocabulary, SequenceDataset


class DigitRepeatDataset(SequenceDataset):
    def __init__(self, config, mode="train", digits=10, repeat=100):
        batch_size = config.get("batch_size", 5)
        timesteps = config.get("timesteps", 1)

        modes = ["train", "test"]
        data = {mode: self._generate_data(digits, repeat) for mode in modes}

        vocabulary = Vocabulary(range(digits))

        super().__init__("Counter", data, batch_size, vocabulary, timesteps)

    def _generate_data(self, digits, repeat):
        data = []
        for i in range(repeat):
            for j in range(digits):
                for k in range(j):
                    data.append(j)
        return np.array(data, np.int32)
