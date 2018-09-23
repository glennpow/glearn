from glearn.datasets.sequence import Vocabulary, SequenceDataset


class DigitRepeatDataset(SequenceDataset):
    def __init__(self, config, digits=10, repeat=100):
        batch_size = config.get("batch_size", 5)
        timesteps = config.get("timesteps", 1)

        data = []
        for i in range(repeat):
            for j in range(digits):
                for k in range(j):
                    data.append(j)

        vocabulary = Vocabulary(range(digits))

        super().__init__("Counter", data, vocabulary, batch_size, timesteps)
