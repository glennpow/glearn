import numpy as np


class Batch(object):
    def __init__(self, mode=None, samples=None):
        self.mode = mode
        self.samples = samples

    def __getitem__(self, indices):
        assert isinstance(indices, str)
        return self.samples.get(indices, None)

    def __iter__(self):
        return self

    def __len__(self):
        return self.sample_count()

    def add_sample(self, sample):
        if self.samples is None:
            self.samples = {key: [] for key, _ in sample.items()}
        assert len(sample) == len(self.samples)

        for key, value in sample.items():
            assert key in self.samples, f"Invalid batch sample key: {key}"
            self.samples[key].append(value)

    def add_samples(self, samples):
        samples_counts = np.array([len(values) for _, values in samples.items()])
        assert np.all(samples_counts == samples_counts[0])

        if self.samples is None:
            self.samples = {key: [] for key, _ in samples.items()}
        assert len(samples) == len(self.samples)

        for key, values in samples.items():
            assert key in self.samples, f"Invalid batch sample key: {key}"
            self.samples[key] += values

    def add_batch(self, batch):
        self.add_samples(batch.samples)

    def sample_count(self):
        if len(self.samples) > 0:
            return len(self.samples[list(self.samples.keys())[0]])
        return 0

    def prepare_feeds(self):
        # override
        raise NotImplementedError()
