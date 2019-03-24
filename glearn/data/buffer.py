import numpy as np


class Buffer(object):
    def __init__(self, mode=None, samples=None, size=None):
        self.mode = mode
        self.samples = samples
        self.size = size

        self._head = 0
        self._count = 0

    def __contains__(self, key):
        if not isinstance(key, str):
            raise KeyError()
        return key in self.samples

    def __getitem__(self, key):
        assert self.samples is not None
        if not isinstance(key, str):
            raise KeyError()
        return self.samples[key]

    def __setitem__(self, key, values):
        if not isinstance(key, str):
            raise KeyError()
        # TODO - pad this to size, if necessary...
        self.samples[key] = np.array(values)

    def __len__(self):
        return self.sample_count()

    def sample_count(self):
        if self.size is not None:
            return min(self._count, self.size)
        return self._count

    def empty(self):
        return self.samples is None or self._count == 0

    def clear(self):
        self._count = 0
        if self.size is None:
            self.samples = None
        else:
            self._head = 0

    def clip(self, size):
        if self.size is None:
            self.samples = {k: v[:size] for k, v in self.samples.items()}

    def _get_slice(self, count):
        return np.array([(self._head + i) % self.size for i in range(count)])

    def _add_samples(self, samples):
        # make sure sample counts are equal for all keys
        samples_counts = np.array([len(values) for _, values in samples.items()])
        samples_count = samples_counts[0]
        assert np.all(samples_counts == samples_count)
        self._count += samples_count

        # make sure sample buffers are initialized
        if self.samples is None:
            self.samples = {}
            for key, value in samples.items():
                value_shape = np.shape(value[0])
                if self.size is None:
                    # unbounded buffers
                    samples_shape = (0,) + value_shape
                else:
                    # bounded buffers
                    samples_shape = (self.size,) + value_shape
                self.samples[key] = np.zeros(samples_shape, dtype=np.float32)
        assert len(samples) == len(self.samples)

        for key, value in samples.items():
            # make sure samples are compatible with this buffer
            assert key in self.samples, f"Invalid buffer sample key: {key}"

            # prepare sample values shape
            sample_values = np.array(value)

            # add sample values for key
            if self.size is None:
                self.samples[key] = np.concatenate([self.samples[key], sample_values])
            else:
                idxs = self._get_slice(samples_count)
                self.samples[key][idxs] = sample_values
                self._head = (idxs[-1] + 1) % self.size

    def add_sample(self, sample):
        self._add_samples({k: np.array(v)[np.newaxis] for k, v in sample.items()})

    def add_samples(self, samples):
        self._add_samples({k: np.array(v) for k, v in samples.items()})

    def add_buffer(self, buffer):
        self.add_samples(buffer.samples)

    def prepare_feeds(self):
        # override
        raise NotImplementedError()
