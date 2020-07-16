import numpy as np
from glearn.utils.log import log_warning


class Buffer(object):
    def __init__(self, mode=None, samples=None, size=None, circular=False):
        self.mode = mode
        self.samples = samples
        self.size = size
        self.circular = circular

        self._head = 0
        self._count = 0
        self._missing_keys = {}

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

    def keys(self):
        return self.samples.keys()

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
            if self.samples is not None:
                for _, values in self.samples.items():
                    values.fill(0)

    def clip(self, size):
        if self.size is None:
            self.samples = {k: v[:size] for k, v in self.samples.items()}

    def _get_slice(self, count):
        # get slice of count indexes
        if self.circular:
            idxs = np.array([(self._head + i) % self.size for i in range(count)])
            self._head = (idxs[-1] + 1) % self.size
        else:
            available = self.size - self.sample_count()
            count = min(count, available)
            idxs = np.array([self._head + i for i in range(count)])
            self._head += count
        return idxs

    def _add_samples(self, samples, single=False):
        for key, values in samples.items():
            values = np.array(values)
            if single:
                values = values[np.newaxis]
            samples[key] = values

        # make sure sample counts are equal for all keys
        samples_counts = np.array([len(values) for _, values in samples.items()])
        samples_count = samples_counts[0]
        assert samples_count > 0
        assert np.all(samples_counts == samples_count)
        original_sample_count = samples_count

        # make sure sample buffers are initialized
        if self.samples is None:
            self.samples = {}
            for key, values in samples.items():
                value_shape = np.shape(values[0])
                value_dtype = values.dtype.base
                if self.size is None:
                    # unbounded buffers
                    samples_shape = (0,) + value_shape
                else:
                    # bounded buffers
                    samples_shape = (self.size,) + value_shape
                self.samples[key] = np.zeros(samples_shape, dtype=value_dtype)

        # get storage indexes for samples
        if self.size:
            idxs = self._get_slice(samples_count)
            if len(idxs) > 0:
                samples_count = len(idxs)
            else:
                available = self.size - self.sample_count()
                log_warning(f"Unable to store any of the {original_sample_count} samples in buffer"
                            f"  |  Available: {available} / {self.size})")
                return 0
        self._count += samples_count

        for key, values in samples.items():
            # make sure samples are compatible with this buffer
            if key not in self.samples:
                if key not in self._missing_keys:
                    self._missing_keys[key] = True
                    log_warning(f"Sample key not found in buffer: {key}")
                    continue

            # add sample values for key
            if self.size is None:
                self.samples[key] = np.concatenate([self.samples[key], values])
            else:
                np.put(self.samples[key], idxs, values[:samples_count])

        return samples_count

    def add_sample(self, sample):
        return self._add_samples(sample, single=True)

    def add_samples(self, samples):
        return self._add_samples(samples)

    def add_buffer(self, buffer):
        return self.add_samples(buffer.samples)

    def get_feeds(self):
        # override
        raise NotImplementedError()
