from glearn.datasets.dataset import Dataset


class LabeledDataset(Dataset):
    def __init__(self, *args, label_names=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_names = label_names

    @property
    def label_count(self):
        return len(self.label_names)

    def encipher_element(self, value):
        label = self.label_names.index(value)
        return super().encipher_element(label)

    def decipher_element(self, value):
        label = super().decipher_element(value)
        return self.label_names[label]
