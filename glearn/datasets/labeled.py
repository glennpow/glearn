from glearn.datasets.dataset import Dataset


class LabeledDataset(Dataset):
    def __init__(self, *args, label_names=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_names = label_names

    def encipher(self, value):
        label = self.label_names.index(value)
        return super().encipher(label)

    def decipher(self, value):
        label = super().decipher(value)
        return self.label_names[label]
