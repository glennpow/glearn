class Dataset(object):
    def __init__(self, inputs, outputs, input_space, output_space):
        self.data = list(zip(inputs, outputs))
        self.input_space = input_space
        self.output_space = output_space

        self.reset()

    def reset(self):
        self.head = 0

    def batch(self, batch_size):
        data = self.data[self.head:batch_size]
        self.head += batch_size
        return data
