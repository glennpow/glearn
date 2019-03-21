class Batch(object):
    def __init__(self, mode="train"):
        self.mode = mode
        self.inputs = []
        self.outputs = []
