from interface import Interface


class Transition(object):
    def __init__(self, observation, action, reward, new_observation, done, info):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.new_observation = new_observation
        self.done = done
        self.info = info


class Batch(object):
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.info = {}


def transition_batch(transitions):
    batch = Batch()
    batch.inputs = [e.observation for e in transitions]
    batch.outputs = [e.action for e in transitions]
    batch.info["transitions"] = transitions
    return batch


class Dataset(object):
    def __init__(self, inputs, outputs, input_space, output_space, deterministic=True):
        self.inputs = inputs
        self.outputs = outputs
        self.input = Interface(input_space)
        self.output = Interface(output_space)
        self.deterministic = deterministic

        self.reset()

    def reset(self):
        self.head = 0

    def batch(self, batch_size):
        batch = Batch()
        batch.inputs = self.inputs[self.head:self.head + batch_size]
        batch.outputs = self.outputs[self.head:self.head + batch_size]

        batch.inputs = [self.input.encode(o) for o in batch.inputs]
        batch.outputs = [self.output.encode(o) for o in batch.outputs]

        self.head = (self.head + batch_size) % len(self.inputs)
        return batch
