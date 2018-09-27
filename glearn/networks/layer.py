from glearn.utils.reflection import get_class


def load_layer(network, index, definition):
    LayerClass = get_class(definition)

    return LayerClass(network, index)


class NetworkLayer(object):
    def __init__(self, network, index):
        self.network = network
        self.index = index
        self.references = {}

    @property
    def layer_type(self):
        return type(self).__name__

    @property
    def context(self):
        return self.network.context

    def build(self, inputs, outputs=None):
        pass

    def prepare_default_feeds(self, graphs, feed_map):
        return feed_map
