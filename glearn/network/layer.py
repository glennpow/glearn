class NetworkLayer(object):
    def __init__(self, index):
        self.index = index
        self.references = {}

    @property
    def layer_type(self):
        return type(self).__name__

    def build(self, policy, inputs, outputs=None):
        pass

    def prepare_default_feeds(self, graphs, feed_map):
        return feed_map
