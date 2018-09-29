import tensorflow as tf
from glearn.utils.config import Configurable


GLOBAL_GRAPH = "*"


class NetworkContext(Configurable):
    def __init__(self, config):
        super().__init__(config)

        self.feeds = {}
        self.fetches = {}
        self.latest_results = {}

    def set_feed(self, name, value, graphs=None):
        # set feed node, for graph or global (None)
        if graphs is None:
            # global graph feed
            graphs = [GLOBAL_GRAPH]
        # apply to specified graphs
        if not isinstance(graphs, list):
            graphs = [graphs]
        for graph in graphs:
            if graph in self.feeds:
                graph_feeds = self.feeds[graph]
            else:
                graph_feeds = {}
                self.feeds[graph] = graph_feeds
            graph_feeds[name] = value

    def create_feed(self, name, graphs=None, shape=(), dtype=tf.float32):
        # create placeholder and set as feed
        ph = tf.placeholder(dtype, shape, name=name)
        self.set_feed(name, ph, graphs)
        return ph

    def get_or_create_feed(self, name, graphs=None, shape=(), dtype=tf.float32):
        # get feed or create if none found
        ph = self.get_feed(name)
        if ph is None:
            return self.create_feed(name, graphs, shape, dtype)
        return ph

    def get_feed(self, name, graph=None):
        # find feed node for graph name
        graph_feeds = self.get_feeds(graph)
        if name in graph_feeds:
            return graph_feeds[name]
        return None

    def get_feeds(self, graphs=None):
        # get all global feeds
        feeds = self.feeds.get(GLOBAL_GRAPH, {})
        if graphs is not None:
            # merge with desired graph feeds
            if not isinstance(graphs, list):
                graphs = [graphs]
            for graph in graphs:
                feeds.update(self.feeds.get(graph, {}))
        return feeds

    def build_feed_dict(self, mapping, graphs=None):
        feeds = self.get_feeds(graphs)
        feed_dict = {}
        for key, value in mapping.items():
            if key in feeds:
                feed = feeds[key]
                feed_dict[feed] = value
            else:
                graph_name = GLOBAL_GRAPH if graphs is None else ", ".join(graphs)
                self.error(f"Failed to find feed '{key}' for graph '{graph_name}'")
        return feed_dict

    def set_fetch(self, name, value, graphs=None):
        # set fetch, for graph or global (None)
        if graphs is None:
            # global graph fetch
            graphs = [GLOBAL_GRAPH]
        # apply to specified graphs
        if not isinstance(graphs, list):
            graphs = [graphs]
        for graph in graphs:
            if graph in self.fetches:
                graph_fetches = self.fetches[graph]
            else:
                graph_fetches = {}
                self.fetches[graph] = graph_fetches
            graph_fetches[name] = value

    def get_fetch(self, name, graph=None):
        # find feed node for graph name
        for g, graph_fetches in self.fetches.items():
            if g == GLOBAL_GRAPH or g == graph:
                if name in graph_fetches:
                    return graph_fetches[name]
        return None

    def get_fetches(self, graphs=None):
        # get all global fetches
        fetches = self.fetches.get(GLOBAL_GRAPH, {})
        if graphs is not None:
            # merge with desired graph fetches
            if not isinstance(graphs, list):
                graphs = [graphs]
            for graph in graphs:
                fetches.update(self.fetches.get(graph, {}))
        return fetches

    def run(self, sess, graphs, feed_map):
        # get configured fetches
        fetches = self.get_fetches(graphs)

        if len(fetches) > 0:
            # build final feed_dict
            feed_dict = self.build_feed_dict(feed_map, graphs=graphs)

            # run graph
            results = sess.run(fetches, feed_dict)

            # store results
            self.latest_results.update(results)

            return results
        return {}
