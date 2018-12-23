import tensorflow as tf
from glearn.utils.config import Configurable
from glearn.utils.printing import colorize


GLOBAL_FEED_GRAPH = "*"


class NetworkContext(Configurable):
    def __init__(self, config):
        super().__init__(config)

        self.feeds = {}
        self.fetches = {}
        self.latest_results = {}

        self.debug_runs = self.config.get("debug_runs", False)

    def set_feed(self, name, value, graphs=None):
        # set feed node, for graph or global (None)
        if graphs is None:
            # global graph feed
            graphs = [GLOBAL_FEED_GRAPH]

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
        feeds = self.feeds.get(GLOBAL_FEED_GRAPH, {})
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
                graph_name = GLOBAL_FEED_GRAPH if graphs is None else ", ".join(graphs)
                self.error(f"Failed to find feed '{key}' for graph: {graph_name}")
        return feed_dict

    def set_fetch(self, name, value, graphs=None):
        # set fetch for graphs (defaults to name)
        if graphs is None:
            graphs = [name]
        elif not isinstance(graphs, list):
            graphs = [graphs]

        # apply to specified graphs
        for graph in graphs:
            if graph in self.fetches:
                graph_fetches = self.fetches[graph]
            else:
                graph_fetches = {}
                self.fetches[graph] = graph_fetches
            graph_fetches[name] = value
        for graph, graph_fetches in self.fetches.items():
            if (graphs is None or graph in graphs) and name in graph_fetches:
                return graph_fetches[name]

    def is_fetch(self, name, graphs=None):
        return self.get_fetch(name, graphs=graphs) is not None

    def get_fetch(self, name, graphs=None):
        # find feed node for graph (defaults to name)
        if graphs is not None and not isinstance(graphs, list):
            graphs = [graphs]
        for graph, graph_fetches in self.fetches.items():
            if (graphs is None or graph in graphs) and name in graph_fetches:
                return graph_fetches[name]
        return None

    def get_fetches(self, graphs):
        # get all fetches for specified graphs
        if not isinstance(graphs, list):
            graphs = [graphs]
        fetches = {}
        for graph in graphs:
            fetches.update(self.fetches.get(graph, {}))
        return fetches

    def run(self, sess, graphs, feed_map):
        # get configured fetches
        fetches = self.get_fetches(graphs)

        if len(fetches) > 0:
            if self.debug_runs:
                graphs_s = ', '.join(graphs)
                feeds_s = ', '.join(list(feed_map.keys()))
                fetches_s = ', '.join(list(fetches.keys()))
                message = f"══ Run: '{graphs_s}' (Feeds: '{feeds_s}', Fetches: '{fetches_s}')"
                self.debug(colorize(message, "cyan", bold=True))

            # build final feed_dict
            feed_dict = self.build_feed_dict(feed_map, graphs=graphs)

            # run graph
            results = sess.run(fetches, feed_dict)

            # store results
            self.latest_results.update(results)

            return results
        return {}
