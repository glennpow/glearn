import tensorflow as tf
from glearn.utils.summary import SummaryWriter, NullSummaryWriter
from glearn.utils.config import Configurable


GLOBAL_GRAPH = "*"


class Policy(Configurable):
    def __init__(self, config):
        super().__init__(config)

        self.multithreaded = config.get("multithreaded", False)  # TODO get this from dataset

        self.feeds = {}
        self.fetches = {}
        self.summaries = {}
        self.latest_results = {}

        if self.rendering:
            self.init_viewer()
        self.init_summaries()
        self.init_model()

    def __str__(self):
        properties = [
            "multi-threaded" if self.multithreaded else "single-threaded",
        ]
        return f"{type(self).__name__}({', '.join(properties)})"

    @property
    def tensorboard_path(self):
        return self.config.tensorboard_path

    def init_model(self):
        pass

    def start_session(self, sess):
        self.start_threading(sess)

        self.start_summaries(sess)

    def stop_session(self, sess):
        self.stop_threading(sess)

        self.stop_summaries(sess)

    def start_threading(self, sess):
        if self.multithreaded:
            # start thread queue
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=sess)

            num_threads = len(self.threads)
            print(f"Started training threads: {num_threads}")

    def stop_threading(self, sess):
        if self.multithreaded:
            # join all threads
            self.coord.request_stop()
            self.coord.join(self.threads)

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
        graph_fetches = self.get_fetches(graph)
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

    def init_summaries(self):
        if self.tensorboard_path is not None:
            self.log(f"Tensorboard log root directory: {self.tensorboard_path}")
            self.summary = SummaryWriter(self.tensorboard_path)
        else:
            self.summary = NullSummaryWriter()

    def start_summaries(self, sess):
        if self.summary is not None:
            self.summary.start(graph=sess.graph, server=self.debugging)

    def stop_summaries(self, sess):
        if self.summary is not None:
            self.summary.stop()

    def reset(self):
        pass

    def create_default_feeds(self):
        if self.supervised:
            inputs = self.dataset.get_inputs()
            outputs = self.dataset.get_outputs()
        else:
            inputs = tf.placeholder(self.input.dtype, (None,) + self.input.shape, name="X")
            outputs = tf.placeholder(self.output.dtype, (None,) + self.output.shape, name="Y")
        self.set_feed("X", inputs)
        self.set_feed("Y", outputs)
        return inputs, outputs

    def prepare_default_feeds(self, graphs, feed_map):
        return feed_map

    def run(self, sess, graphs, feed_map, global_step=None):
        # get configured fetches
        fetches = self.get_fetches(graphs)

        # also fetch summaries
        self.summary.prepare_fetches(fetches, graphs)

        if len(fetches) > 0:
            # build final feed_dict
            feed_dict = self.build_feed_dict(feed_map, graphs=graphs)

            # run graph
            results = sess.run(fetches, feed_dict)

            # handle summaries
            self.summary.process_results(results, global_step=global_step)

            # store results
            self.latest_results.update(results)

            return results
        return {}

    @property
    def viewer(self):
        return self.config.viewer

    @property
    def rendering(self):
        return self.viewer.rendering

    def init_viewer(self):
        # register for events from viewer
        self.viewer.add_listener(self)

    def on_key_press(self, key, modifiers):
        pass
