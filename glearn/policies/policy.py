import tensorflow as tf
from glearn.utils.printing import colorize
from glearn.utils.summary import SummaryWriter, NullSummaryWriter


class Policy(object):
    def __init__(self, config):
        self.config = config

        self.multithreaded = config.get("multithreaded", False)

        self.feeds = {}
        self.fetches = {}
        self.layers = {}
        self.results = {}
        self.summaries = {}
        self.training = False

        if self.rendering:
            self.init_viewer()
        self.init_summaries()
        self.init_model()

    def __str__(self):
        properties = [
            "multi-threaded" if self.multithreaded else "single-threaded",
        ]
        return f"{type(self).__name__}({', '.join(properties)})"

    def log(self, *args):
        print(*args)

    def error(self, message):
        self.log(colorize(message, "red"))

    @property
    def debugging(self):
        return self.config.debugging

    @property
    def project(self):
        return self.config.project

    @property
    def dataset(self):
        return self.config.dataset

    @property
    def env(self):
        return self.config.env

    @property
    def supervised(self):
        return self.config.supervised

    @property
    def reinforcement(self):
        return self.config.reinforcement

    @property
    def input(self):
        return self.config.input

    @property
    def output(self):
        return self.config.output

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
            graphs = ["*"]
        if not isinstance(graphs, list):
            graphs = [graphs]
        # apply to specified graphs
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

    def get_feeds(self, graph=None):
        # get all global feeds
        feeds = self.feeds.get("*", {})
        if graph is not None:
            # merge with desired graph feeds
            feeds.update(self.feeds.get(graph, {}))
        return feeds

    def build_feed_dict(self, mapping, graph=None):
        feeds = self.get_feeds(graph)
        feed_dict = {}
        for key, value in mapping.items():
            if key in feeds:
                feed = feeds[key]
                feed_dict[feed] = value
            else:
                graph_name = "GLOBAL" if graph is None else graph
                self.error(f"Failed to find feed '{key}' for graph '{graph_name}'")
        return feed_dict

    def set_fetch(self, name, value, graphs=None):
        # set fetch, for graph or global (None)
        if graphs is None:
            # global graph fetch
            graphs = ["*"]
        if not isinstance(graphs, list):
            graphs = [graphs]
        # apply to specified graphs
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

    def get_fetches(self, graph):
        # get all global fetches
        fetches = self.fetches.get("*", {})
        if graph != "*":
            # merge with desired graph fetches
            fetches.update(self.fetches.get(graph, {}))
        return fetches

    def add_layer(self, type_name, layer):
        if type_name in self.layers:
            type_layers = self.layers[type_name]
        else:
            type_layers = []
            self.layers[type_name] = type_layers
        type_layers.append(layer)

    def get_layer(self, type_name, index=0):
        if type_name in self.layers:
            type_layers = self.layers[type_name]
            if index < len(type_layers):
                return type_layers[index]
        return None

    def get_layer_count(self, type_name=None):
        if type_name is None:
            return len(self.layers)
        else:
            if type_name in self.layers:
                return len(self.layers[type_name])
        return 0

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

    def prepare_default_feeds(self, graph, feed_map):
        return feed_map

    def run(self, sess, graph, feed_map, global_step=None, summary_family=None):
        # get configured fetches
        fetches = self.get_fetches(graph)

        # also fetch summaries
        if summary_family is None:
            summary_family = graph
        self.summary.prepare_fetches(fetches, summary_family)

        if len(fetches) > 0:
            # build final feed_dict
            feed_dict = self.build_feed_dict(feed_map, graph=graph)

            # run graph
            results = sess.run(fetches, feed_dict)

            # handle summaries
            self.summary.process_results(results, summary_family, global_step=global_step)

            # store results
            self.results[graph] = results

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
