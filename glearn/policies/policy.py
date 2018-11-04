import tensorflow as tf
from glearn.utils.summary import SummaryWriter, NullSummaryWriter
from glearn.networks.context import NetworkContext


class Policy(NetworkContext):
    def __init__(self, config):
        super().__init__(config)

        self.multithreaded = config.get("multithreaded", False)  # TODO get this from dataset

        if self.rendering:
            self.init_viewer()
        self.init_summaries()
        self.build_model()

    def __str__(self):
        properties = [
            "multithreaded" if self.multithreaded else "single-threaded",
        ]
        return f"{type(self).__name__}({', '.join(properties)})"

    @property
    def tensorboard_path(self):
        return self.config.tensorboard_path

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

    def init_summaries(self):
        if self.tensorboard_path is not None:
            self.log(f"Tensorboard log root directory: {self.tensorboard_path}")
            self.summary = SummaryWriter(self.tensorboard_path)
        else:
            self.summary = NullSummaryWriter()

    def start_summaries(self, sess):
        if self.summary is not None:
            self.summary.start(graph=sess.graph, server=True)

    def stop_summaries(self, sess):
        if self.summary is not None:
            self.summary.stop()

    def create_default_feeds(self):
        if self.supervised:
            inputs = self.dataset.get_inputs()
            outputs = self.dataset.get_outputs()
        else:
            inputs = tf.placeholder(self.input.dtype, (None,) + self.input.shape, name="X")
            outputs = tf.placeholder(self.output.dtype, (None,) + self.output.shape, name="Y")

        # add reference to interfaces
        inputs.interface = self.input
        outputs.interface = self.output

        # set feeds
        self.set_feed("X", inputs)
        self.set_feed("Y", outputs)

        # set fetches
        self.set_fetch("X", inputs, ["evaluate", "debug"])
        self.set_fetch("Y", outputs, "evaluate")
        return inputs, outputs

    def build_model(self):
        # create input placeholders
        with tf.name_scope('feeds'):
            inputs, outputs = self.create_default_feeds()

        # build main prediction model
        y = self.build_predict(inputs, outputs)

        # store prediction
        self.set_fetch("predict", y, ["predict", "evaluate"])

    def build_predict(self, inputs, outputs):
        pass

    def reset(self):
        pass

    def prepare_default_feeds(self, graphs, feed_map):
        return feed_map

    def get_fetches(self, graphs=None):
        fetches = super().get_fetches(graphs)

        # also fetch summaries
        self.summary.prepare_fetches(fetches, graphs)

        return fetches

    def run(self, sess, graphs, feed_map):
        results = super().run(sess, graphs, feed_map)

        # process summaries
        if len(results) > 0:
            self.summary.process_results(results)

        return results

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
