import numpy as np
import tensorflow as tf
from glearn.networks.context import NetworkContext


class Policy(NetworkContext):
    def __init__(self, config):
        super().__init__(config)

        self.multithreaded = config.get("multithreaded", False)  # TODO get this from dataset

        if self.rendering:
            self.init_viewer()
        self.build_model()

    def __str__(self):
        properties = [
            "multithreaded" if self.multithreaded else "single-threaded",
        ]
        return f"{type(self).__name__}({', '.join(properties)})"

    def start_session(self):
        self.start_threading()

    def stop_session(self):
        self.stop_threading()

    def start_threading(self):
        if self.multithreaded:
            # start thread queue
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

            num_threads = len(self.threads)
            print(f"Started training threads: {num_threads}")

    def stop_threading(self):
        if self.multithreaded:
            # join all threads
            self.coord.request_stop()
            self.coord.join(self.threads)

    def build_model(self):
        # create input/output nodes
        self.build_inputs()

        # build prediction model
        self.build_predict()

        # build loss
        self.build_loss()

    def build_inputs(self):
        with tf.name_scope('feeds'):
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
            self.set_fetch("X", inputs, "debug")
            self.set_fetch("Y", outputs, "debug")

        self.inputs = inputs
        self.outputs = outputs

        # default output for debugging
        if self.debugging:
            batch_size = self.config.get("batch_size", 1)
            self.default_output = np.zeros((batch_size,) + self.output.shape, self.output.dtype)

    def build_predict(self):
        # override
        pass

    def build_loss(self):
        # override
        pass

    def reset(self):
        # override
        pass

    def prepare_default_feeds(self, graphs, feed_map):
        # make sure we have outputs defined
        # if self.debugging and "Y" not in feed_map:
        #     feed_map["Y"] = self.default_output

        return feed_map

    def get_fetches(self, graphs=None):
        fetches = super().get_fetches(graphs)

        # also fetch summaries
        self.summary.prepare_fetches(fetches, graphs)

        return fetches

    def get_info(self):
        return {
            "Description": str(self),
        }

    def run(self, graphs, feed_map):
        results = super().run(graphs, feed_map)

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
