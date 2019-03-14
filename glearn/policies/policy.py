import tensorflow as tf
from glearn.networks.context import NetworkContextProxy


class Policy(NetworkContextProxy):
    def __init__(self, config, context):
        super().__init__(config, context=context)

        self.multithreaded = config.get("multithreaded", False)  # TODO get this from dataset
        self.threads = None

        if self.rendering:
            self.init_viewer()

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
        if self.multithreaded and self.threads:
            # join all threads
            self.coord.request_stop()
            self.coord.join(self.threads)
            self.threads = None

    def build_predict(self):
        # override
        pass

    def build_loss(self):
        # override
        return None

    def optimize_loss(self, loss):
        # override
        return None

    def reset(self):
        # override
        pass

    def prepare_default_feeds(self, queries, feed_map):
        # override
        return feed_map

    def get_info(self):
        return {
            "Description": str(self),
        }

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
