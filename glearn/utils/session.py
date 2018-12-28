import tensorflow as tf


class DebuggableSession(tf.Session):
    def __init__(self, config, config_proto=None, **kwargs):
        super().__init__(config=config_proto, **kwargs)

        self.config = config

        # debug trace
        debug_trace = self.config.get("debug_trace", False)
        if self.config.debugging and debug_trace:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.run_options = None
            self.run_metadata = None

    def run(self, fetches, feed_dict=None):
        return super().run(fetches, feed_dict=feed_dict, options=self.run_options,
                           run_metadata=self.run_metadata)
