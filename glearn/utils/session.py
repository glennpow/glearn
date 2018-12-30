import tensorflow as tf


class DebuggableSession(tf.Session):
    def __init__(self, config, config_proto=None, **kwargs):
        if config_proto is None:
            config_kwargs = config.get("session_config", None)
            if config_kwargs is not None:
                config_proto = tf.ConfigProto(**config_kwargs)

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
        results = super().run(fetches, feed_dict=feed_dict, options=self.run_options,
                              run_metadata=self.run_metadata)

        if self.run_metadata is not None:
            self.config.summary.add_run_metadata(self.run_metadata)

        return results
