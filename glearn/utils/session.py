import tensorflow as tf


class DebuggableSession(tf.InteractiveSession):
    def __init__(self, config, config_proto=None, **kwargs):
        if config_proto is None:
            config_kwargs = config.get("session_config", None)
            if config_kwargs is not None:
                config_proto = tf.ConfigProto(**config_kwargs)

        super().__init__(config=config_proto, **kwargs)

        self.config = config
        self.debug_trace = self.config.is_debugging("debug_trace")

        # debug trace
        if self.debug_trace:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.run_options = None
            self.run_metadata = None

    def run(self, fetches, feed_dict=None):
        results = super().run(fetches, feed_dict=feed_dict, options=self.run_options,
                              run_metadata=self.run_metadata)

        if self.run_metadata is not None:
            def add_fetch_metadata(fetch, name):
                self.config.summary.add_run_metadata(self.run_metadata, name)
            self._map_fetches(fetches, add_fetch_metadata)

        return results

    def _map_fetches(self, fetches, func, name=None):
        if isinstance(fetches, dict):
            for fetch_name, fetch in fetches.items():
                self._map_fetches(fetch, func, name=fetch_name)
        elif isinstance(fetches, list) or isinstance(fetches, tuple):
            for fetch in fetches:
                self._map_fetches(fetch, func)
        else:
            if name is None and hasattr(fetches, "name"):
                name = fetches.name
            func(fetches, name)
