import numpy as np
import tensorflow as tf
from glearn.utils.config import Configurable
from glearn.utils.printing import print_tabular
from glearn.utils.collections import subtraction


GLOBAL_FEED_FAMILY = "*"


def num_variable_parameters(variables):
    # get total parameters in given variables
    return np.sum([np.product([vi.value for vi in v.get_shape()]) for v in variables])


def num_all_parameters():
    # get total network parameters
    return num_variable_parameters(tf.all_variables())


def num_global_parameters():
    # get total global parameters
    return num_variable_parameters(tf.global_variables())


def num_model_parameters():
    # get total model parameters
    return num_variable_parameters(tf.model_variables())


def num_trainable_parameters():
    # get total trainable parameters
    return num_variable_parameters(tf.trainable_variables())


def saveable_objects():
    # get all saveable objects
    return tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)


def num_saveable_objects():
    # get count of saveable objects
    return len(saveable_objects())


class NetworkContext(Configurable):
    def __init__(self, config):
        super().__init__(config)

        self.feeds = {}
        self.fetches = {}

        self.debug_runs = self.is_debugging("debug_runs")
        self.debug_runs_ignored = self.config.get("debug_runs_ignored", None)

    def add_feed(self, name, value, query=None):
        # set feed node, for query or global (None)
        if query is None:
            # global query feed
            query = [GLOBAL_FEED_FAMILY]

        # apply to specified query
        if not isinstance(query, list):
            query = [query]
        for query_name in query:
            if query_name in self.feeds:
                query_feeds = self.feeds[query_name]
            else:
                query_feeds = {}
                self.feeds[query_name] = query_feeds
            query_feeds[name] = value

    def create_feed(self, name, shape=(), dtype=tf.float32, query=None):
        # create placeholder and set as feed
        with tf.variable_scope("feeds/"):
            feed = tf.placeholder(dtype, shape, name=name)
            self.add_feed(name, feed, query)
        return feed

    def get_or_create_feed(self, name, shape=(), dtype=tf.float32, query=None):
        # get feed or create if none found
        feed = self.get_feed(name, query=query)
        if feed is None:
            return self.create_feed(name, shape, dtype, query)
        return feed

    def has_feed(self, name, query=None):
        # does this feed already exist for query
        return self.get_feed(name, query=query) is not None

    def get_feed(self, name, query=None):
        # find feed node for query name
        query_feeds = self.get_feeds(query=query)
        if name in query_feeds:
            return query_feeds[name]
        return None

    def get_feeds(self, query=None):
        # get all global feeds
        feeds = self.feeds.get(GLOBAL_FEED_FAMILY, {})
        if query is not None:
            # merge with desired query feeds
            if not isinstance(query, list):
                query = [query]
            for query_name in query:
                feeds.update(self.feeds.get(query_name, {}))
        return feeds

    def build_feed_dict(self, mapping, query=None):
        feeds = self.get_feeds(query)
        feed_dict = {}
        for key, value in mapping.items():
            if isinstance(key, str):
                if key in feeds:
                    feed = feeds[key]
                    feed_dict[feed] = value
                else:
                    query_name = GLOBAL_FEED_FAMILY if query is None else ", ".join(query)
                    self.error(f"Failed to find feed '{key}' for query: {query_name}")
            else:
                feed_dict[key] = value
        return feed_dict

    def add_fetch(self, name, value, query=None):
        # set fetch for query (defaults to name)
        if query is None:
            query = [name]
        elif not isinstance(query, list):
            query = [query]

        # apply to specified query
        for query_name in query:
            if query_name in self.fetches:
                query_fetches = self.fetches[query_name]
            else:
                query_fetches = {}
                self.fetches[query_name] = query_fetches
            query_fetches[name] = value

    def is_fetch(self, name, query=None):
        return self.get_fetch(name, query=query) is not None

    def get_fetch(self, name, query=None):
        # find feed node for query (defaults to name)
        if query is not None and not isinstance(query, list):
            query = [query]
        for query_name, query_fetches in self.fetches.items():
            if (query is None or query_name in query) and name in query_fetches:
                # return if found
                return query_fetches[name]
        return None

    def get_fetches(self, query):
        # get all fetches for specified query
        if not isinstance(query, list):
            query = [query]
        fetches = {}
        for query_name in query:
            fetches.update(self.fetches.get(query_name, {}))

        # also fetch summaries
        self.summary.prepare_fetches(fetches, query)

        return fetches

    def has_query(self, query):
        return len(self.get_fetches(query)) > 0

    def add_metric(self, name, value, query=None):
        # add metric to console log
        if query is None:
            query = "evaluate"
        self.add_fetch(name, value, query=query)

        # add metric to summary (TODO - could allow histograms too)
        if len(value.shape) > 0:
            value = tf.reduce_mean(value)
        self.summary.add_scalar(name, value, query=query)

    def run(self, query, feed_map):
        # get configured fetches
        fetches = self.get_fetches(query)

        if len(fetches) > 0:
            if self.debug_runs:
                if not self.debug_runs_ignored or subtraction(query, self.debug_runs_ignored):
                    query_s = ', '.join(query)
                    fetches_s = ', '.join(list(fetches.keys()))
                    info = {
                        "Run": {
                            "Queries": query_s,
                            "Fetches": fetches_s,
                        },
                        "Feeds": feed_map,
                    }

                    print_tabular(info, grouped=True, show_type=True, color="cyan", bold=True)

            # build final feed_dict
            feed_dict = self.build_feed_dict(feed_map, query=query)

            # run query
            results = self.sess.run(fetches, feed_dict)

            return results

        self.warning(f"No fetches found for query: {query}", once=True)
        return {}

    def optimize_loss(self, loss, networks=None, var_list=None, definition=None, name=None):
        optimize_name = ",".join([network.name for network in networks]) if networks else loss.name

        # get optimization definition
        if definition is None:
            if networks is None:
                self.warning(f"No definition found for loss optimization: {optimize_name}")
                definition = {}
            else:
                definition = networks[0].definition
        learning_rate = definition.get("learning_rate", 1e-4)
        debug_gradients = self.is_debugging("debug_gradients")  # TODO - check network configs?

        # learning rate decay
        if isinstance(learning_rate, list):
            lr_decay = learning_rate[1]
            lr_decay_intervals = learning_rate[2] if len(learning_rate) >= 3 else 1
            decay_steps = int(lr_decay_intervals * self.config.get_epoch_size())
            learning_rate = tf.train.exponential_decay(learning_rate[0], self.global_step,
                                                       decay_steps, lr_decay, staircase=True)

        # create optimizer
        optimizer_name = definition.get("optimizer", "sgd")
        if optimizer_name == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_name == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise Exception(f"Unknown optimizer type specified in config: {optimizer_name}")

        # get gradients and trainable variables
        if var_list is None:
            var_list = [v for network in networks for v in network.trainable_variables()]
        grads_tvars = optimizer.compute_gradients(loss, var_list=var_list)
        grads_tvars = [(g, v) for (g, v) in grads_tvars if g is not None]

        if len(grads_tvars) > 0:
            # check if we require unzipping grad/vars
            max_grad_norm = definition.get("max_grad_norm", None)
            require_unzip = debug_gradients or max_grad_norm is not None
            if require_unzip:
                grads, tvars = zip(*grads_tvars)

            # apply gradient clipping
            if max_grad_norm is not None:
                with tf.variable_scope("clipped_gradients"):
                    grads, global_norm = tf.clip_by_global_norm(grads, max_grad_norm,
                                                                name="clip_by_global_norm")

                    # metrics to observe clipped gradient ratio and global norm
                    div = global_norm + 1.0e-8
                    clipped_ratio = tf.maximum(global_norm - max_grad_norm, 0) / div
                    self.summary.add_scalar("global_norm", global_norm, query=name)
                    self.summary.add_scalar("clipped_ratio", clipped_ratio, query=name)

            if require_unzip:
                grads_tvars = zip(grads, tvars)

            # apply gradients
            optimize = optimizer.apply_gradients(grads_tvars)

            # add learning rate and gradient summaries
            self.summary.add_scalar("learning_rate", learning_rate, query=name)
            if debug_gradients:
                self.summary.add_variables(tvars, query=name)
                self.summary.add_gradients(zip(grads, tvars), query=name)

            # fetch
            if name is not None:
                self.add_fetch(name, optimize)
        else:
            self.warning(f"No gradients found for optimization: {optimize_name}")
            optimize = tf.no_op()

        return optimize


class NetworkContextProxy(Configurable):
    def __init__(self, config, context):
        super().__init__(config)

        self.context = context

    def add_feed(self, name, value, query=None):
        return self.context.add_feed(name, value, query=query)

    def create_feed(self, name, shape=(), dtype=tf.float32, query=None):
        return self.context.create_feed(name, shape=shape, dtype=dtype, query=query)

    def get_or_create_feed(self, name, shape=(), dtype=tf.float32, query=None):
        return self.context.get_or_create_feed(name, shape=shape, dtype=dtype, query=query)

    def get_feed(self, name, query=None):
        return self.context.get_feed(name, query=query)

    def get_feeds(self, query=None):
        return self.context.get_feeds(query=query)

    def build_feed_dict(self, mapping, query=None):
        return self.context.build_feed_dict(mapping=mapping, query=query)

    def add_fetch(self, name, value, query=None):
        return self.context.add_fetch(name, value, query=query)

    def is_fetch(self, name, query=None):
        return self.context.is_fetch(name, query=query)

    def get_fetch(self, name, query=None):
        return self.context.get_fetch(name, query=query)

    def get_fetches(self, query):
        return self.context.get_fetches(query)

    def add_metric(self, name, value, query=None):
        self.context.add_metric(name, value, query=query)

    def run(self, query, feed_map):
        return self.context.run(query, feed_map=feed_map)
