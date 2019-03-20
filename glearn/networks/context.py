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
        self.latest_results = {}

        self.debug_runs = self.is_debugging("debug_runs")
        self.debug_runs_ignored = self.config.get("debug_runs_ignored", None)

    def set_feed(self, name, value, queries=None):
        # set feed node, for query or global (None)
        if queries is None:
            # global query feed
            queries = [GLOBAL_FEED_FAMILY]

        # apply to specified queries
        if not isinstance(queries, list):
            queries = [queries]
        for query in queries:
            if query in self.feeds:
                query_feeds = self.feeds[query]
            else:
                query_feeds = {}
                self.feeds[query] = query_feeds
            query_feeds[name] = value

    def create_feed(self, name, queries=None, shape=(), dtype=tf.float32):
        # create placeholder and set as feed
        ph = tf.placeholder(dtype, shape, name=name)
        self.set_feed(name, ph, queries)
        return ph

    def get_or_create_feed(self, name, queries=None, shape=(), dtype=tf.float32):
        # get feed or create if none found
        ph = self.get_feed(name, queries=queries)
        if ph is None:
            return self.create_feed(name, queries, shape, dtype)
        return ph

    def has_feed(self, name, queries=None):
        # does this feed already exist for queries
        return self.get_feed(name, queries=queries) is not None

    def get_feed(self, name, queries=None):
        # find feed node for query name
        query_feeds = self.get_feeds(queries=queries)
        if name in query_feeds:
            return query_feeds[name]
        return None

    def get_feeds(self, queries=None):
        # get all global feeds
        feeds = self.feeds.get(GLOBAL_FEED_FAMILY, {})
        if queries is not None:
            # merge with desired query feeds
            if not isinstance(queries, list):
                queries = [queries]
            for query in queries:
                feeds.update(self.feeds.get(query, {}))
        return feeds

    def build_feed_dict(self, mapping, queries=None):
        feeds = self.get_feeds(queries)
        feed_dict = {}
        for key, value in mapping.items():
            if isinstance(key, str):
                if key in feeds:
                    feed = feeds[key]
                    feed_dict[feed] = value
                else:
                    query_name = GLOBAL_FEED_FAMILY if queries is None else ", ".join(queries)
                    self.error(f"Failed to find feed '{key}' for query: {query_name}")
            else:
                feed_dict[key] = value
        return feed_dict

    def add_fetch(self, name, value, queries=None):
        # set fetch for queries (defaults to name)
        if queries is None:
            queries = [name]
        elif not isinstance(queries, list):
            queries = [queries]

        # apply to specified queries
        for query in queries:
            if query in self.fetches:
                query_fetches = self.fetches[query]
            else:
                query_fetches = {}
                self.fetches[query] = query_fetches
            query_fetches[name] = value

    def is_fetch(self, name, queries=None):
        return self.get_fetch(name, queries=queries) is not None

    def get_fetch(self, name, queries=None):
        # find feed node for query (defaults to name)
        if queries is not None and not isinstance(queries, list):
            queries = [queries]
        for query, query_fetches in self.fetches.items():
            if (queries is None or query in queries) and name in query_fetches:
                # return if found
                return query_fetches[name]
        return None

    def get_fetches(self, queries):
        # get all fetches for specified queries
        if not isinstance(queries, list):
            queries = [queries]
        fetches = {}
        for query in queries:
            fetches.update(self.fetches.get(query, {}))

        # also fetch summaries
        self.summary.prepare_fetches(fetches, queries)

        return fetches

    def add_metric(self, name, value, query=None):
        # add a metric to log to console and summary
        if query is None:
            query = "evaluate"
        self.add_fetch(name, value, queries=query)
        self.summary.add_scalar(name, value, query=query)

    def run(self, queries, feed_map):
        # get configured fetches
        fetches = self.get_fetches(queries)

        if len(fetches) > 0:
            if self.debug_runs:
                if not self.debug_runs_ignored or subtraction(queries, self.debug_runs_ignored):
                    queries_s = ', '.join(queries)
                    fetches_s = ', '.join(list(fetches.keys()))
                    info = {
                        "Run": {
                            "Queries": queries_s,
                            "Fetches": fetches_s,
                        },
                        "Feeds": feed_map,
                    }

                    print_tabular(info, grouped=True, show_type=True, color="cyan", bold=True)

            # build final feed_dict
            feed_dict = self.build_feed_dict(feed_map, queries=queries)

            # run query
            results = self.sess.run(fetches, feed_dict)

            # store results
            self.latest_results.update(results)

            return results

        self.warning(f"No fetches found for queries: {queries}", once=True)
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
        lr_decay = definition.get("lr_decay", None)
        if lr_decay is not None:
            lr_decay_intervals = definition.get("lr_decay_intervals", 1)
            decay_steps = int(lr_decay_intervals * self.config.get_interval_size())
            learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
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

        # check if we require unzipping grad/vars
        max_grad_norm = definition.get("max_grad_norm", None)
        require_unzip = debug_gradients or max_grad_norm is not None
        if require_unzip:
            grads, tvars = zip(*grads_tvars)

        # apply gradient clipping
        if max_grad_norm is not None:
            with tf.name_scope("clipped_gradients"):
                grads, global_norm = tf.clip_by_global_norm(grads, max_grad_norm)

                # metrics to observe clipped gradient ratio and global norm
                if debug_gradients:
                    clipped_ratio = tf.maximum(global_norm - max_grad_norm, 0) / global_norm
                    self.summary.add_scalar("global_norm", global_norm)
                    self.summary.add_scalar("clipped_ratio", clipped_ratio)

        if require_unzip:
            grads_tvars = zip(grads, tvars)

        # apply gradients
        optimize = optimizer.apply_gradients(grads_tvars)

        # add learning rate and gradient summaries
        self.summary.add_scalar("learning_rate", learning_rate)
        if debug_gradients:
            self.summary.add_gradients(zip(grads, tvars))

        # fetch
        if name is not None:
            self.add_fetch(name, optimize)

        return optimize


class NetworkContextProxy(Configurable):
    def __init__(self, config, context):
        super().__init__(config)

        self.context = context

    def set_feed(self, name, value, queries=None):
        return self.context.set_feed(name, value, queries=queries)

    def create_feed(self, name, queries=None, shape=(), dtype=tf.float32):
        return self.context.create_feed(name, queries=queries, shape=shape, dtype=dtype)

    def get_or_create_feed(self, name, queries=None, shape=(), dtype=tf.float32):
        return self.context.get_or_create_feed(name, queries=queries, shape=shape, dtype=dtype)

    def get_feed(self, name, query=None):
        return self.context.get_feed(name, query=query)

    def get_feeds(self, queries=None):
        return self.context.get_feeds(queries=queries)

    def build_feed_dict(self, mapping, queries=None):
        return self.context.build_feed_dict(mapping=mapping, queries=queries)

    def add_fetch(self, name, value, queries=None):
        return self.context.add_fetch(name, value, queries=queries)

    def is_fetch(self, name, queries=None):
        return self.context.is_fetch(name, queries=queries)

    def get_fetch(self, name, queries=None):
        return self.context.get_fetch(name, queries=queries)

    def get_fetches(self, queries):
        return self.context.get_fetches(queries)

    def add_metric(self, name, value, query=None):
        self.context.add_metric(name, value, query=query)

    def run(self, queries, feed_map):
        return self.context.run(queries, feed_map=feed_map)
