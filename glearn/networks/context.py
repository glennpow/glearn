import numpy as np
import tensorflow as tf
from glearn.utils.config import Configurable
from glearn.utils.printing import print_tabular


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

        self.debug_runs = self.config.is_debugging("debug_runs")

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
        ph = self.get_feed(name)
        if ph is None:
            return self.create_feed(name, queries, shape, dtype)
        return ph

    def get_feed(self, name, query=None):
        # find feed node for query name
        query_feeds = self.get_feeds(query)
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

    def set_fetch(self, name, value, queries=None):
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

        # TODO - not sure why this was returning anything here...
        # self.get_fetch(name, queries=queries)

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
        return fetches

    def run(self, queries, feed_map):
        # get configured fetches
        fetches = self.get_fetches(queries)

        if len(fetches) > 0:
            if self.debug_runs:
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
        return {}
