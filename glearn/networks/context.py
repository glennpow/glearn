import numpy as np
import tensorflow as tf
from glearn.utils.config import Configurable


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

    def set_feed(self, name, value, families=None):
        # set feed node, for family or global (None)
        if families is None:
            # global family feed
            families = [GLOBAL_FEED_FAMILY]

        # apply to specified families
        if not isinstance(families, list):
            families = [families]
        for family in families:
            if family in self.feeds:
                family_feeds = self.feeds[family]
            else:
                family_feeds = {}
                self.feeds[family] = family_feeds
            family_feeds[name] = value

    def create_feed(self, name, families=None, shape=(), dtype=tf.float32):
        # create placeholder and set as feed
        ph = tf.placeholder(dtype, shape, name=name)
        self.set_feed(name, ph, families)
        return ph

    def get_or_create_feed(self, name, families=None, shape=(), dtype=tf.float32):
        # get feed or create if none found
        ph = self.get_feed(name)
        if ph is None:
            return self.create_feed(name, families, shape, dtype)
        return ph

    def get_feed(self, name, family=None):
        # find feed node for family name
        family_feeds = self.get_feeds(family)
        if name in family_feeds:
            return family_feeds[name]
        return None

    def get_feeds(self, families=None):
        # get all global feeds
        feeds = self.feeds.get(GLOBAL_FEED_FAMILY, {})
        if families is not None:
            # merge with desired family feeds
            if not isinstance(families, list):
                families = [families]
            for family in families:
                feeds.update(self.feeds.get(family, {}))
        return feeds

    def build_feed_dict(self, mapping, families=None):
        feeds = self.get_feeds(families)
        feed_dict = {}
        for key, value in mapping.items():
            if isinstance(key, str):
                if key in feeds:
                    feed = feeds[key]
                    feed_dict[feed] = value
                else:
                    family_name = GLOBAL_FEED_FAMILY if families is None else ", ".join(families)
                    self.error(f"Failed to find feed '{key}' for family: {family_name}")
            else:
                feed_dict[key] = value
        return feed_dict

    def set_fetch(self, name, value, families=None):
        # set fetch for families (defaults to name)
        if families is None:
            families = [name]
        elif not isinstance(families, list):
            families = [families]

        # apply to specified families
        for family in families:
            if family in self.fetches:
                family_fetches = self.fetches[family]
            else:
                family_fetches = {}
                self.fetches[family] = family_fetches
            family_fetches[name] = value

        # TODO - not sure why this was returning anything here...
        # self.get_fetch(name, families=families)

    def is_fetch(self, name, families=None):
        return self.get_fetch(name, families=families) is not None

    def get_fetch(self, name, families=None):
        # find feed node for family (defaults to name)
        if families is not None and not isinstance(families, list):
            families = [families]
        for family, family_fetches in self.fetches.items():
            if (families is None or family in families) and name in family_fetches:
                # return if found
                return family_fetches[name]
        return None

    def get_fetches(self, families):
        # get all fetches for specified families
        if not isinstance(families, list):
            families = [families]
        fetches = {}
        for family in families:
            fetches.update(self.fetches.get(family, {}))
        return fetches

    def run(self, families, feed_map):
        # get configured fetches
        fetches = self.get_fetches(families)

        if len(fetches) > 0:
            if self.debug_runs:
                families_s = ', '.join(families)
                feeds_s = ', '.join(list(feed_map.keys()))
                fetches_s = ', '.join(list(fetches.keys()))
                message = (f"Run | Families: '{families_s}' | Feeds: '{feeds_s}'"
                           f" | Fetches: '{fetches_s}'")
                self.log(message, "cyan", bold=True)

            # build final feed_dict
            feed_dict = self.build_feed_dict(feed_map, families=families)

            # run family
            results = self.sess.run(fetches, feed_dict)

            # store results
            self.latest_results.update(results)

            return results
        return {}
