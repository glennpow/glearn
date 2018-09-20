import os
import shutil
import tensorflow as tf


SUMMARY_FETCH_ID = "__summary__"
DEFAULT_SUBDIRECTORY = "events"


class SummaryWriter(object):
    def __init__(self, path):
        self.path = path

        self.simple_values = {}
        self.summaries = {}
        self.summary_fetches = {}
        self.writers = {}

    def start(self, append=False, **kwargs):
        self.kwargs = kwargs

        if not append:
            shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)

    def stop(self):
        for _, writer in self.writers.items():
            writer.close()
        self.writers = {}

    def add_simple_value(self, name, value, family=None):
        if family in self.simple_values:
            family_values = self.simple_values[family]
        else:
            family_values = {}
            self.simple_values[family] = family_values
        family_values[name] = value

    def clear_simple_values(self, family=None):
        self.simple_values.pop(family, None)

    def add_scalar(self, name, tensor, family=None):
        # HACK - avoiding family being repeated twice in tensorboard tag
        # summary = tf.summary.scalar(name, tensor, family=family)
        if family is not None:
            name = f"{family}/{name}"
        summary = tf.summary.scalar(name, tensor, family=None)

        if family in self.summaries:
            family_summaries = self.summaries[family]
        else:
            family_summaries = []
            self.summaries[family] = family_summaries
        family_summaries.append(summary)
        return summary

    def add_histogram(self, name, values, family=None):
        summary = tf.summary.histogram(name, values, family=family)

        if family in self.summaries:
            family_summaries = self.summaries[family]
        else:
            family_summaries = []
            self.summaries[family] = family_summaries
        family_summaries.append(summary)
        return summary

    def get_fetch(self, family=None):
        if family in self.summary_fetches:
            return self.summary_fetches[family]
        if family in self.summaries:
            fetch = tf.summary.merge(self.summaries[family])
            self.summary_fetches[family] = fetch
            return fetch
        return None

    def prepare_fetches(self, fetches, family=None):
        summary_fetch = self.get_fetch(family)
        if summary_fetch is not None:
            fetches[SUMMARY_FETCH_ID] = summary_fetch

    def process_results(self, results, family=None, global_step=None):
        if SUMMARY_FETCH_ID in results:
            summary = results[SUMMARY_FETCH_ID]
            self.write(summary, family=family, global_step=global_step)
            results.pop(SUMMARY_FETCH_ID, None)

    def summary_scope(self, name, family=None):
        if family is None:
            return name
        # return f"{family}/{family}/{name}"  # HACK - see above
        return f"{family}/{name}"

    def write(self, results, family=None, global_step=None):
        # get writer
        path = os.path.abspath(self.path)
        if family is None:
            path = os.path.join(path, DEFAULT_SUBDIRECTORY)
        else:
            path = os.path.join(path, family)
        if family in self.writers:
            writer = self.writers[family]
        else:
            writer = tf.summary.FileWriter(path, **self.kwargs)
            self.writers[family] = writer

        # write results
        writer.add_summary(results, global_step=global_step)

        # write simple values
        if family in self.simple_values:
            family_values = self.simple_values[family]
            summary_values = []
            for name, value in family_values.items():
                tag = self.summary_scope(name, family)
                summary_values.append(tf.Summary.Value(tag=tag, simple_value=value))
            simple_summary = tf.Summary(value=summary_values)
            writer.add_summary(simple_summary, global_step=global_step)

        # flush writer
        writer.flush()


class NullSummaryWriter(object):
    def __init__(self, **kwargs):
        pass

    def start(self, **kwargs):
        pass

    def stop(self, **kwargs):
        pass

    def add_simple_value(self, **kwargs):
        pass

    def clear_simple_values(self, **kwargs):
        pass

    def add_scalar(self, **kwargs):
        return None

    def add_histogram(self, **kwargs):
        return None

    def get_fetch(self, **kwargs):
        return None

    def write(self, **kwargs):
        pass
