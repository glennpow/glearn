import os
import shutil
import tensorflow as tf


class SummaryWriter(object):
    def __init__(self, path):
        self.path = path

        self.summaries = {}
        self.summary_fetches = {}
        self.writers = {}

    def start(self, append=False, **kwargs):
        self.kwargs = kwargs

        if not append:
            shutil.rmtree(self.path)

    def stop(self):
        for _, writer in self.writers.items():
            writer.close()
        self.writers = {}

    def add_scalar(self, name, tensor, family=None):
        summary = tf.summary.scalar(name, tensor, family=family)

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

    def write(self, results, family=None, global_step=None):
        family = family or 'events'
        if family in self.writers:
            writer = self.writers[family]
        else:
            path = os.path.join(os.path.abspath(self.path), family)
            writer = tf.summary.FileWriter(path, **self.kwargs)
            self.writers[family] = writer

        writer.add_summary(results, global_step=global_step)
        writer.flush()
