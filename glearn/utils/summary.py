import os
import shutil
import tensorflow as tf
from subprocess import Popen


SUMMARY_KEY_PREFIX = "_summary_"
DEFAULT_SUBDIRECTORY = "events"


class SummaryWriter(object):
    class Results(object):
        def __init__(self, family):
            self.family = family
            self.results = []
            self.values = {}

    def __init__(self, path):
        self.path = path

        self.summaries = {}
        self.summary_fetches = {}
        self.summary_results = {}
        self.writers = {}
        self.server = None

    def start(self, append=False, server=False, **kwargs):
        self.kwargs = kwargs

        # prepare directory
        if not append:
            shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)

        # prepare server
        if server:
            command = ["tensorboard", "--logdir", self.path]
            self.server = Popen(command)
            print(f"Started tensorboard server with PID: {self.server.pid}")

    def stop(self):
        for _, writer in self.writers.items():
            writer.close()
        self.writers = {}

        # stop server
        if self.server is not None:
            self.server.terminate()
            self.server = None

    def get_summary_results(self, family):
        if family not in self.summary_results:
            self.summary_results[family] = self.Results(family)
        return self.summary_results[family]

    def add_simple_value(self, name, value, family=None):
        summary_results = self.get_summary_results(family)
        summary_results.values[name] = value  # TODO - average

        # if family in self.simple_values:
        #     family_values = self.simple_values[family]
        # else:
        #     family_values = {}
        #     self.simple_values[family] = family_values
        # family_values[name] = value
        # if family not in self.summary_families:
        #     self.summary_families.append(family)

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
        # HACK - avoiding family being repeated twice in tensorboard tag
        # summary = tf.summary.histogram(name, values, family=family)
        if family is not None:
            name = f"{family}/{name}"
        summary = tf.summary.histogram(name, values, family=None)

        if family in self.summaries:
            family_summaries = self.summaries[family]
        else:
            family_summaries = []
            self.summaries[family] = family_summaries
        family_summaries.append(summary)
        return summary

    def add_activation(self, tensor, family=None):
        if tensor is None:
            return
        name = tensor.op.name
        self.add_histogram(f"{name}/activation", tensor, family=family)
        self.add_scalar(f"{name}/sparsity", tf.nn.zero_fraction(tensor), family=family)

    def add_gradients(self, grads_tvars, family=None):
        for grad, tvar in grads_tvars:
            if grad is None:
                continue
            name = tvar.op.name
            self.add_histogram(f"{name}/gradient", grad, family=family)

    def get_fetch(self, family=None):
        if family in self.summary_fetches:
            return self.summary_fetches[family]
        if family in self.summaries:
            fetch = tf.summary.merge(self.summaries[family])
            self.summary_fetches[family] = fetch
            return fetch
        return None

    def get_family_key(self, family=None):
        if family is None:
            return SUMMARY_KEY_PREFIX
        return f"{SUMMARY_KEY_PREFIX}{family}"

    def prepare_fetches(self, fetches, families=None):
        if not isinstance(families, list):
            families = [families]
        for family in families:
            fetch = self.get_fetch(family)
            if fetch is not None:
                fetches[self.get_family_key(family)] = fetch

    def process_results(self, results):
        results_keys = list(results.keys())
        for key in results_keys:
            if key.startswith(SUMMARY_KEY_PREFIX):
                family = key[len(SUMMARY_KEY_PREFIX):]
                if len(family) == 0:
                    family = None
                family_results = results.pop(key, None)
                summary_results = self.get_summary_results(family)
                summary_results.results.append(family_results)

                # if family in self.summary_results:
                #     raise Exception(f"Clobbering summary results: '{family}'")
                # else:
                #     self.summary_results[family] = family_results
                # if family not in self.summary_families:
                #     self.summary_families.append(family)

    def summary_scope(self, name, family=None):
        if family is None:
            return name
        # return f"{family}/{family}/{name}"  # HACK - see above
        return f"{family}/{name}"

    def flush(self, global_step=None):
        # flush summary buffers
        for family, summary_results in self.summary_results.items():
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
            if len(summary_results.results) > 0:
                summary = summary_results.results[0]  # TODO - average
                writer.add_summary(summary, global_step=global_step)

            # write simple values
            summary_values = []
            for name, value in summary_results.values.items():
                tag = self.summary_scope(name, family)
                summary_values.append(tf.Summary.Value(tag=tag, simple_value=value))
            simple_summary = tf.Summary(value=summary_values)
            writer.add_summary(simple_summary, global_step=global_step)

            # flush writer
            writer.flush()
        # reset buffers
        self.summary_results = {}


class NullSummaryWriter(object):
    def __init__(self, **kwargs):
        pass

    def start(self, **kwargs):
        pass

    def stop(self, **kwargs):
        pass

    def add_simple_value(self, **kwargs):
        pass

    def add_scalar(self, **kwargs):
        return None

    def add_histogram(self, **kwargs):
        return None

    def get_fetch(self, **kwargs):
        return None

    def prepare_fetches(self, **kwargs):
        pass

    def process_results(self, **kwargs):
        pass

    def flush(self, **kwargs):
        pass
