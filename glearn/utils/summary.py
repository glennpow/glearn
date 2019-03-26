import os
import atexit
import numpy as np
import tensorflow as tf
from subprocess import Popen
from glearn.utils.log import log, log_warning
from glearn.utils.path import remove_empty_dirs


SUMMARY_KEY_PREFIX = "_summary_"
DEFAULT_EVALUATE_QUERY = "evaluate"
DEFAULT_EXPERIMENT_QUERY = "experiment"


class SummaryWriter(object):
    class Results(object):
        def __init__(self, query):
            self.query = query
            self.results = []
            self.simple_summaries = {}

    def __init__(self, config):
        self.config = config
        self.server = None

    @property
    def sess(self):
        return self.config.sess

    def start(self, **kwargs):
        self.summary_path = self.config.summary_path

        self.summaries = {}
        self.summary_fetches = {}
        self.summary_results = {}
        self.run_metadatas = {}
        self.writers = {}

        # get graph
        self.kwargs = kwargs
        if "graph" not in self.kwargs:
            self.kwargs["graph"] = self.sess.graph

        server = self.config.get("tensorboard", False)
        if server:
            # start tensorboard server
            if self.server is None:
                # if this is the first evaluation, clean all experiment summaries
                self.clean_summaries(self.config.tensorboard_path)

                self.start_server()
        else:
            # clean only this evaluation's summaries
            self.clean_summaries(self.summary_path)
        os.makedirs(self.summary_path, exist_ok=True)

    def stop(self):
        for _, writer in self.writers.items():
            writer.close()
        self.writers = {}

    def start_server(self):
        if self.server is None:
            # start tensorboard server
            path = self.config.tensorboard_path
            port = 6006
            self.server = Popen(["tensorboard", "--logdir", path])
            atexit.register(self.stop_server)

            log(f"Started tensorboard server: http://{self.config.ip}:{port}  ({path})")

    def stop_server(self):
        # stop tensorboard server
        if self.server is not None:
            log(f"Stopping tensorboard server")

            self.server.terminate()
            self.server = None

    def clean_summaries(self, path):
        # delete all events.out.tfevents files, and cleanup empty dirs
        for root, dirs, files in os.walk(path):
            for sub_path in files:
                if sub_path.startswith("events.out.tfevents"):
                    os.remove(os.path.join(root, sub_path))
        remove_empty_dirs(path)

    def get_summary_results(self, query):
        if query not in self.summary_results:
            self.summary_results[query] = self.Results(query)
        return self.summary_results[query]

    def add_simple_summary(self, name, query=None, allow_overwrite=False, **kwargs):
        query = query or DEFAULT_EXPERIMENT_QUERY
        summary_results = self.get_summary_results(query)
        tag = self.summary_scope(name, query)

        if not allow_overwrite and tag in summary_results.simple_summaries:
            log_warning(f"Overwriting simple summary value: {tag}")
        summary_results.simple_summaries[tag] = tf.Summary.Value(tag=tag, **kwargs)

    def add_simple_value(self, name, value, query=None, allow_overwrite=False):
        self.add_simple_summary(name, simple_value=value, query=query,
                                allow_overwrite=allow_overwrite)

    def set_simple_value(self, name, value, query=None):
        self.add_simple_value(name, value, query=query, allow_overwrite=True)

    def add_summary_value(self, name, summary, query=None):
        query = query or DEFAULT_EVALUATE_QUERY
        if query in self.summaries:
            query_summaries = self.summaries[query]
        else:
            query_summaries = []
            self.summaries[query] = query_summaries
        query_summaries.append(summary)
        return summary

    def add_scalar(self, name, tensor, query=None):
        summary = tf.summary.scalar(name, tensor)
        return self.add_summary_value(name, summary, query=query)

    def add_histogram(self, name, values, query=None):
        summary = tf.summary.histogram(name, values)
        return self.add_summary_value(name, summary, query=query)

    def add_activation(self, tensor, query=None):
        if tensor is None:
            return
        name = tensor.op.name
        self.add_histogram(f"{name}/activation", tensor, query=query)
        self.add_scalar(f"{name}/sparsity", tf.nn.zero_fraction(tensor), query=query)

    def add_variables(self, tvars, query=None):
        for tvar in tvars:
            name = tvar.op.name
            self.add_histogram(f"{name}/value", tvar, query=query)

    def add_gradients(self, grads_tvars, query=None):
        for grad, tvar in grads_tvars:
            if grad is None:
                continue
            name = tvar.op.name
            self.add_histogram(f"{name}/gradient", grad, query=query)

    def add_images(self, name, images, max_outputs=3, query=None):
        summary = tf.summary.image(name, images, max_outputs=max_outputs)
        return self.add_summary_value(name, summary, query=query)

    def add_simple_images(self, name, images, max_outputs=3, query=None, allow_overwrite=False):
        # matplotlib allows image encoding.  the imports are here since they are slow.
        import io
        import matplotlib
        try:
            # first try this
            matplotlib.use('TkAgg')
        except Exception:
            # fallback backend
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # convert image to 3-channel
        if images.shape[-1] == 1:
            images = np.stack((np.squeeze(images, axis=-1),) * 3, axis=-1)

        for i, image in enumerate(images):
            im_bytes = io.BytesIO()
            plt.imsave(im_bytes, image, format='png')
            summary_image = tf.Summary.Image(encoded_image_string=im_bytes.getvalue())
            self.add_simple_summary(f"{name}/{i}", image=summary_image, query=query)

    def set_simple_images(self, name, images, max_outputs=3, query=None):
        self.add_simple_images(name, images, max_outputs=max_outputs, query=query,
                               allow_overwrite=True)

    def add_text(self, name, tensor, query=None):
        summary = tf.summary.text(name, tensor)
        return self.add_summary_value(name, summary, query=query)

    def write_text(self, name, tensor, query=None):
        # TODO - convert to: add_simple_text(...)
        query = query or DEFAULT_EXPERIMENT_QUERY
        tag = self.summary_scope(name, query)
        summary = tf.summary.text(tag, tensor)
        self.config.sess.run({self.get_query_key(query): summary})

    def add_run_metadata(self, run_metadata, query=None):
        self.run_metadatas[query] = run_metadata

    def get_fetch(self, query=None):
        if query in self.summary_fetches:
            return self.summary_fetches[query]
        if query in self.summaries:
            fetch = tf.summary.merge(self.summaries[query])
            self.summary_fetches[query] = fetch
            return fetch
        return None

    def get_query_key(self, query=None):
        if query is None:
            return SUMMARY_KEY_PREFIX
        return f"{SUMMARY_KEY_PREFIX}{query}"

    def prepare_fetches(self, fetches, queries=None):
        if not isinstance(queries, list):
            queries = [queries]
        for query in queries:
            fetch = self.get_fetch(query)
            if fetch is not None:
                fetches[self.get_query_key(query)] = fetch

    def process_results(self, results):
        results_keys = list(results.keys())
        for key in results_keys:
            if key.startswith(SUMMARY_KEY_PREFIX):
                query = key[len(SUMMARY_KEY_PREFIX):]
                if len(query) == 0:
                    query = None
                query_results = results.pop(key, None)
                summary_results = self.get_summary_results(query)
                summary_results.results.append(query_results)

    def summary_scope(self, name, query=None):
        if query is None:
            return name
        return f"{query}/{name}"

    def flush(self, global_step=None):
        # collect all relevant queries
        queries = set(list(self.summary_results.keys()) + list(self.run_metadatas.keys()))

        # flush summary data
        for query in queries:
            # get writer
            path = os.path.abspath(self.summary_path)
            if query is None:
                query = DEFAULT_EVALUATE_QUERY
            path = os.path.join(path, query)
            if query in self.writers:
                writer = self.writers[query]
            else:
                writer = tf.summary.FileWriter(path, **self.kwargs)
                self.writers[query] = writer

            # write any summary results for query
            summary_results = self.summary_results.pop(query, None)
            if summary_results is not None:
                # write results
                if len(summary_results.results) > 0:
                    summary = summary_results.results[0]  # TODO - average
                    writer.add_summary(summary, global_step=global_step)

                # write simple values
                summary_values = list(summary_results.simple_summaries.values())
                simple_summary = tf.Summary(value=summary_values)
                writer.add_summary(simple_summary, global_step=global_step)

            # write any metadata results for query
            run_metadata = self.run_metadatas.pop(query, None)
            if run_metadata is not None:
                if query is not None:
                    tag = f"{query}/step{global_step}"
                else:
                    tag = f"step{global_step}"
                writer.add_run_metadata(run_metadata, tag, global_step)

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

    def add_scalar(self, **kwargs):
        return None

    def add_histogram(self, **kwargs):
        return None

    def add_activation(self, **kwargs):
        return None

    def add_gradients(self, **kwargs):
        return None

    def get_fetch(self, **kwargs):
        return None

    def prepare_fetches(self, **kwargs):
        pass

    def process_results(self, **kwargs):
        pass

    def flush(self, **kwargs):
        pass
