import time
import random
import queue
import threading
import numpy as np
import tensorflow as tf
import pyglet
from glearn.networks.context import num_global_parameters, num_trainable_parameters, NetworkContext
from glearn.policies import load_policy
from glearn.policies.random import RandomPolicy
from glearn.utils.printing import print_update, print_tabular, getch
from glearn.utils.profile import run_profile, open_profile
from glearn.utils.memory import print_virtual_memory, print_gpu_memory


class Trainer(NetworkContext):
    def __init__(self, config, evaluate_interval=10, keep_prob=1, **kwargs):
        super().__init__(config)

        self.policy = None
        self.policy_scope = None
        self.kwargs = kwargs

        self.debug_evaluate_pause = self.is_debugging("debug_evaluate_pause")
        self.debug_memory = self.is_debugging("debug_memory")
        self.debug_raise_numerics = self.is_debugging("debug_raise_numerics")

        self.debug_metrics = {}

        self.evaluate_interval = evaluate_interval
        self.keep_prob = keep_prob  # TODO - refactor

        self.epoch = 0
        self.batch = None

    def __str__(self):
        properties = [
            self.learning_type(),
            "training" if self.training else "evaluating",
        ]
        return f"{type(self).__name__}({', '.join(properties)})"

    def get_info(self):
        info = {
            "Description": str(self),
            "Total Global Parameters": num_global_parameters(),
            "Total Trainable Parameters": num_trainable_parameters(),
            "Evaluate Interval": self.evaluate_interval,
        }
        return info

    def print_info(self):
        # gather info
        info = {}
        info["Trainer"] = self.get_info()
        if self.policy:
            info["Policy"] = self.policy.get_info()

        # print a table with all this info
        print()
        print_tabular(info, grouped=True, show_type=False, color="white", bold=True)
        print()

    def learning_type(self):
        return "unknown"

    def reset(self, mode="train"):
        # override
        return 1

    def build_models(self, random=False):
        # build input feeds
        self.build_inputs()

        # build policy model
        self.build_policy(random=random)

        # build trainer model
        self.build_trainer()

    def build_inputs(self):
        with tf.variable_scope('feeds/'):
            if self.has_dataset:
                inputs = self.dataset.get_inputs(self)
                outputs = self.dataset.get_outputs(self)

                # # add feeds
                self.add_feed("X", inputs)
                self.add_feed("Y", outputs)
            else:
                inputs = self.create_feed("X", shape=(None,) + self.input.shape,
                                          dtype=self.input.dtype)
                outputs = self.create_feed("Y", shape=(None,) + self.output.shape,
                                           dtype=self.output.dtype)

            # add reference to interfaces
            inputs.interface = self.input
            outputs.interface = self.output

            # add fetches
            self.add_fetch("X", inputs, query="evaluate")
            self.add_fetch("Y", outputs, query="evaluate")

    def build_policy(self, random=False):
        # build policy, if defined
        if self.config.has("policy"):
            with self.variable_scope(self.policy_scope):
                if random:
                    self.policy = RandomPolicy(self.config, self)
                else:
                    self.policy = load_policy(self.config, self)

                self.policy.build_predict(self.get_feed("X"))

    def build_trainer(self):
        # build default policy optimize, if defined
        if self.policy:
            query = "policy_optimize"
            with self.variable_scope(query):
                # build policy loss
                loss = self.policy.build_loss(self.get_feed("Y"))

                # minimize policy loss
                self.policy.optimize_loss(loss, name=query)

    def prepare_feeds(self, query, feed_map):
        if self.policy:
            self.policy.prepare_default_feeds(query, feed_map)

        # dropout  FIXME - implement this like batch_norm
        if self.is_optimize(query):
            feed_map["dropout"] = self.keep_prob
        else:
            feed_map["dropout"] = 1

    def run(self, query, feed_map={}, render=True):
        if not isinstance(query, list):
            query = [query]

        # check numerics
        if self.debug_raise_numerics:
            if self.is_optimize(query):
                query.append("check_numerics")

        # run policy for query with feeds
        self.prepare_feeds(query, feed_map)
        results = super().run(query, feed_map)

        # view results
        if render:
            self.viewer.view_results(query, feed_map, results)

        return results

    def fetch(self, name, feed_map={}, squeeze=False):
        # shortcut to fetch a single query/value
        results = self.run(name, feed_map)
        fetch_result = results[name]

        if squeeze:
            fetch_result = np.squeeze(fetch_result)
        return fetch_result

    def predict(self, inputs):
        # input as feed map
        feed_map = {"X": [inputs]}

        # infer single prediction
        return self.run("predict", feed_map)

    def get_batch(self, mode="train"):
        # override
        return None

    def add_debug_metrics(self, queries):
        if self.debugging:
            self.debug_metrics.update(queries)

    def is_optimize(self, query):
        return np.any(["optimize" in query_name for query_name in query])

    def is_evaluate(self, query):
        return "evaluate" in query

    def pre_optimize(self, feed_map):
        pass

    def post_optimize(self, feed_map):
        pass

    def should_optimize(self):
        return self.training

    def get_optimize_query(self, batch):
        query = ["policy_optimize"]

        # debug metrics
        if self.debug_metrics:
            query.append("debug_metrics")

        return query

    def optimize(self, batch):
        # run default policy optimize query
        feed_map = batch.get_feeds()
        query = self.get_optimize_query(batch)
        results = self.run(query, feed_map)

        # print any debug metrics
        if self.debug_metrics:
            debug_values = results.pop("debug_metrics", None)
            if debug_values:
                print_tabular(debug_values)

        return results

    def optimize_and_report(self, batch):
        results = self.optimize(batch)

        # get current global step
        global_step = self.config.update_global_step()
        self.current_global_step = global_step

        # print log
        print_update(["Optimizing", f"Epoch: {self.epoch}", f"Global Step: {global_step}"])
        return results

    def evaluate_counter(self):
        # default to evaluate based on optimization step
        return self.current_global_step

    def should_evaluate(self):
        if not self.training:
            return True

        return (self.evaluate_counter() - (self.last_eval_counter or 0)) >= self.evaluate_interval

    def extra_evaluate_stats(self):
        # override
        return {}

    def evaluate(self):
        # Evaluate using the test dataset
        query = "evaluate"

        # prepare dataset partition
        evaluate_steps = self.reset(mode="test")

        # get batch data and desired query
        eval_start_time = time.time()
        averaged_results = {}
        report_step = random.randrange(evaluate_steps)
        report_results = None
        report_feed_map = None
        for step in range(evaluate_steps):
            print_update(["Evaluating", f"Progress: {step} / {evaluate_steps}"])

            reporting = step == report_step
            self.batch = self.get_batch(mode="test")
            feed_map = self.batch.get_feeds()

            # run evaluate query
            results = self.run(query, feed_map, render=reporting)

            # gather reporting results
            if reporting:
                report_results = results
                report_feed_map = {k: v for k, v in feed_map.items() if isinstance(k, str)}
            for k, v in results.items():
                if self.is_fetch(k, "evaluate") and \
                   (isinstance(v, float) or isinstance(v, int)):
                    if k in averaged_results:
                        averaged_results[k] += v
                    else:
                        averaged_results[k] = v

            if self.experiment_yield():
                return

        if report_results is not None:
            # log stats for current evaluation
            current_time = time.time()
            train_elapsed_time = current_time - self.start_time
            epoch_elapsed_time = current_time - self.epoch_start_time
            eval_elapsed_time = eval_start_time - (self.last_eval_time or self.start_time)
            eval_steps = self.current_global_step - (self.last_eval_step or 0)
            steps_per_second = eval_steps / eval_elapsed_time
            self.last_eval_time = current_time
            self.last_eval_step = self.current_global_step
            self.last_eval_counter = self.evaluate_counter()
            stats = {
                "global step": self.current_global_step,
                "training time": train_elapsed_time,
                "steps/second": steps_per_second,
                "epoch step": self.epoch_step,
                "epoch time": epoch_elapsed_time,
            }
            stats.update(self.extra_evaluate_stats())
            table = {f"Epoch: {self.epoch}": stats}

            # average evaluate results
            averaged_results = {k: v / evaluate_steps for k, v in averaged_results.items()}
            report_results.update(averaged_results)

            # remove None values
            report_results = {k: v for k, v in report_results.items() if v is not None}

            self.process_evaluate_results(report_results)

            # print inputs and results
            table["Feeds"] = report_feed_map
            table["Results"] = report_results

            # print tabular results
            print_tabular(table, grouped=True)

            # summaries
            self.summary.add_simple_value("steps_per_second", steps_per_second)

            # profile memory
            if self.debug_memory:
                print_virtual_memory()
                print_gpu_memory()

        # save model
        self.save()

        print()

    def evaluate_and_report(self):
        self.evaluate()

        # option to pause after each evaluation
        if self.debug_evaluate_pause:
            self.pause()

        # finally update global step
        if not self.training:
            self.current_global_step += 1

    def process_evaluate_results(self, results):
        # override
        pass

    def experiment_loop(self):
        # override
        pass

    def experiment_yield(self, flush_summary=False):
        # write summary results
        if flush_summary:
            self.summary.flush(global_step=self.current_global_step)

        while True:
            # check if experiment stopped
            if not self.running:
                return True

            # render frame
            if self.rendering:
                self.render()

            # handle terminal input
            self.handle_terminal_input()

            # loop while paused
            if self.paused:
                if not self.rendering:
                    # HACK until terminal input is working
                    print("Experiment paused.  Press Esc to exit, Space to resume...")
                    key = getch()
                    if ord(key) == 27:
                        self.on_key_press(pyglet.window.key.ESCAPE, None)
                    elif ord(key) == 32:
                        self.on_key_press(pyglet.window.key.SPACE, None)
                else:
                    time.sleep(0)
            else:
                break
        return False

    def execute(self, render=False, profile=False, random=False):
        try:
            self.running = True
            self.paused = False
            self.start_time = time.time()

            # initialize render viewer
            if self.rendering:
                self.init_viewer()

            # build models
            self.build_models(random=random)

            # debug metrics
            if self.debug_metrics:
                self.add_fetch("debug_metrics", self.debug_metrics)

            # check for invalid values in the current graph
            if self.debug_raise_numerics:
                self.add_fetch("check_numerics", tf.add_check_numerics_ops())

            # start session
            self.config.start_session()
            if self.policy:
                self.policy.start_session()

            # prepare viewer
            self.viewer.prepare(self)

            # print experiment info
            self.print_info()

            # get current global step, and prepare evaluation counters
            global_step = tf.train.global_step(self.sess, self.global_step)
            self.current_global_step = global_step
            self.last_eval_time = None
            self.last_eval_step = self.current_global_step
            self.last_eval_counter = self.evaluate_counter()

            # start listening for terminal input
            self.start_terminal_input()

            if profile:
                # profile experiment loop
                profile_path = run_profile(self.experiment_loop, self.config)

                # show profiling results
                open_profile(profile_path)
            else:
                # run training loop without profiling
                self.experiment_loop()
        finally:
            self.stop_terminal_input()

            # cleanup session after evaluation
            if self.policy:
                self.policy.stop_session()
            self.config.close_session()

            # cleanup render viewer
            if self.rendering:
                self.close_viewer()

    def init_viewer(self):
        # register for events from viewer
        self.viewer.add_listener(self)

    def close_viewer(self):
        self.viewer.close()

    def render(self, mode="human"):
        self.viewer.render()

    def pause(self):
        self.paused = True

    def start_terminal_input(self):
        # FIXME - make this work (...and refactor out of this class)
        return

        self.terminal_input = queue.Queue()
        self.terminal_input_stop = threading.Event()

        def listen_terminal_input():
            while not self.terminal_input_stop.is_set():
                key = getch(blocking=False)
                if key != '':
                    self.terminal_input.put_nowait(key)

                time.sleep(0.1)

        self.terminal_input_thread = threading.Thread(target=listen_terminal_input)
        self.terminal_input_thread.start()

    def stop_terminal_input(self):
        # FIXME - make this work
        return

        if self.terminal_input_thread:
            self.terminal_input_stop.set()
            self.terminal_input_thread.join()

    def handle_terminal_input(self):
        # FIXME - make this work
        return

        while not self.terminal_input.empty():
            key = self.terminal_input.get()
            self.handle_terminal_key(key)

    def handle_terminal_key(self, key):
        print(f"KEY PRESS: '{key}' {ord(key)}")
        # TODO - forward these to on_key_press()

    def on_key_press(self, key, modifiers):
        # feature visualization keys
        if key == pyglet.window.key.ESCAPE:
            self.warning("Experiment cancelled by user")
            self.running = False
        elif key == pyglet.window.key.SPACE:
            self.warning(f"Experiment {'unpaused' if self.paused else 'paused'} by user")
            self.paused = not self.paused
