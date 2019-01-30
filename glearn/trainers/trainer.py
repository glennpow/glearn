import time
import random
import numpy as np
import tensorflow as tf
import pyglet
from glearn.datasets.dataset import Transition, transition_batch
from glearn.utils.collections import intersects
from glearn.utils.config import Configurable
from glearn.networks.context import num_global_parameters, num_trainable_parameters, \
    saveable_objects
from glearn.utils.printing import print_update, print_tabular
from glearn.utils.profile import run_profile, open_profile
from glearn.utils.memory import print_virtual_memory, print_gpu_memory


class Trainer(Configurable):
    def __init__(self, config, policy, epochs=None, episodes=None,
                 max_episode_time=None, min_episode_reward=None, evaluate_interval=10, epsilon=0,
                 keep_prob=1, **kwargs):
        super().__init__(config)

        self.policy = policy
        self.kwargs = kwargs

        self.batch_size = self.config.get("batch_size", 1)
        self.debug_gradients = self.config.is_debugging("debug_gradients")
        self.debug_memory = self.config.is_debugging("debug_memory")
        self.debug_numerics = self.config.is_debugging("debug_numerics")

        self.epochs = epochs
        self.episodes = episodes
        self.max_episode_time = max_episode_time
        self.min_episode_reward = min_episode_reward
        self.evaluate_interval = evaluate_interval

        self.epsilon = epsilon
        self.keep_prob = keep_prob

        self.epoch_step = 0
        self.episode_step = 0
        self.observation = None
        self.transitions = []
        self.episode_reward = 0
        self.batch = None

        if self.rendering:
            self.init_viewer()
        self.init_optimizer()

    def __str__(self):
        properties = [
            "supervised" if self.supervised else "reinforcement",
            "training" if self.training else "evaluating",
        ]
        return f"{type(self).__name__}({', '.join(properties)})"

    def get_info(self):
        return {
            "Description": str(self),
            "Total Global Parameters": num_global_parameters(),
            "Total Trainable Parameters": num_trainable_parameters(),
            "Total Saveable Objects": len(saveable_objects()),
        }

    def print_info(self):
        # gather info
        info = {}
        if self.supervised:
            info["Dataset"] = self.dataset.get_info()
        else:
            info["Environment"] = {
                "Description": self.env.name,
                "Input": self.input,
                "Output": self.output,
            }
        info["Trainer"] = self.get_info()
        info["Policy"] = self.policy.get_info()

        # print a table with all this info
        print()
        print_tabular(info, grouped=True, show_type=False, color="white", bold=True)
        print()

    def reset(self):
        # reset env and episode
        if self.env is not None:
            self.observation = self.env.reset()
            self.transitions = []
            self.episode_reward = 0

    def get_interval_size(self):
        if self.supervised:
            # interval size is epoch worth of steps
            return self.dataset.get_epoch_size(mode="train")
        else:
            # interval is a single episode
            return 1

    def optimize_loss(self, loss, query, definition=None, update_global_step=True):
        # default definition
        if definition is None:
            definition = self.kwargs

        global_step = self.global_step

        learning_rate = definition.get("learning_rate", 1e-2)

        # learning rate decay
        lr_decay = definition.get("lr_decay", None)
        if lr_decay is not None:
            lr_decay_intervals = definition.get("lr_decay_intervals", 1)
            decay_steps = int(lr_decay_intervals * self.get_interval_size())
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps,
                                                       lr_decay, staircase=True)

        # create optimizer
        optimizer_name = definition.get("optimizer", "sgd")
        if optimizer_name == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_name == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise Exception(f"Unknown optimizer type specified in config: {optimizer_name}")

        # get gradients and trainable variables
        grads_tvars = optimizer.compute_gradients(loss)

        # check if we require unzipping grad/vars
        max_grad_norm = definition.get("max_grad_norm", None)
        require_unzip = self.debug_gradients or max_grad_norm is not None
        if require_unzip:
            grads, tvars = zip(*grads_tvars)

        # apply gradient clipping
        if max_grad_norm is not None:
            with tf.name_scope("clipped_gradients"):
                clipped_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

                # metric to observe clipped gradient ratio
                if self.debugging:
                    safe_grads = [g if g is not None else 0 for g in grads]
                    safe_clipped_grads = [g if g is not None else 0 for g in clipped_grads]
                    unequal = [tf.reduce_mean(tf.cast(tf.not_equal(g, clipped_g), tf.float32))
                               for g, clipped_g in list(zip(safe_grads, safe_clipped_grads))]
                    clipped_ratio = tf.reduce_mean(unequal)
                    if max_grad_norm is not None:
                        self.summary.add_scalar("clipped_ratio", clipped_ratio, query)

                grads = clipped_grads

        if require_unzip:
            grads_tvars = zip(grads, tvars)

        # apply gradients
        optimizer_global_step = global_step if update_global_step else None
        optimize = optimizer.apply_gradients(grads_tvars, global_step=optimizer_global_step)

        # add learning rate and gradient summaries
        self.summary.add_scalar("learning_rate", learning_rate, query)
        if self.debug_gradients:
            self.summary.add_gradients(zip(grads, tvars), "evaluate")

        return optimize

    def init_optimizer(self):
        # overwrite
        pass

    def prepare_feeds(self, queries, feed_map):
        self.policy.prepare_default_feeds(queries, feed_map)

        # dropout
        if intersects(["policy_optimize", "value_optimize"], queries):
            feed_map["dropout"] = self.keep_prob
        else:
            feed_map["dropout"] = 1

        return feed_map

    def run(self, queries, feed_map={}, render=True):
        if not isinstance(queries, list):
            queries = [queries]

        # run policy for queries with feeds
        feed_map = self.prepare_feeds(queries, feed_map)
        results = self.policy.run(queries, feed_map)

        # view results
        if render:
            self.viewer.view_results(queries, feed_map, results)

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

        # get desired queries
        queries = ["predict"]

        # evaluate and extract single prediction
        results = self.run(queries, feed_map)
        return results["predict"][0]

    def action(self):
        # decaying epsilon-greedy
        epsilon = self.epsilon if self.training else 0
        if isinstance(epsilon, list):
            t = min(1, self.current_global_step / epsilon[2])
            epsilon = t * (epsilon[1] - epsilon[0]) + epsilon[0]
            self.summary.add_simple_value("epsilon", epsilon, "experiment")

        # get action
        if epsilon > 0 and np.random.random() < epsilon:
            # choose epsilon-greedy random action  (TODO - could implement this in tf)
            return self.output.sample()
        else:
            # choose optimal policy action
            return self.predict(self.observation)

    def rollout(self):
        # get action
        action = self.action()

        # perform action
        new_observation, reward, done, info = self.env.step(action)

        # build and process transition
        transition = Transition(self.observation, action, reward, new_observation, done, info)
        self.process_transition(transition)

        # record transition
        self.transitions.append(transition)

        # update stats
        self.observation = new_observation
        self.episode_reward += transition.reward
        return transition

    def process_transition(self, transition):
        pass

    def get_batch(self, mode="train"):
        if self.supervised:
            # supervised batch of samples
            return self.dataset.get_batch(mode=mode)
        else:
            # unsupervised experience replay batch of samples
            batch = transition_batch(self.transitions[:self.batch_size])
            feed_map = {
                "X": batch.inputs,
                "Y": batch.outputs,
            }
            return batch, feed_map

    def pre_optimize(self, feed_map):
        pass

    def post_optimize(self, feed_map):
        pass

    def should_optimize(self):
        if not self.training:
            return False
        if self.supervised:
            return True
        else:
            return len(self.transitions) >= self.batch_size

    def optimize(self):
        # get either supervised or unsupervised batch data and feeds
        self.batch, feed_map = self.get_batch()

        # run all desired queries
        results = self.run(["policy_optimize"], feed_map)

        # get current global step (TODO: add global_step fetch into above run)
        global_step = tf.train.global_step(self.sess, self.global_step)
        self.current_global_step = global_step

        # print log
        iteration_name = "Epoch" if self.supervised else "Episode"
        iteration = self.epoch if self.supervised else self.episode
        print_update(f"Optimizing | {iteration_name}: {iteration} | Global Step: {global_step}")
        return results

    def should_evaluate(self):
        return self.current_global_step % self.evaluate_interval == 0

    def evaluate(self, experiment_yield):
        # Evaluate using the test dataset
        queries = ["evaluate"]

        # prepare dataset partition
        if self.supervised:
            epoch_size = self.dataset.reset(mode="test")
        else:
            epoch_size = 1

        # get batch data and desired queries
        eval_start_time = time.time()
        averaged_results = {}
        report_step = random.randrange(epoch_size)
        report_results = None
        report_feed_map = None
        for step in range(epoch_size):
            print_update(f"Evaluating | Progress: {step}/{epoch_size}")

            reporting = step == report_step
            self.batch, feed_map = self.get_batch(mode="test")

            # run evaluate queries
            results = self.run(queries, feed_map, render=reporting)

            # gather reporting results
            if reporting:
                report_results = results
                report_feed_map = {k: v for k, v in feed_map.items() if isinstance(k, str)}
            for k, v in results.items():
                if self.policy.is_fetch(k, "evaluate") and \
                   (isinstance(v, float) or isinstance(v, int)):
                    if k in averaged_results:
                        averaged_results[k] += v
                    else:
                        averaged_results[k] = v

            if experiment_yield():
                return

        if report_results is not None:
            # log stats for current evaluation
            current_time = time.time()
            train_elapsed_time = current_time - self.start_time
            iteration_elapsed_time = current_time - self.iteration_start_time
            eval_elapsed_time = eval_start_time - (self.last_eval_time or self.start_time)
            eval_steps = self.current_global_step - (self.last_eval_step or 0)
            steps_per_second = eval_steps / eval_elapsed_time
            self.last_eval_time = current_time
            self.last_eval_step = self.current_global_step
            stats = {
                "global step": self.current_global_step,
                "training time": train_elapsed_time,
                "steps/second": steps_per_second,
            }
            if self.supervised:
                stats.update({
                    "epoch step": self.epoch_step,
                    "epoch time": iteration_elapsed_time,
                })
                table = {f"Epoch: {self.epoch}": stats}
            else:
                stats.update({
                    "episode steps": self.episode_step,
                    "episode time": iteration_elapsed_time,
                    "reward": self.episode_reward,
                    "max reward": self.max_episode_reward,
                })
                table = {f"Episode: {self.episode}": stats}

            # average evaluate results
            averaged_results = {k: v / epoch_size for k, v in averaged_results.items()}
            report_results.update(averaged_results)

            # remove None values
            report_results = {k: v for k, v in report_results.items() if v is not None}

            # print inputs and results
            table["Inputs"] = report_feed_map
            table["Evaluation"] = report_results

            # print tabular results
            print_tabular(table, grouped=True)

            # summaries
            self.summary.add_simple_value("steps_per_second", steps_per_second, "experiment")

            # profile memory
            if self.debug_memory:
                print_virtual_memory()
                print_gpu_memory()

        # save model
        self.config.save()

        print()

    def start(self, render=False, profile=False):
        self.max_episode_reward = None
        self.running = True
        self.paused = False
        self.start_time = time.time()

        # check for invalid values in the current graph
        if self.debug_numerics:
            self.policy.set_fetch("check", tf.add_check_numerics_ops(), "policy_optimize")

        # prepare viewer
        self.viewer.prepare(self)

        # print experiment info
        self.print_info()

        # yield after each experiment iteration
        def experiment_yield(flush_summary=False):
            # write summary results
            if flush_summary:
                self.policy.summary.flush(global_step=self.current_global_step)

            while True:
                # check if experiment stopped
                if not self.running:
                    return True

                # render frame
                if render:
                    self.render()

                # loop while paused
                if self.paused:
                    time.sleep(0)
                else:
                    break
            return False

        # main experiment loop
        def experiment_loop():
            # get current global step, and prepare evaluation counters
            global_step = tf.train.global_step(self.sess, self.global_step)
            self.current_global_step = global_step
            self.last_eval_time = None
            self.last_eval_step = self.current_global_step

            # do supervised or reinforcement loop
            if self.supervised:
                self.experiment_supervised_loop(experiment_yield)
            else:
                self.experiment_reinforcement_loop(experiment_yield)

        if profile:
            # profile experiment loop
            profile_path = run_profile(experiment_loop, self.config)

            # show profiling results
            open_profile(profile_path)
        else:
            # run training loop without profiling
            experiment_loop()

    def experiment_supervised_loop(self, experiment_yield):
        # supervised learning
        if self.training:
            # train desired epochs
            epoch = 0
            while self.epochs is None or epoch < self.epochs:
                # start current epoch
                epoch_size = self.dataset.reset(mode="train")
                self.epoch = epoch
                self.iteration_start_time = time.time()

                # epoch summary
                global_epoch = self.current_global_step / epoch_size
                self.summary.add_simple_value("epoch", global_epoch, "experiment")

                for step in range(epoch_size):
                    # epoch time
                    self.epoch_step = step + 1

                    # optimize batch
                    self.optimize()

                    # evaluate if time to do so
                    if self.should_evaluate():
                        self.evaluate(experiment_yield)

                    if experiment_yield(True):
                        return
                epoch += 1
        else:
            # evaluate single epoch
            self.epoch = 0
            self.iteration_start_time = time.time()
            self.evaluate(experiment_yield)

    def experiment_reinforcement_loop(self, experiment_yield):
        # reinforcement learning
        episode = 0
        reset_evaluate = True

        while self.episodes is None or episode < self.episodes:
            # start current episode
            self.iteration_start_time = time.time()
            self.episode = episode
            self.reset()
            self.episode_step = 0

            # episode count summary
            self.summary.add_simple_value("episode", episode, "experiment")

            if reset_evaluate:
                episode_rewards = []
                episode_times = []
                episode_steps = []

            while self.running:
                if experiment_yield(True):
                    return

                # rollout
                transition = self.rollout()
                done = transition.done
                self.episode_step += 1

                # episode time
                current_time = time.time()
                episode_time = current_time - self.iteration_start_time
                if self.max_episode_time is not None:
                    # episode timeout
                    if episode_time > self.max_episode_time:
                        done = True

                # episode performance
                if self.min_episode_reward is not None:
                    # episode poor performance
                    if self.episode_reward < self.min_episode_reward:
                        done = True

                if done:
                    # track max episode reward
                    if self.max_episode_reward is None \
                       or self.episode_reward > self.max_episode_reward:
                        self.max_episode_reward = self.episode_reward

                    # track episode reward, time and steps
                    episode_rewards.append(self.episode_reward)
                    episode_time = time.time() - self.iteration_start_time
                    episode_times.append(episode_time)
                    episode_steps.append(self.episode_step)

                    # optimize when enough transitions have been gathered
                    if self.should_optimize():
                        self.optimize()

                    # evaluate if time to do so
                    if self.should_evaluate():
                        self.evaluate(experiment_yield)

                        # episode summary values
                        self.summary.add_simple_value("episode_reward", np.mean(episode_rewards),
                                                      "experiment")
                        self.summary.add_simple_value("max_episode_reward",
                                                      self.max_episode_reward, "experiment")
                        self.summary.add_simple_value("episode_time", np.mean(episode_times),
                                                      "experiment")
                        self.summary.add_simple_value("episode_steps", np.mean(episode_steps),
                                                      "experiment")

                        # env summary values
                        if hasattr(self.env, "evaluate"):
                            self.env.evaluate(self.policy)

                        reset_evaluate = True
                    break

            if not self.running:
                return

            episode += 1

    @property
    def viewer(self):
        return self.config.viewer

    @property
    def rendering(self):
        return self.viewer.rendering

    def init_viewer(self):
        # register for events from viewer
        self.viewer.add_listener(self)

    def render(self, mode="human"):
        if self.env is not None:
            self.env.render(mode=mode)
        self.viewer.render()

    def on_key_press(self, key, modifiers):
        # feature visualization keys
        if key == pyglet.window.key.ESCAPE:
            self.warning("Experiment cancelled by user")
            self.viewer.close()
            self.running = False
        elif key == pyglet.window.key.SPACE:
            self.warning(f"Experiment {'unpaused' if self.paused else 'paused'} by user")
            self.paused = not self.paused
