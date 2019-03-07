import time
import random
import numpy as np
import tensorflow as tf
import pyglet
from glearn.data.transition import Transition, TransitionBatch
from glearn.networks.context import num_global_parameters, num_trainable_parameters, \
    saveable_objects, NetworkContext
from glearn.policies import load_policy
from glearn.policies.random import RandomPolicy
from glearn.utils.collections import intersects
from glearn.utils.printing import print_update, print_tabular
from glearn.utils.profile import run_profile, open_profile
from glearn.utils.memory import print_virtual_memory, print_gpu_memory


class Trainer(NetworkContext):
    def __init__(self, config, epochs=None, episodes=None,
                 max_episode_time=None, min_episode_reward=None, evaluate_interval=10, epsilon=0,
                 keep_prob=1, **kwargs):
        super().__init__(config)

        self.policy = None
        self.policy_scope = None
        self.kwargs = kwargs

        self.batch_size = self.config.get("batch_size", 1)
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
        self.state = None
        self.transitions = []
        self.episode_reward = 0
        self.batch = None

    def __str__(self):
        properties = [
            "training" if self.training else "evaluating",
            self.learning_type(),
        ]
        if self.has_env:
            properties.append("on-policy" if self.on_policy() else "off-policy")
        return f"{type(self).__name__}({', '.join(properties)})"

    def get_info(self):
        info = {
            "Description": str(self),
            "Total Global Parameters": num_global_parameters(),
            "Total Trainable Parameters": num_trainable_parameters(),
            "Total Saveable Objects": len(saveable_objects()),
            "Evaluate Interval": self.evaluate_interval,
        }
        if self.has_dataset:
            info.update({
                "Epochs": self.epochs,
            })
        else:
            info.update({
                "Episodes": self.episodes,
                "Max Episode Time": self.max_episode_time,
                "Min Episode Reward": self.min_episode_reward,
            })
        return info

    def print_info(self):
        # gather info
        info = {}
        if self.has_dataset:
            info["Dataset"] = self.dataset.get_info()
        else:
            info["Environment"] = {
                "Description": self.env.name,
                "Input": self.input,
                "Output": self.output,
            }
        info["Trainer"] = self.get_info()
        if self.policy:
            info["Policy"] = self.policy.get_info()

        # print a table with all this info
        print()
        print_tabular(info, grouped=True, show_type=False, color="white", bold=True)
        print()

    def learning_type(self):
        return "supervised" if self.has_dataset else "reinforcement"

    def on_policy(self):
        # override
        return False

    def off_policy(self):
        return not self.on_policy()

    def reset(self):
        # reset env and episode
        if self.env is not None:
            self.state = self.env.reset()
            self.episode_reward = 0

    def build_models(self, random=False):
        # initialize render viewer
        if self.rendering:
            self.init_viewer()

        # build policy model
        if self.policy_scope is not None:
            with tf.variable_scope(f"{self.policy_scope}/"):
                self.build_policy(random=random)
        else:
            self.build_policy()

        # build trainer model
        self.build_trainer()

    def build_policy(self, random=False):
        # create policy, if defined
        if self.config.has("policy"):
            if random:
                self.policy = RandomPolicy(self.config, self)
            else:
                self.policy = load_policy(self.config, self)

            self.policy.build_policy()

    def build_trainer(self):
        # overwrite
        pass

    def prepare_feeds(self, queries, feed_map):
        if self.policy:
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
        results = super().run(queries, feed_map)

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
            # choose epsilon-greedy random action
            return self.output.sample()
        else:
            # choose optimal policy action
            return self.predict(self.state)

    def rollout(self):
        # get action
        action = self.action()

        # perform action
        next_state, reward, done, info = self.env.step(action)

        # build and process transition
        transition = Transition(self.state, action, reward, next_state, done, info)
        self.process_transition(transition)

        # record transition
        self.transitions.append(transition)

        # update stats
        self.state = next_state
        self.episode_reward += transition.reward
        return transition

    def process_transition(self, transition):
        pass

    def get_batch(self, mode="train"):
        if self.has_dataset:
            # dataset batch of samples
            return self.dataset.get_batch(mode=mode)
        else:
            # env experience replay batch of samples (TODO - ReplayBuffer)
            batch = TransitionBatch(self.transitions[:self.batch_size], mode=mode)
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
        if self.has_dataset:
            return True
        else:
            return len(self.transitions) >= self.batch_size

    def optimize(self, batch, feed_map):
        # run desired queries
        return self.run(["policy_optimize"], feed_map)

    def optimize_and_report(self, batch, feed_map):
        results = self.optimize(batch, feed_map)

        # get current global step
        # global_step = tf.train.global_step(self.sess, self.global_step)
        global_step = self.config.update_global_step()
        self.current_global_step = global_step

        # print log
        iteration_name = "Epoch" if self.has_dataset else "Episode"
        iteration = self.epoch if self.has_dataset else self.episode
        print_update(f"Optimizing | {iteration_name}: {iteration} | Global Step: {global_step}")
        return results

    def should_evaluate(self):
        if self.has_env and len(self.transitions) < self.batch_size:
            return False
        return not self.training or self.current_global_step % self.evaluate_interval == 0

    def evaluate(self, experiment_yield):
        # Evaluate using the test dataset
        queries = ["evaluate"]

        # prepare dataset partition
        if self.has_dataset:
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
                if self.is_fetch(k, "evaluate") and \
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
            if self.has_dataset:
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
            table["Feeds"] = report_feed_map
            table["Results"] = report_results

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

    def execute(self, render=False, profile=False, random=False):
        try:
            self.max_episode_reward = None
            self.running = True
            self.paused = False
            self.start_time = time.time()

            # build models
            self.build_models(random=random)

            # start session
            self.config.start_session()
            if self.policy:
                self.policy.start_session()

            # check for invalid values in the current graph
            if self.debug_numerics:
                self.add_fetch("check", tf.add_check_numerics_ops(), "policy_optimize")

            # prepare viewer
            self.viewer.prepare(self)

            # print experiment info
            self.print_info()

            # yield after each experiment iteration
            def experiment_yield(flush_summary=False):
                # write summary results
                if flush_summary:
                    self.summary.flush(global_step=self.current_global_step)

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

                # do specific experiment loop
                if self.has_dataset:
                    self.experiment_dataset_loop(experiment_yield)
                else:
                    self.experiment_env_loop(experiment_yield)

            if profile:
                # profile experiment loop
                profile_path = run_profile(experiment_loop, self.config)

                # show profiling results
                open_profile(profile_path)
            else:
                # run training loop without profiling
                experiment_loop()
        finally:
            # cleanup session after evaluation
            if self.policy:
                self.policy.stop_session()
            self.config.close_session()

    def experiment_dataset_loop(self, experiment_yield):
        # dataset learning
        if self.training:
            # train desired epochs
            epoch = 1
            while self.epochs is None or epoch <= self.epochs:
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
                    self.batch, feed_map = self.get_batch()
                    self.optimize_and_report(self.batch, feed_map)

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

    def experiment_env_loop(self, experiment_yield):
        # reinforcement learning
        episode = 1
        reset_evaluate = True

        while self.episodes is None or episode <= self.episodes:
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

                    # stats update
                    episode_num = len(episode_rewards)
                    transitions = len(self.transitions)
                    print_update(f"Simulating | Episode: {episode_num} | Time: {episode_time:.02}"
                                 f" | Reward: {self.episode_reward} | Transitions: {transitions}")

                    # optimize when enough transitions have been gathered
                    processed_transitions = False
                    if self.should_optimize():
                        processed_transitions = True
                        self.batch, feed_map = self.get_batch()
                        self.optimize_and_report(self.batch, feed_map)

                    # evaluate if time to do so
                    if self.should_evaluate():
                        processed_transitions = True
                        self.evaluate(experiment_yield)

                        # episode summary values
                        avg_rewards = np.mean(episode_rewards)
                        self.summary.add_simple_value("episode_reward", avg_rewards,
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

                        if not self.training:
                            self.current_global_step += 1

                    if processed_transitions:
                        if self.on_policy():
                            # clear transitions after processing
                            self.transitions = []

                    break

            if not self.running:
                return

            episode += 1

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
            # self.viewer.close()
            self.running = False
        elif key == pyglet.window.key.SPACE:
            self.warning(f"Experiment {'unpaused' if self.paused else 'paused'} by user")
            self.paused = not self.paused
