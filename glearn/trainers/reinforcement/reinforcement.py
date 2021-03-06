import time
import numpy as np
from glearn.trainers.trainer import Trainer
from glearn.data.transition import Transition
from glearn.data.episode import Episode, EpisodeBuffer
from glearn.utils.stats import RunningAverage
from glearn.utils.printing import print_update


class ReinforcementTrainer(Trainer):
    def __init__(self, config, epochs=None,
                 max_episode_time=None, max_episode_steps=None, min_episode_reward=None,
                 averaged_episodes=50, **kwargs):
        super().__init__(config, **kwargs)

        self.epochs = epochs
        self.max_episode_time = max_episode_time
        self.max_episode_steps = max_episode_steps
        self.min_episode_reward = min_episode_reward
        self.averaged_episodes = averaged_episodes

        self.state = None
        self.episode = Episode()
        self.buffer = EpisodeBuffer(config, self)

        self._zero_reward_warning = False
        self.env_render_results = None

        self.debug_numerics = self.config.is_debugging("debug_numerics")

        # assert that we have an environment to interact with
        assert self.has_env

    def get_info(self):
        info = super().get_info()
        info.update({
            "Strategy": "on-policy" if self.on_policy() else "off-policy",
            "Epochs": self.epochs,
            "Max Episode Time": self.max_episode_time,
            "Max Episode Steps": self.max_episode_steps,
            "Min Episode Reward": self.min_episode_reward,
        })
        return info

    def learning_type(self):
        return "reinforcement"

    def on_policy(self):
        # override
        return True

    def off_policy(self):
        return not self.on_policy()

    def reset(self, mode="train", episode_count=1):
        if mode == "train":
            # reset env and episode
            self.state = self.env.reset()
            self.episode.reset(episode_count)
        elif mode == "test":
            self._zero_reward_warning = False
        return 1

    def build_Q(self, state=None, count=1, name="Q", definition=None, query=None):
        # prepare inputs
        if state is None:
            state = self.get_feed("X")
        if definition is None:
            definition = self.Q_definition

        # build Q-networks
        Q_networks = []
        for i in range(count):
            Q_name = f"{name}_{i + 1}" if count > 1 else name
            Q_network = self.build_network(Q_name, definition, state, query=query)
            Q_networks.append(Q_network)
        if count > 1:
            return Q_networks
        else:
            return Q_networks[0]

    def build_V(self, state=None, name="V", definition=None, query=None):
        # prepare inputs
        if state is None:
            state = self.get_feed("X")
        if definition is None:
            definition = self.V_definition

        # build V-network
        V_network = self.build_network(name, definition, state, query=query)

        return V_network

    def fetch_V(self, state, name="V"):
        # fetch single value estimate for state
        feed_map = {"X": [state]}
        return self.fetch(name, feed_map, squeeze=True)

    def action(self):
        # choose optimal policy action
        results = self.predict(self.state)

        # extract predicted action and other results
        action = results.pop("predict").reshape(self.config.output.shape)
        predict_info = {k: np.squeeze(v[0]) for k, v in results.items()}
        return action, predict_info

    def sanitize_action(self, action):
        if self.debug_numerics:
            # scrub non-numbers
            if not np.any(np.isfinite(action)):
                self.warning(f"Detected invalid action values: {action}", once=True)
            action = np.nan_to_num(action)
        return action

    def rollout(self):
        # choose action
        action, predict_info = self.action()

        # prepare to perform action
        state = self.state
        step = self.episode_step
        timestamp = time.time()

        # sanitize environment action
        env_action = self.sanitize_action(action)

        # step the environment and get results
        next_state, reward, done, step_info = self.env.step(env_action)

        # combine info
        info = predict_info
        info.update(step_info)

        # build and process transition
        transition = Transition(step, timestamp, state, action, reward, next_state, done, info)
        self.process_transition(transition)

        # record transition
        self.episode.add_transition(transition)

        # update stats
        self.state = next_state
        return transition

    def process_transition(self, transition):
        # override
        pass

    def process_episode(self, episode):
        # ignore zero-reward episodes  (FIXME - why did I have this?  div by zero somewhere?)
        # if np.count_nonzero(episode["reward"]) == 0:
        #     if not self._zero_reward_warning:
        #         self.warning("Ignoring episode(s) with zero rewards!")
        #         self._zero_reward_warning = True
        #     return False
        return True

    def get_batch(self, mode="train"):
        # get env experience replay batch of episodes
        return self.buffer.get_batch(mode=mode)

    def should_optimize(self, done):
        if not super().should_optimize():
            return False

        # on-policy requires full episodes in buffer
        if self.on_policy() and not done:
            return False

        # also make sure buffer is ready
        return self.buffer.is_ready()

    def evaluate_counter(self):
        # evaluate based on epoch count, rather than optimization step
        return self.epoch

    def should_evaluate(self):
        if not super().should_evaluate():
            return False

        # also make sure buffer is ready
        return self.buffer.is_ready()

    def evaluate(self):
        super().evaluate()

        # env summary values
        if hasattr(self.env, "evaluate"):
            self.env.evaluate(self.policy)

    def experiment_loop(self):
        # reinforcement learning loop
        self.epoch = 0
        self.episode_count = 0
        max_episode_reward = None

        # running average stats
        average_rewards = RunningAverage(window=self.averaged_episodes)
        average_times = RunningAverage(window=self.averaged_episodes)
        average_steps = RunningAverage(window=self.averaged_episodes)

        def report_and_yield():
            # episode summary values
            self.summary.add_simple_value("episode_time_avg", average_times.value)
            self.summary.add_simple_value("episode_steps_avg", average_steps.value)
            self.summary.add_simple_value("episode_reward_avg", average_rewards.value)
            self.summary.add_simple_value("episode_reward_max", max_episode_reward)

            return self.experiment_yield(True)

        while self.epochs is None or self.epoch < self.epochs:
            # start current epoch
            self.epoch_start_time = time.time()
            self.epoch += 1
            self.epoch_step = 0
            self.epoch_episodes = 0
            optimized = False

            self.summary.add_simple_value("epoch", self.epoch)

            # iterate through episodes until an optimization has occurred
            while not optimized:
                # start current episode
                self.episode_start_time = time.time()
                self.episode_count += 1
                self.episode_step = 0
                self.reset(episode_count=self.episode_count)

                self.summary.set_simple_value("episode", self.episode_count)

                # step through single episode
                done = False
                while not done:
                    if self.experiment_yield():
                        return

                    # rollout
                    transition = self.rollout()
                    done = transition.done
                    self.episode_step += 1

                    # check episode timeout
                    episode_time = time.time() - self.episode_start_time
                    if self.max_episode_time is not None:
                        if episode_time > self.max_episode_time:
                            done = True
                    if self.max_episode_steps is not None:
                        if self.episode_step >= self.max_episode_steps:
                            done = True

                    # check episode performance
                    if self.min_episode_reward is not None:
                        if self.episode.reward < self.min_episode_reward:
                            done = True

                    if done:
                        # process and store episode
                        if self.process_episode(self.episode):
                            self.buffer.add_episode(self.episode)

                        # track max episode reward
                        if max_episode_reward is None \
                           or self.episode.reward > max_episode_reward:
                            max_episode_reward = self.episode.reward

                        # track episode reward, time and steps
                        average_rewards.add(self.episode.reward)
                        average_times.add(episode_time)
                        average_steps.add(self.episode_step)
                        self.epoch_episodes += 1

                        # stats update
                        print_update(["Simulating", f"Global Step: {self.current_global_step}",
                                      f"Episode: {self.episode_count}",
                                      f"Last Duration: {episode_time:.02f}",
                                      f"Last Reward: {self.episode.reward:.02f}",
                                      f"Buffer: {self.buffer.get_info()}"])

                    # optimize when enough transitions have been gathered
                    if self.should_optimize(done):
                        self.batch = self.get_batch()
                        self.optimize_and_report(self.batch)

                        self.epoch_step += 1
                        optimized = True

                        if report_and_yield():
                            return

                    if not self.running:
                        return

            # evaluate if time to do so
            if self.should_evaluate():
                self.evaluate_and_report()

                if not self.training:
                    if report_and_yield():
                        return

            # prepare buffer for next epoch
            self.buffer.update()

    def render(self, mode="human"):
        render_mode = self.config.get("render_mode", mode)
        if render_mode is not None:
            self.env_render_results = self.env.render(mode=render_mode)

        super().render(mode=mode)
