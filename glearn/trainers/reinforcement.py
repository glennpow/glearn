import time
import numpy as np
import tensorflow as tf
from glearn.trainers import Trainer
from glearn.data.transition import Transition
from glearn.data.episode import Episode, EpisodeBuffer
from glearn.utils.printing import print_update


class ReinforcementTrainer(Trainer):
    def __init__(self, config, epochs=None, max_episode_time=None, min_episode_reward=None,
                 epsilon=0, **kwargs):
        super().__init__(config, **kwargs)

        self.epochs = epochs
        self.max_episode_time = max_episode_time
        self.min_episode_reward = min_episode_reward
        self.epsilon = epsilon

        self.state = None
        self.episode = None
        self.buffer = EpisodeBuffer(config, self)

        self._zero_reward_warning = False

    def get_info(self):
        info = super().get_info()
        info.update({
            "Strategy": "on-policy" if self.on_policy() else "off-policy",
            "Epochs": self.epochs,
            "Max Episode Time": self.max_episode_time,
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
            self.episode = Episode(episode_count)
        elif mode == "test":
            self._zero_reward_warning = False
        return 1

    def build_Q(self, state=None, action=None, count=1, name="Q", definition=None):
        # prepare inputs
        if state is None:
            state = self.get_feed("X")
        if action is None:
            action = self.get_feed("Y")
        Q_inputs = tf.concat([state, action], 1)
        if definition is None:
            definition = self.Q_definition

        # build Q-networks
        Q_networks = []
        for i in range(count):
            Q_name = f"{name}_{i + 1}" if count > 1 else name
            Q_network = self.build_network(Q_name, definition, Q_inputs)
            Q_networks.append(Q_network)
        if count > 1:
            return Q_networks
        else:
            return Q_networks[0]

    def optimize_Q(self, td_error, name="Q"):
        query = f"{name}_optimize"
        with tf.name_scope(query):
            Q_network = self.networks[name]

            with tf.name_scope("loss"):
                Q_loss = -td_error * deriv(Q(s, a))  # TODO

            # minimize Q-loss
            Q_network.optimize_loss(Q_loss)

    def build_V(self, state=None, name="V", definition=None):
        # prepare inputs
        if state is None:
            state = self.get_feed("X")
        if definition is None:
            definition = self.V_definition

        # build V-network
        V_network = self.build_network(name, definition, state)
        return V_network

    def fetch_V(self, state, name="V"):
        # fetch single value estimate for state
        feed_map = {"X": [state]}
        return self.fetch(name, feed_map, squeeze=True)

    def optimize_V(self, name="V"):
        query = f"{name}_optimize"
        with tf.name_scope(query):
            with tf.name_scope("loss"):
                # advantage = self.get_advantage(target)
                advantage = self.get_fetch("advantage")

                # value loss minimizes squared advantage
                # TODO - I think the constant could/should be represented as V_coef=0.5
                V_loss = 0.5 * tf.reduce_mean(tf.square(advantage))

                # averages
                # V_value = tf.reduce_mean(value)
                # V_target = tf.reduce_mean(target)
                # V_advantage = tf.reduce_mean(advantage)

            # summaries
            # self.add_metric("value", V_value, query=query)
            # self.add_metric("target", V_target, query=query)
            # self.add_metric("advantage", V_advantage, query=query)
            self.add_metric(f"{name}_loss", V_loss, query=query)

            # minimize V-loss
            self.networks[name].optimize_loss(V_loss, name=query)

        return V_loss

    def build_advantage(self, target, baseline, normalize=True, query=None):
        # calculate advantage estimate (td_error), using td_target
        advantage = target - baseline

        # aborghi normalization
        if normalize:
            mean, std = tf.nn.moments(advantage, axes=[1])
            advantage = (advantage - mean) / (std + 1e-8)

        # fetch
        self.add_fetch("advantage", advantage)

        # summary
        self.add_metric("advantage", tf.reduce_mean(advantage), query=query)

        return advantage

    def action(self):
        # decaying epsilon-greedy
        epsilon = self.epsilon if self.training else 0
        if isinstance(epsilon, list):
            t = min(1, self.current_global_step / epsilon[2])
            epsilon = t * (epsilon[1] - epsilon[0]) + epsilon[0]
            self.summary.set_simple_value("epsilon", epsilon)

        # get action
        if epsilon > 0 and np.random.random() < epsilon:
            # choose epsilon-greedy random action
            return [self.output.sample()]
        else:
            # choose optimal policy action
            return self.predict(self.state)

    def rollout(self):
        # get action
        action = self.action()

        # perform action
        step = self.episode_step
        timestamp = time.time()

        state = self.state
        env_action = action
        if self.output.discrete:
            env_action = env_action[0]  # HACK? - is this the case for all discrete envs?
        next_state, reward, done, info = self.env.step(env_action)

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
        # ignore zero-reward episodes
        if np.count_nonzero(episode["reward"]) == 0:
            if not self._zero_reward_warning:
                self.warning("Ignoring episode(s) with zero rewards!")
                self._zero_reward_warning = True
            return False
        return True

    def get_batch(self, mode="train"):
        # get env experience replay batch of episodes
        return self.buffer.get_batch(mode=mode)

    def should_optimize(self):
        if not super().should_optimize():
            return False
        return self.buffer.is_ready()

    def should_evaluate(self):
        if not super().should_evaluate():
            return False
        return self.buffer.is_ready()

    def extra_evaluate_stats(self):
        return {
            "max reward": self.max_episode_reward,
        }

    def evaluate(self):
        super().evaluate()

        # episode summary values
        self.summary.add_simple_value("average_episode_time", np.mean(self.episode_times))
        self.summary.add_simple_value("average_episode_steps", np.mean(self.episode_steps))
        self.summary.add_simple_value("average_episode_reward", np.mean(self.episode_rewards))
        self.summary.add_simple_value("max_episode_reward", self.max_episode_reward)

        # env summary values
        if hasattr(self.env, "evaluate"):
            self.env.evaluate(self.policy)

    def experiment_loop(self):
        # reinforcement learning loop
        self.epoch = 0
        self.episode_count = 0
        self.max_episode_reward = None

        def reset_evaluate_stats():
            self.episode_rewards = []
            self.episode_times = []
            self.episode_steps = []

        while self.epochs is None or self.epoch < self.epochs:
            # start current epoch
            self.epoch_start_time = time.time()
            self.epoch += 1
            self.epoch_step = 0
            self.epoch_episodes = 0

            self.summary.add_simple_value("epoch", self.epoch)

            reset_evaluate_stats()

            while True:
                # start current episode
                self.episode_start_time = time.time()
                self.episode_count += 1
                self.episode_step = 0
                self.reset(episode_count=self.episode_count)

                self.summary.set_simple_value("episode", self.episode_count)

                while self.running:
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

                    # check episode performance
                    if self.min_episode_reward is not None:
                        if self.episode.reward < self.min_episode_reward:
                            done = True

                    if done:
                        # process and store episode
                        if self.process_episode(self.episode):
                            self.buffer.add_episode(self.episode)

                        # track max episode reward
                        if self.max_episode_reward is None \
                           or self.episode.reward > self.max_episode_reward:
                            self.max_episode_reward = self.episode.reward

                        # track episode reward, time and steps
                        self.episode_rewards.append(self.episode.reward)
                        self.episode_times.append(episode_time)
                        self.episode_steps.append(self.episode_step)
                        self.epoch_episodes += 1

                        # stats update
                        print_update(f"Simulating | Global Step: {self.current_global_step} "
                                     f"| Episode: {self.episode_count} "
                                     f"| Last Time: {episode_time:.02} "
                                     f"| Last Reward: {self.episode.reward} "
                                     f"| Buffer: {self.buffer.get_info()}")

                        break

                # optimize and evaluate when enough transitions have been gathered
                optimizing = self.should_optimize()
                evaluating = self.should_evaluate()
                if optimizing or evaluating:
                    if optimizing:
                        self.batch = self.get_batch()
                        self.optimize_and_report(self.batch)

                        self.epoch_step += 1

                    # evaluate if time to do so
                    if evaluating:
                        self.evaluate_and_report()

                        reset_evaluate_stats()

                    # prepare buffer for next epoch
                    self.buffer.update()

                    if self.experiment_yield(True):
                        return

                if not self.running:
                    return

                if optimizing:
                    break

    def render(self, mode="human"):
        self.env.render(mode=mode)

        super().render(mode=mode)
