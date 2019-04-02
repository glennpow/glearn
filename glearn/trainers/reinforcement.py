import time
import numpy as np
from glearn.trainers import Trainer
from glearn.data.transition import Transition
from glearn.data.episode import Episode
from glearn.data.replay_buffer import ReplayBuffer
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
        self.replay_buffer = ReplayBuffer(config, self)

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
        return self.replay_buffer.get_batch(mode=mode)

    def should_optimize(self):
        if not super().should_optimize():
            return False
        return self.replay_buffer.is_ready()

    def should_evaluate(self):
        if not super().should_evaluate():
            return False
        return self.replay_buffer.is_ready()

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
                            self.replay_buffer.add_episode(self.episode)

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
                                     f"| Time: {episode_time:.02} "
                                     f"| Reward: {self.episode.reward} "
                                     f"| Transitions: {self.episode.transition_count()}")

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
                    self.replay_buffer.update()

                    if self.experiment_yield(True):
                        return

                if not self.running:
                    return

                if optimizing:
                    break

    def render(self, mode="human"):
        self.env.render(mode=mode)

        super().render(mode=mode)
