import time
import numpy as np
from glearn.trainers import Trainer
from glearn.data.transition import Transition, TransitionBatch
from glearn.utils.printing import print_update


class ReinforcementTrainer(Trainer):
    def __init__(self, config, episodes=None, max_episode_time=None, min_episode_reward=None,
                 epsilon=0, **kwargs):
        super().__init__(config, **kwargs)

        self.episodes = episodes
        self.max_episode_time = max_episode_time
        self.min_episode_reward = min_episode_reward

        self.epsilon = epsilon

        self.state = None
        self.transitions = []
        self.episode_reward = 0

    def get_info(self):
        info = super().get_info()
        info.update({
            "Strategy": "on-policy" if self.on_policy() else "off-policy",
            "Episodes": self.episodes,
            "Max Episode Time": self.max_episode_time,
            "Min Episode Reward": self.min_episode_reward,
        })
        return info

    def learning_type(self):
        return "reinforcement"

    def on_policy(self):
        # override
        return False

    def off_policy(self):
        return not self.on_policy()

    def reset(self):
        # reset env and episode
        self.state = self.env.reset()
        self.episode_reward = 0

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
        env_action = action
        if self.output.discrete:
            env_action = env_action[0]  # HACK? - is this the case for all envs?
        next_state, reward, done, info = self.env.step(env_action)

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

    def get_iteration_name(self):
        return "Episode"

    def get_batch(self, mode="train"):
        # env experience replay batch of samples (TODO - ReplayBuffer)
        batch = TransitionBatch(self.transitions[:self.batch_size], mode=mode)
        feed_map = {
            "X": batch.inputs,
            "Y": batch.outputs,
        }
        return batch, feed_map

    def should_optimize(self):
        if not super().should_optimize():
            return False
        return len(self.transitions) >= self.batch_size

    def should_evaluate(self):
        if not super().should_evaluate():
            return False
        return len(self.transitions) >= self.batch_size

    def extra_evaluate_stats(self):
        return {
            "reward": self.episode_reward,
            "max reward": self.max_episode_reward,
        }

    def experiment_loop(self):
        # reinforcement learning
        episode = 1
        reset_evaluate = True
        self.max_episode_reward = None

        while self.episodes is None or episode <= self.episodes:
            # start current episode
            self.iteration_start_time = time.time()
            self.iteration = episode
            self.iteration_step = 0
            self.reset()

            # episode count summary
            self.summary.add_simple_value("episode", episode, "experiment")

            if reset_evaluate:
                episode_rewards = []
                episode_times = []
                episode_steps = []

            while self.running:
                if self.experiment_yield(True):
                    return

                # rollout
                transition = self.rollout()
                done = transition.done
                self.iteration_step += 1

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
                    episode_steps.append(self.iteration_step)

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
                        self.evaluate()

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
