import numpy as np
from glearn.trainers.reinforcement import ReinforcementTrainer


class ReinforceTrainer(ReinforcementTrainer):
    def __init__(self, config, gamma=0.95, **kwargs):
        self.gamma = gamma

        self.create_feed("discount_rewards", ["policy_optimize", "evaluate"], (None, 1))

        super().__init__(config, **kwargs)

    def calculate_discount_rewards(self, rewards):
        # gather discounted rewards
        trajectory_length = len(rewards)
        reward = 0
        discount_rewards = np.zeros(trajectory_length)
        for i in reversed(range(trajectory_length)):
            reward = rewards[i] + self.gamma * reward
            discount_rewards[i] = reward

        # normalize
        std = np.std(discount_rewards)
        mean = np.mean(discount_rewards)
        discount_rewards = (discount_rewards - mean) / std
        return discount_rewards

    # def prepare_feeds(self, queries, feed_map):
    #     if intersects(["policy_optimize", "evaluate"], queries):
    #         # build value feed map with rewards
    #         if self.batch is not None:
    #             # compute discounted rewards
    #             batch = self.batch
    #             discount_rewards = self.calculate_discount_rewards(batch.rewards)
    #             feed_map["discount_rewards"] = discount_rewards
    #         else:
    #             shape = np.shape(feed_map["X"])[:-1] + (1,)
    #             feed_map["discount_rewards"] = np.zeros(shape)

    #     return super().prepare_feeds(queries, feed_map)

    def optimize(self, batch, feed_map):
        # compute discounted rewards
        batch = self.batch
        discount_rewards = self.calculate_discount_rewards(batch.rewards)
        feed_map["discount_rewards"] = discount_rewards

        # run desired queries
        return self.run(["policy_optimize"], feed_map)
