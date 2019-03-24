import copy
import numpy as np
from .transition import TransitionBuffer


class ReplayBuffer(TransitionBuffer):
    def __init__(self, config, trainer):
        self.config = config
        self.trainer = trainer

        # configure size
        self.definition = config.get("replay_buffer", {})
        size = self.definition.get("size", self.batch_size)
        assert size >= self.batch_size, \
            f"ReplayBuffer not large enough for batches: {size} > {self.batch_size}"

        super().__init__(size=size)

        self._total_episodes = 0
        self._total_transitions = 0
        self._start_time = None

    @property
    def batch_size(self):
        return self.config.batch_size

    def total_episodes(self):
        return self._total_episodes

    def total_transitions(self):
        return self.total_samples()

    def is_ready(self):
        return self.sample_count() >= self.batch_size

    def add_episode(self, episode):
        # for on-policy, should not overflow buffer
        episode_length = episode.transition_count()
        if self.trainer.on_policy():
            available = self.size - self.sample_count()
            if episode_length > available:
                episode.clip(available)

        # append to and trim buffer
        self.add_buffer(episode)

        # update total episodes stored
        self._total_episodes += 1
        self._total_transitions += episode_length

    def add_summaries(self):
        query = "replay_buffer"
        t = self.config.time()
        summary = self.config.summary

        # rate summaries
        if self._start_time is None:
            self._start_time = t
        else:
            elapsed = t - self._start_time
            episodes_per_second = self._total_episodes / elapsed
            transitions_per_second = self._total_transitions / elapsed
            summary.add_simple_value("episodes_per_second", episodes_per_second, query)
            summary.add_simple_value("transitions_per_second", transitions_per_second, query)

        # count summaries
        summary.add_simple_value("total_episodes", self._total_episodes, query)
        summary.add_simple_value("total_transitions", self._total_transitions, query)
        summary.add_simple_value("current_transitions", self.transition_count(), query)

    def get_batch(self, mode="train"):
        assert not self.empty()

        # collect batch of episodes
        if self.trainer.on_policy():
            idxs = np.array(list(range(self.batch_size)))
        else:
            idxs = np.random.choice(self.sample_count(), self.batch_size, replace=False)

        # slice out these batches
        batch_samples = {key: copy.deepcopy(values[idxs]) for key, values in self.samples.items()}

        # merge episode transitions
        return TransitionBuffer(mode=mode, samples=batch_samples)

    def update(self):
        self.add_summaries()

        if self.trainer.on_policy():
            self.clear()
        # TODO - could evict based on age here for off-policy
