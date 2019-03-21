import numpy as np
from glearn.utils.config import Configurable
from .transition import TransitionBatch


class ReplayBuffer(Configurable):
    def __init__(self, config, trainer):
        # TODO - allow batch_size to indicate num transitions as well
        super().__init__(config)

        self.trainer = trainer

        # configure size
        self.definition = config.get("replay_buffer", {})
        self.size = self.definition.get("size", self.batch_size)
        assert self.size >= self.batch_size, \
            f"ReplayBuffer not large enough for batches: {self.size} > {self.batch_size}"

        self._total_episodes = 0
        self._total_transitions = 0
        self._current_transitions = 0

        self.clear()

    def clear(self):
        self.episodes = []

    def total_episodes(self):
        return self._total_episodes

    def current_episodes(self):
        return len(self.episodes)

    def total_transitions(self):
        return self._total_transitions

    def current_transitions(self):
        return self._current_transitions

    def add_episode(self, episode):
        # append to and trim buffer
        self.episodes.append(episode)
        self.episodes = self.episodes[-self.size:]  # TODO - could allow other/random evictions

        # update stats
        self._total_episodes += 1
        transition_count = episode.transition_count()
        self._total_transitions += transition_count
        self._current_transitions += transition_count

    def is_ready(self):
        return self.current_episodes() >= self.batch_size

    def get_batch(self, mode="train"):
        # collect batch of episodes
        if self.trainer.on_policy():
            idxs = np.array(list(range(self.batch_size)))
        else:
            idxs = np.random.choice(len(self.episodes), self.batch_size, replace=False)

        # merge episode transitions
        batch = TransitionBatch(mode=mode)
        for idx in idxs:
            batch.add_batch(self.episodes[idx])
        return batch

    def update(self):
        if self.trainer.on_policy():
            self.clear()
