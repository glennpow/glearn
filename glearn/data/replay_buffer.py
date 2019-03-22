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
        self._start_time = None

        self.clear()

    def clear(self):
        self.episodes = []
        self._current_transitions = 0

    def total_episodes(self):
        return self._total_episodes

    def current_episodes(self):
        return len(self.episodes)

    def total_transitions(self):
        return self._total_transitions

    def current_transitions(self):
        return self._current_transitions

    def is_ready(self):
        return self.current_episodes() >= self.batch_size

    def add_episode(self, episode):
        # append to and trim buffer
        self.episodes.append(episode)

        # trim buffer to max size
        self.trim()

        # update basic stats
        self._total_episodes += 1
        transition_count = episode.transition_count()
        self._total_transitions += transition_count
        self._current_transitions += transition_count

    def add_summaries(self):
        # update rate stats
        if self._start_time is None:
            self._start_time = self.time()
            episodes_per_second = 0
            transitions_per_second = 0
        else:
            elapsed = self.time() - self._start_time
            episodes_per_second = self._total_episodes / elapsed
            transitions_per_second = self._total_transitions / elapsed

        # summaries
        query = "replay_buffer"
        self.summary.add_simple_value("total_episodes", self._total_episodes, query)
        self.summary.add_simple_value("current_episodes", self.current_episodes(), query)
        self.summary.add_simple_value("total_transitions", self._total_transitions, query)
        self.summary.add_simple_value("current_transitions", self._current_transitions, query)
        self.summary.add_simple_value("episodes_per_second", episodes_per_second, query)
        self.summary.add_simple_value("transitions_per_second", transitions_per_second, query)

    def trim(self):
        # handle evictions
        evict_count = self.current_episodes() - self.size
        if evict_count > 0:
            idxs = np.array(list(range(evict_count)))
            # TODO - could allow other (random-based) strategy
            # idxs = np.random.choice(len(self.episodes), evict_count, replace=False)

            for idx in idxs:
                self._current_transitions -= self.episodes[idx].transition_count()
            self.episodes = self.episodes[-self.size:]  # only works for age-based strategy

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
        self.add_summaries()

        if self.trainer.on_policy():
            self.clear()
        # TODO - could evict based on age here
