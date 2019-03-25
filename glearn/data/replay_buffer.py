import time
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

    def _add_summaries(self):
        query = "replay_buffer"
        t = time.time()
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

        # image summaries
        self._add_image_summaries(summary, query, {
            "transition_age": {
                "values": self.get_ages(),
                "min_color": [0, 1, 0],
                "max_color": [1, 0, 0],
            },
            "transition_step": {
                "values": self.samples["step"],
                "min_color": [0, 1, 1],
                "max_color": [0, 0, 1],
            },
        })

    def _add_image_summaries(self, summary, query, image_definitions):
        # prepare for image summaries
        sample_count = self.sample_count()
        image_width = int(np.ceil(np.sqrt(sample_count)))
        image_height = int(np.ceil(sample_count / image_width))
        image_size = image_width * image_height
        image_pad = image_size - sample_count

        def normalize(values):
            # normalize values in [0, 1]
            # valid_values = [v for v in values if v > 0]
            # if len(valid_values) > 0:
            #     if image_pad > 0:
            #         values = np.concatenate([values, np.zeros(image_pad)])
            #     invalid_idxs = np.array([i for i, v in enumerate(values) if v == 0])

            #     min_value = np.amin(valid_values)
            #     max_value = np.amax(valid_values)

            min_value = np.amin(values)
            max_value = np.amax(values)
            span_value = (max_value - min_value)
            if span_value > 0:
                values = (values - min_value) / span_value
            return np.clip(values, 0, 1)

        def get_colors(values):
            # interpolate colors
            min_color = np.array(definition.get("min_color", [0, 0, 0]))
            max_color = np.array(definition.get("max_color", [1, 1, 1]))
            span_color = max_color - min_color
            return np.matmul(values[:, np.newaxis], span_color[np.newaxis, :]) + min_color

        # loop through building images from definitions and writing summaries
        for name, definition in image_definitions.items():
            values = normalize(definition["values"])
            rgb_values = get_colors(values)

            invalid_color = definition.get("invalid_color", [0, 0, 0])
            if image_pad > 0:
                padding = np.tile(invalid_color, (image_pad, 1))
                rgb_values = np.concatenate([rgb_values, padding])
            image = rgb_values.reshape((1, image_height, image_width, -1))
            summary.write_images(name, image, query=query)

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
        self._add_summaries()

        if self.trainer.on_policy():
            self.clear()
        # TODO - could evict based on age here for off-policy
