import time
import numpy as np
from .transition import TransitionBuffer


class Episode(TransitionBuffer):
    def reset(self, id):
        self.id = id

        self.clear()


class EpisodeBuffer(TransitionBuffer):
    def __init__(self, config, trainer):
        self.config = config
        self.trainer = trainer

        # configure buffer
        self.definition = config.get("episode_buffer", {})
        self.batch_episodes = self.definition.get("batch_episodes", False)
        if self.batch_episodes:
            size = None
        else:
            size = self.definition.get("size", self.batch_size)
            assert size >= self.batch_size, \
                f"EpisodeBuffer not large enough for batches: {size} > {self.batch_size}"
        circular = not self.trainer.on_policy()

        # initialize as transition buffer
        super().__init__(size=size, circular=circular)

        self._total_epochs = 0
        self._total_episodes = 0
        self._total_transitions = 0
        self._current_episodes = 0
        self._start_time = None
        self._batches = np.array([], dtype=np.int32)

    def get_info(self):
        if self.batch_episodes:
            return (f"(Episodes: {self._current_episodes} / {self.batch_size}, "
                    f"Transitions: {self.transition_count()})")
        else:
            return (f"(Episodes: {self._current_episodes}, "
                    f"Transitions: {self.transition_count()} / {self.batch_size})")

    @property
    def batch_size(self):
        return self.config.batch_size

    def total_episodes(self):
        return self._total_episodes

    def total_transitions(self):
        return self.total_samples()

    def clear(self):
        super().clear()

        self._current_episodes = 0
        self._batches = np.array([], dtype=np.int32)

    def is_ready(self):
        if self.batch_episodes:
            return self._current_episodes >= self.batch_size
        else:
            return self.sample_count() >= self.batch_size

    def add_episode(self, episode):
        # append to and trim buffer
        stored = self.add_buffer(episode)

        if stored > 0:
            # update total episodes stored
            self._current_episodes += 1
            self._total_episodes += 1
            self._total_transitions += stored

    def _add_summaries(self):
        query = "episode_buffer"
        t = time.time()
        summary = self.config.summary

        # rate summaries
        if self._start_time is None:
            self._start_time = t
        else:
            elapsed = t - self._start_time
            episodes_per_second = self._total_episodes / elapsed
            episodes_per_epoch = self._total_episodes / self._total_epochs
            transitions_per_second = self._total_transitions / elapsed
            summary.add_simple_value("episodes_per_second", episodes_per_second, query)
            summary.add_simple_value("episodes_per_epoch", episodes_per_epoch, query)

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
                "min_color": [0, 0, 1],
                "max_color": [0, 1, 1],
                "marker": {
                    "value": self.samples["done"],
                    "color": [1, 1, 1],
                }
            },
            "transition_batches": {
                "values": self._batches,
                "min_color": [1, 1, 0],
                "max_color": [1, 0, 1],
            },
        })

    def _add_image_summaries(self, summary, query, image_definitions):
        # prepare for image summaries
        sample_count = self.sample_count()
        size = sample_count if self.size is None else self.size
        image_width = int(np.ceil(np.sqrt(size)))
        image_height = int(np.ceil(size / image_width))
        image_size = image_width * image_height

        def normalize(values, definition):
            # normalize values between [0, 1]
            values = values.astype(np.float32)
            min_value = np.amin(values)
            max_value = np.amax(values)
            span_value = (max_value - min_value)
            if span_value > 0:
                values = (values - min_value) / span_value
            return np.clip(values, 0, 1)

        def get_colors(values, definition):
            # interpolate colors
            min_color = np.array(definition.get("min_color", [0, 0, 0]))
            max_color = np.array(definition.get("max_color", [1, 1, 1]))
            span_color = max_color - min_color
            return np.matmul(values[:, np.newaxis], span_color[np.newaxis, :]) + min_color

        # loop through building images from definitions and writing summaries
        for name, definition in image_definitions.items():
            values = definition["values"]
            if values is None or len(values) == 0:
                continue
            values = normalize(values[:sample_count], definition)
            rgb_values = get_colors(values, definition)

            # markers
            if "marker" in definition:
                marker_value = definition["marker"]["value"][:sample_count, np.newaxis]
                marker_colors = np.tile(definition["marker"]["color"], (len(rgb_values), 1))
                rgb_values = np.where(marker_value, marker_colors, rgb_values)

            # pad with invalid color
            invalid_color = definition.get("invalid_color", [0, 0, 0])
            image_pad = image_size - len(values)
            if image_pad > 0:
                padding = np.tile(invalid_color, (image_pad, 1))
                rgb_values = np.concatenate([rgb_values, padding])

            # build final summary image
            image = rgb_values.reshape((1, image_height, image_width, -1))
            summary.add_simple_images(name, image, query=query)

    def get_batch(self, mode="train"):
        assert not self.empty()

        # collect batch of episodes
        if self.batch_episodes:
            idxs = np.array(range(self.sample_count()))
        # elif self.trainer.on_policy():  # FIXME - No?
        #     idxs = np.array(list(range(self.batch_size)))
        else:
            idxs = np.random.choice(self.sample_count(), self.batch_size, replace=False)

        # remember when samples were used in batches
        self._batches.resize((self.sample_count(),))
        self._batches[idxs] = self.trainer.current_global_step

        # slice out these batches
        batch_samples = {key: np.copy(values[idxs]) for key, values in self.samples.items()}

        # merge episode transitions
        return TransitionBuffer(mode=mode, samples=batch_samples)

    def update(self):
        self._total_epochs += 1

        self._add_summaries()

        if self.trainer.on_policy():
            self.clear()

        # TODO - could also evict based on age/etc. here for off-policy
