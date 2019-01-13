from glearn.utils.config import Configurable


class ViewerMode(Configurable):
    def __init__(self, config):
        super().__init__(config)

    @property
    def viewer(self):
        return self.config.viewer

    def prepare(self, trainer):
        self.trainer = trainer
        self.policy = trainer.policy

    def view_results(self, families, feed_map, results):
        pass

    def on_key_press(self, key, modifiers):
        pass
