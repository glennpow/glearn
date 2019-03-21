from .transition import TransitionBatch


class Episode(TransitionBatch):
    def __init__(self, id, **kwargs):
        super().__init__(**kwargs)

        self.id = id
