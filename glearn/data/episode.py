from .transition import TransitionBuffer


class Episode(TransitionBuffer):
    def __init__(self, id, **kwargs):
        super().__init__(**kwargs)

        self.id = id
