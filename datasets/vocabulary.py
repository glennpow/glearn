from collections import abc
import numpy as np


class Vocabulary(object):
    def __init__(self, words):
        self.words = words
        self.size = len(self.words)
        self.word_to_ids = dict(zip(words, range(len(words))))

    def encode(self, word):
        if isinstance(word, str):
            return self.word_to_ids.get(word, None)
        elif isinstance(word, abc.Iterable):
            # TODO omit Nones...
            return [self.encode(w) for w in word]
        else:
            print(f"Unknown vocabulary word type: {word} ({type(word)})")
            return None

    def decode(self, id):
        if isinstance(id, int) or np.isscalar(id):
            if id < self.size:
                return self.words[id]
        elif isinstance(id, abc.Iterable):
            # TODO omit Nones...
            return [self.decode(i) for i in id]
        else:
            print(f"Unknown vocabulary word ID type: {id} ({type(id)})")
            return None
