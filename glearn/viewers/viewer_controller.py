from collections import abc
import numpy as np
from glearn.utils.reflection import get_class
from glearn.utils.config import Configurable


class ViewerController(Configurable):
    def __init__(self, config, render=True):
        super().__init__(config)

        self.listeners = []
        self._viewer = None

        if render and config.has("viewer"):
            try:
                ViewerClass = get_class(config.get("viewer"))
                self._viewer = ViewerClass(config)
                self.init_viewer()

                if self.env is not None:
                    self.env.unwrapped.viewer = self._viewer
            except Exception as e:
                self.error(f"Failed to load viewer: {e}")

    def get_viewer(self):
        if self._viewer is not None:
            return self._viewer
        if self.env is not None:
            if hasattr(self.env.unwrapped, "viewer"):
                return self.env.unwrapped.viewer
        return None

    def init_viewer(self):
        # register for events from viewer
        if self._viewer is not None and self._viewer.window:
            self._viewer.window.push_handlers(self)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    def __getattr__(self, attr):
        viewer = self.get_viewer()
        if viewer is not None:
            return getattr(viewer, attr)
        return None

    @property
    def rendering(self):
        return self._viewer is not None

    @property
    def width(self):
        if self._viewer is not None:
            return self._viewer.width
        return 0

    @property
    def height(self):
        if self._viewer is not None:
            return self._viewer.height
        return 0

    def get_size(self):
        if self._viewer is not None:
            return (int(self._viewer.width), int(self._viewer.height))
        return (0, 0)

    def prepare(self, trainer):
        if hasattr(self._viewer, "prepare"):
            self._viewer.prepare(trainer)

    def render(self):
        if self._viewer is not None and hasattr(self._viewer, "render"):
            self._viewer.render()

    def view_results(self, query, feed_map, results):
        if hasattr(self._viewer, "view_results"):
            self._viewer.view_results(query, feed_map, results)

    def on_key_press(self, key, modifiers):
        for listener in self.listeners:
            if hasattr(listener, "on_key_press"):
                listener.on_key_press(key, modifiers)

    def add_listener(self, listener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

    def process_image(values, rows=None, cols=None, chans=None):
        # get image dimensions
        values_dims = len(values.shape)
        if values_dims == 1:
            vrows = 1
            vcols = values.shape[0]
            vchans = 1
        elif values_dims == 2:
            vrows, vcols = values.shape
            vchans = 1
        elif values_dims == 3:
            vrows, vcols, vchans = values.shape
        else:
            print(f"Too many dimensions ({values_dims} > 3) on passed image data")
            return values

        # get final rows/cols
        if rows is None:
            rows = vrows
        if cols is None:
            cols = vcols

        # init channel mapping
        if isinstance(chans, int):
            chans = range(chans)
        elif isinstance(chans, abc.Iterable):
            pass
        else:
            chans = range(vchans)
        nchans = len(chans)

        # create processed image
        processed = np.zeros((rows, cols, nchans))

        # calculate value ranges, extract channels and normalize
        flat_values = values.ravel()
        size = len(flat_values)
        value_min = min(flat_values)
        value_max = max(flat_values)
        value_range = max([0.1, value_max - value_min])
        flat_values = [int((v - value_min) / value_range * 255) for v in flat_values]
        done = False
        for y in range(rows):
            if done:
                break
            for x in range(cols):
                if done:
                    break
                for c in range(nchans):
                    idx = y * vcols + x + chans[c]
                    if idx >= size:
                        done = True
                        break
                    value = flat_values[idx]
                    processed[y][x][c] = value
        return processed
