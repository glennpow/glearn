import numpy as np
from glearn.utils.config import Configurable
from glearn.utils.reflection import get_class


class AdvancedViewer(Configurable):
    def __init__(self, config, display=None, width=None, height=None, zoom=1, modes=None):
        super().__init__(config)

        import pyglet

        self.isopen = False
        self.display = display

        self.images = {}
        self.labels = {}

        if width is None:
            width = 100
        if height is None:
            height = 100
        self.width = width
        self.height = height
        self.zoom = zoom

        self.window = pyglet.window.Window(width=width * zoom, height=height * zoom, visible=False,
                                           display=self.display, vsync=False, resizable=True)
        self.window.push_handlers(self)
        self.isopen = True

        self.modes = []
        if modes is not None:
            for mode_config in modes:
                ModeClass = get_class(mode_config)
                mode = ModeClass(config)
                self.modes.append(mode)

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()

    def initialize_gl(self):
        import pyglet.gl as gl

        # Set clear color
        gl.glClearColor(0, 0, 0, 0)

        # Set antialiasing
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_POLYGON_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        # Set alpha blending
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # Set viewport
        # gl.glViewport(0, 0, width, height)

    def prepare(self, trainer):
        for mode in self.modes:
            mode.prepare(trainer)

    def view_results(self, families, feed_map, results):
        for mode in self.modes:
            mode.view_results(families, feed_map, results)

    def on_key_press(self, key, modifiers):
        for mode in self.modes:
            mode.on_key_press(key, modifiers)

    def set_zoom(self, zoom):
        self.zoom = zoom

        self.window.set_size(self.width * self.zoom, self.height * self.zoom)
        pass

    def set_size(self, width, height):
        self.width = width
        self.height = height

        self.window.set_size(self.width * self.zoom, self.height * self.zoom)

    def on_resize(self, width, height):
        self.width = width / self.zoom
        self.height = height / self.zoom

    def on_close(self):
        self.isopen = False

    def set_main_image(self, values):
        height, width, chans = values.shape
        self.set_size(width, height)
        self.add_image("*", values, x=0, y=0, width=width, height=height)

    def imshow(self, arr):
        # backwards compatibility with Atari envs
        self.set_main_image(arr)

    def add_image(self, name, values, x=0, y=0, width=None, height=None):
        import pyglet

        # get image data
        rows, cols, chans = values.shape
        if width is None:
            width = cols
        if height is None:
            height = rows

        # convert data to image format
        values = np.array(values).ravel().astype(int).tolist()
        bytes_conv = bytes(values)

        # infer format
        if chans >= 4:
            fmt = "RGBA"
        elif chans == 3:
            fmt = "RGB"
        elif chans == 1:
            fmt = "I"

        # TODO - could cache/reuse image objects and call set_data on them?
        image = pyglet.image.ImageData(cols, rows, fmt, bytes_conv, pitch=cols * -chans)

        self.images[name] = (image, x, y, width, height)

    def remove_image(self, name):
        self.images.pop(name, None)

    def remove_images(self, prefix):
        self.images = {name: image for name, image in self.images.items()
                       if not name.startswith(prefix)}

    def add_label(self, name, message, x=0, y=0, anchor_x='left', anchor_y='bottom',
                  font_name='Times New Roman', font_size=16, **kwargs):
        import pyglet

        label = pyglet.text.Label(message, x=x, y=y, anchor_x=anchor_x, anchor_y=anchor_y,
                                  font_name=font_name, font_size=font_size, **kwargs)
        self.labels[name] = label

    def remove_label(self, name):
        self.labels.pop(name, None)

    def remove_labels(self, prefix):
        self.labels = {name: label for name, label in self.labels.items()
                       if not name.startswith(prefix)}

    def render(self):
        if len(self.images) + len(self.labels) == 0:
            return

        if not self.window.visible:
            self.window.set_visible(True)
            self.window.activate()

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # TODO - HANDLE ZOOM
        # # Initialize Projection matrix
        # glMatrixMode( GL_PROJECTION )
        # glLoadIdentity()

        # # Initialize Modelview matrix
        # glMatrixMode( GL_MODELVIEW )
        # glLoadIdentity()
        # # Save the default modelview matrix
        # glPushMatrix()

        # # Clear window with ClearColor
        # glClear( GL_COLOR_BUFFER_BIT )

        # # Set orthographic projection matrix
        # glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

        # draw custom images
        for image_data in self.images.values():
            image, x, y, width, height = image_data

            # apply zoom
            x *= self.zoom
            y *= self.zoom
            width *= self.zoom
            height *= self.zoom

            # TODO... pixel interpolation
            # gl.EnableTex2d(image.tex_id) ?
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

            # blit image
            image.blit(x, y, width=width, height=height)

        # draw custom labels
        for label in self.labels.values():
            # apply zoom
            base = (label.x, label.y, label.font_size, label.width, label.height)
            label.x *= self.zoom
            label.y *= self.zoom
            label.width = label.width * self.zoom if label.width is not None else None
            label.height = label.height * self.zoom if label.height is not None else None
            label.font_size *= self.zoom

            label.draw()

            label.x, label.y, label.font_size, label.width, label.height = base

        self.window.flip()
