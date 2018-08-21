import numpy as np
import pyglet


class AdvancedViewer(object):
    def __init__(self, display=None, width=None, height=None):
        self.isopen = False
        self.display = display
        self.images = {}

        if width is None:
            width = 100
        if height is None:
            height = 100
        self.window = pyglet.window.Window(width=width, height=height, visible=False,
                                           display=self.display, vsync=False, resizable=True)
        self.window.push_handlers(self)
        self.isopen = True

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()

    def on_resize(self, width, height):
        self.width = width
        self.height = height

    def on_close(self):
        self.isopen = False

    def add_image(self, name, values, x=0, y=0, width=None, height=None):
        # TODO - could cache/reuse image objects and call set_data on them...
        self.images[name] = (values, x, y, width, height)

    def remove_image(self, name):
        self.images.pop(name, None)

    def imshow(self, arr):
        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        height, width, chans = arr.shape
        zoom = 4
        width *= zoom
        height *= zoom
        self.window.set_size(width, height)
        self.add_image("*", arr, x=0, y=0, width=width, height=height)

    def render(self):
        if len(self.images) == 0:
            return

        if not self.window.visible:
            self.window.set_visible(True)
            self.window.activate()

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # draw custom images (such advanced!)
        for image_data in self.images.values():
            # get image data
            values, x, y, width, height = image_data
            rows, cols, chans = values.shape
            if width is None:
                width = cols
            if height is None:
                height = rows

            # convert data to image format (TODO - cache & be more careful with types here)
            values = np.array(values).ravel().astype(int).tolist()
            bytes_conv = bytes(values)

            # infer format
            if chans >= 4:
                fmt = "RGBA"
            elif chans == 3:
                fmt = "RGB"
            elif chans == 1:
                fmt = "I"

            # could cache/reuse image objects and call set_data on them...
            image = pyglet.image.ImageData(cols, rows, fmt, bytes_conv, pitch=cols * -chans)

            # TODO... pixel interpolation
            # from pyglet.gl import *
            # gl.EnableTex2d(...)
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

            # blit image
            image.blit(x, y, width=width, height=height)

        self.window.flip()
