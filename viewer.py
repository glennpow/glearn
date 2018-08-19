import numpy as np
import pyglet


class AdvancedViewer(object):
    def __init__(self, display=None, width=None, height=None):
        # self.window = None
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

    def on_key_press(self, key, modifiers):
        if key == pyglet.window.key.ESCAPE:
            self.close()
            exit(0)

    def add_image(self, name, values, x=0, y=0, width=None, height=None):
        # TODO - could cache/reuse image objects and call set_data on them...
        self.images[name] = (values, x, y, width, height)

    def remove_image(self, name):
        self.images.pop(name, None)

    def imshow(self, arr):
        height, width, chans = arr.shape
        self.window.set_size(4 * width, 4 * height)
        # self.width = width
        # self.height = height
        self.window.set_visible(True)
        # if self.window is None:
        #     self.window = pyglet.window.Window(width=4 * width, height=4 * height,
        #                                        display=self.display, vsync=False, resizable=True)
        #     self.width = width
        #     self.height = height
        #     self.isopen = True

        #     @self.window.event
        #     def on_resize(width, height):
        #         self.width = width
        #         self.height = height

        #     @self.window.event
        #     def on_close():
        #         self.isopen = False

        #     @self.window.event
        #     def on_key_press(key, modifiers):
        #         self.on_key_press(key, modifiers)

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        image = pyglet.image.ImageData(width, height, 'RGB', arr.tobytes(), pitch=width * -chans)
        image.blit(0, 0, width=self.window.width, height=self.window.height)

        # draw custom images (such advanced!)
        for image_data in self.images.values():
            # get image data
            values, x, y, width, height = image_data
            rows, cols, chans = values.shape
            if width is None:
                width = cols
            if height is None:
                height = rows

            # convert data to image format (TODO - should be more careful with types here)
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

            # TODO... interpolation
            # from pyglet.gl import *
            # gl.EnableTex2d(...)
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

            # blit image
            image.blit(x, y, width=width, height=height)

        self.window.flip()
