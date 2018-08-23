import numpy as np
import pyglet


class AdvancedViewer(object):
    def __init__(self, display=None, width=None, height=None, zoom=4):
        self.isopen = False
        self.display = display
        self.zoom = zoom
        self.images = {}
        self.labels = {}

        if width is None:
            width = 100
        if height is None:
            height = 100
        self.width = width
        self.height = height
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

    def set_zoom(self, zoom):
        self.zoom = zoom

        # TODO? - should set zoom, and adjust window size immediately
        #   need to store (width, height, zoom) tuple, and resize using the products
        # self.window.set_size(self.width * zoom, self.height * zoom)
        pass

    def on_resize(self, width, height):
        self.width = width
        self.height = height

    def on_close(self):
        self.isopen = False

    def set_main_image(self, values):
        height, width, chans = values.shape
        width *= self.zoom  # TODO - better handling of zoom
        height *= self.zoom
        self.window.set_size(width, height)
        self.add_image("*", values, x=0, y=0, width=width, height=height)

    def add_image(self, name, values, x=0, y=0, width=None, height=None):
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

    def add_label(self, name, message, x=0, y=0, anchor_x='left', anchor_y='bottom',
                  font_name='Times New Roman', font_size=16):
        label = pyglet.text.Label(message, x=x, y=y, anchor_x=anchor_x, anchor_y=anchor_y,
                                  font_name=font_name, font_size=font_size)
        self.labels[name] = label

    def remove_label(self, name):
        self.labels.pop(name, None)

    def imshow(self, arr):
        # backwards compatibility with Atari envs
        self.set_main_image(arr)

    def render(self):
        if len(self.images) == 0:
            return

        if not self.window.visible:
            self.window.set_visible(True)
            self.window.activate()

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # draw custom images
        for image_data in self.images.values():
            image, x, y, width, height = image_data

            # TODO... pixel interpolation
            # from pyglet.gl import *
            # gl.EnableTex2d(image.tex_id) ?
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

            # blit image
            image.blit(x, y, width=width, height=height)

        # draw custom labels (such advanced viewing!)
        for label  in self.labels.values():
            label.draw()

        self.window.flip()
