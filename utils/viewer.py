import numpy as np
import pyglet
from pyglet.gl import *


class AdvancedViewer(object):
    def __init__(self, display=None, width=None, height=None, zoom=1):
        self.isopen = False
        self.display = display

        self.images = {}
        self.labels = {}
        self.label_spacing = 0

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

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()

    def initialize_gl(self):
        # Set clear color
        glClearColor(0, 0, 0, 0)

        # Set antialiasing
        glEnable(gl.GL_LINE_SMOOTH)
        glEnable(gl.GL_POLYGON_SMOOTH)
        glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        # Set alpha blending
        glEnable(gl.GL_BLEND)
        glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # Set viewport
        # glViewport(0, 0, width, height)

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
                  font_name='Times New Roman', font_size=16, **kwargs):
        label = pyglet.text.Label(message, x=x, y=y, anchor_x=anchor_x, anchor_y=anchor_y,
                                  font_name=font_name, font_size=font_size, **kwargs)
        self.labels[name] = label

    def remove_label(self, name):
        self.labels.pop(name, None)

    def set_label_spacing(self, spacing):
        self.label_spacing = spacing

    def imshow(self, arr):
        # backwards compatibility with Atari envs
        self.set_main_image(arr)

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

            x *= self.zoom
            y *= self.zoom
            width *= self.zoom
            height *= self.zoom

            # TODO... pixel interpolation
            # gl.EnableTex2d(image.tex_id) ?
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

            # blit image
            image.blit(x, y, width=width, height=height)

        # draw custom labels (such advanced viewing!)
        offset = 0
        for label in self.labels.values():
            origin_y = label.y
            label.y += offset
            label.draw()
            label.y = origin_y

            offset += label.content_height + self.label_spacing  # HACK - calc this better

        self.window.flip()
