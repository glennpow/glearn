import numpy as np
import pyglet
from glearn.viewers.modes.viewer_mode import ViewerMode
from glearn.networks.layers.conv2d import Conv2dLayer


class CNNViewerMode(ViewerMode):
    def __init__(self, config, visualize_grid=[1, 1], **kwargs):
        super().__init__(config, **kwargs)

        self.visualize_grid = visualize_grid
        self.visualize_layer = None
        self.visualize_feature = None
        self.last_results = None

    def prepare(self, trainer):
        super().prepare(trainer)

        network = self.policy.network
        self.filters = self.config.find("filters")
        conv3d_layers = network.get_layers(Conv2dLayer)
        if self.debugging:
            n = 0
            for layer in conv3d_layers:
                features = layer.references["features"]
                for f in features:
                    network.context.set_fetch(f"conv2d_{n}", f, "predict")
                    n += 1

    def view_results(self, queries, feed_map, results):
        if self.debugging:
            # visualize prediction results
            if self.supervised and "evaluate" in queries:
                self.view_predict(feed_map["X"], feed_map["Y"], results["predict"])

            # visualize CNN features
            if "conv2d_0" in results:
                self.view_features(results)

    def on_key_press(self, key, modifiers):
        super().on_key_press(key, modifiers)

        # feature visualization keys
        if self.debugging:
            if key == pyglet.window.key._0:
                self.clear_visualize()
            elif key == pyglet.window.key.EQUAL:
                if self.visualize_layer is None:
                    self.visualize_layer = 0
                    self.visualize_feature = 0
                else:
                    max_layers = len(self.filters)
                    self.visualize_layer = min(self.visualize_layer + 1, max_layers - 1)
                max_features = self.filters[self.visualize_layer][2]
                self.visualize_feature = min(self.visualize_feature, max_features - 1)
                self.view_features()
            elif key == pyglet.window.key.MINUS:
                if self.visualize_layer is not None:
                    self.visualize_layer -= 1
                    if self.visualize_layer < 0:
                        self.clear_visualize()
                    else:
                        max_features = self.filters[self.visualize_layer][2]
                        self.visualize_feature = min(self.visualize_feature, max_features - 1)
                        self.view_features()
            elif key == pyglet.window.key.BRACKETRIGHT:
                if self.visualize_layer is not None:
                    max_features = self.filters[self.visualize_layer][2]
                    self.visualize_feature = min(self.visualize_feature + 1, max_features - 1)
                    self.view_features()
            elif key == pyglet.window.key.BRACKETLEFT:
                if self.visualize_layer is not None:
                    self.visualize_feature = max(self.visualize_feature - 1, 0)
                    self.view_features()

    def view_predict(self, inputs, outputs, predict):
        # build image grid of inputs
        grid = self.visualize_grid + [1]
        image_size = np.multiply(self.input.shape, grid)
        width = self.input.shape[1]
        height = self.input.shape[0]
        image = np.zeros(image_size)
        for row in range(grid[0]):
            for col in range(grid[1]):
                index = row * grid[1] + col
                if index >= len(inputs):
                    break
                input_image = inputs[index] * 255
                x = col * width
                y = row * height
                image[y:y + height, x:x + width] = input_image

                expected = outputs[index][0]
                predicted = predict[index][0]
                correct = predicted == expected
                predict_s = f"{predicted}"
                color = (0, 255, 0, 255) if correct else (255, 0, 0, 255)
                lx = x + width
                ly = image_size[0] - (y + height)
                self.viewer.add_label(f"action:{index}", predict_s, x=lx, y=ly, font_size=8,
                                      color=color, anchor_x='right', anchor_y='bottom')
        self.viewer.set_main_image(image)

    def view_features(self, results=None):
        if results is None:
            results = self.last_results
        else:
            self.last_results = results
        if self.visualize_layer is not None and results is not None:
            # get layer values to visualize
            values = results[f"conv2d_{self.visualize_layer}"]
            flat_values = values.ravel()
            value_min = min(flat_values)
            value_max = max(flat_values)
            value_range = max([0.1, value_max - value_min])

            # build grid of feature images
            vh = self.viewer.height
            width = self.input.shape[1]
            height = self.input.shape[0]
            for row in range(self.visualize_grid[0]):
                for col in range(self.visualize_grid[1]):
                    # build image for selected feature
                    index = row * self.visualize_grid[1] + col
                    _, f_height, f_width, _ = values.shape
                    image = np.zeros((f_height, f_width, 1))
                    for y, f_row in enumerate(values[index]):
                        for x, f_col in enumerate(f_row):
                            value = f_col[self.visualize_feature]
                            image[y][x][0] = int((value - value_min) / value_range * 255)

                    # add image
                    x = col * width
                    y = vh - (row + 1) * height
                    self.viewer.add_image(f"feature:{index}", image, x=x, y=y, width=width,
                                          height=height)

    def clear_visualize(self):
        self.visualize_layer = None
        self.viewer.remove_images("feature")
