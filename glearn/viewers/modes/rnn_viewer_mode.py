import math
from glearn.viewers.modes.viewer_mode import ViewerMode


class RNNViewerMode(ViewerMode):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.visualize_embeddings = False  # HACK - expose these?
        self.visualize_embedded = False  # HACK

    def prepare(self, trainer):
        super().prepare(trainer)

        # cache the desired dims here
        if self.visualize_embeddings:
            self.max_embeddings = 40
            size = self.max_embeddings * self.hidden_size
            stride = self.hidden_size
        elif self.visualize_embedded:
            size = self.hidden_size * self.timesteps
            stride = self.timesteps
        else:
            self.viewer.set_size(512, 512)
            return
        cols = math.ceil(math.sqrt(size) / stride) * stride
        rows = math.ceil(size / cols)
        self.viewer.set_size(cols, rows)

    def view_results(self, graphs, feed_map, results):
        if "debug" in graphs:
            # visualize debug dataset results
            self.update_visualize(feed_map, results)

    def update_visualize(self, feed_map, results):
        cols, rows = self.viewer.get_size()

        if self.visualize_embeddings:
            # render embeddings params
            values = results["embedding"][:self.max_embeddings]
            values = self.viewer.process_image(values, rows=rows, cols=cols)
            self.viewer.set_main_image(values)
        elif self.visualize_embedded:
            # render embedded representation of input
            values = results["embedded"]
            batch = 0
            values = values[batch]
            values = self.viewer.process_image(values, rows=rows, cols=cols)
            self.viewer.set_main_image(values)

        # show labels with targets/predictions
        num_labels = 1
        vocabulary = self.dataset.vocabulary  # HACK
        input = self.output.decode(results["X"])
        input_batch = vocabulary.decode(input[:num_labels])
        target = self.output.decode(results["Y"])
        target_batch = vocabulary.decode(target[:num_labels])
        predict = self.output.decode(results["predict"])
        predict_batch = vocabulary.decode(predict[:num_labels])
        # predict_batch = np.reshape(predict_batch, [num_labels, self.timesteps])
        for i in range(num_labels):
            input_seq = " ".join([str(x) for x in input_batch[i]])
            target_seq = " ".join([str(x) for x in target_batch[i]])
            predict_seq = " ".join([str(x) for x in predict_batch[i]])
            prediction_message = (f"INPUT:  {input_seq}\n"
                                  f"TARGET:  {target_seq}"
                                  f"\nPREDICT: {predict_seq}")
            self.viewer.add_label(f"prediction_{i}", prediction_message, width=cols,
                                  multiline=True, font_name="Courier New", font_size=12)
