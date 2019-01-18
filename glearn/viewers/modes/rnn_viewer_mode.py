import math
from glearn.viewers.modes.viewer_mode import ViewerMode


class RNNViewerMode(ViewerMode):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

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

    def view_results(self, families, feed_map, results):
        if self.debugging and "evaluate" in families:
            cols, rows = self.viewer.get_size()

            if "embeddings" in results:
                # render embeddings params
                values = results["embedding"][:self.max_embeddings]
                values = self.viewer.process_image(values, rows=rows, cols=cols)
                self.viewer.set_main_image(values)
            elif "embedded" in results:
                # render embedded representation of input
                values = results["embedded"]
                batch = 0
                values = values[batch]
                values = self.viewer.process_image(values, rows=rows, cols=cols)
                self.viewer.set_main_image(values)

            # show labels with targets/predictions
            num_labels = 1
            input_batch = self.dataset.decipher(results["X"][:num_labels])
            target_batch = self.dataset.decipher(results["Y"][:num_labels])
            predict_batch = self.dataset.decipher(results["predict"][:num_labels])

            for i in range(num_labels):
                input_seq = " ".join([str(x) for x in input_batch[i]])
                target_seq = " ".join([str(x) for x in target_batch[i]])
                predict_seq = " ".join([str(x) for x in predict_batch[i]])
                prediction_message = (f"INPUT:  {input_seq}\n"
                                      f"TARGET:  {target_seq}"
                                      f"\nPREDICT: {predict_seq}")
                self.viewer.add_label(f"prediction_{i}", prediction_message, width=cols,
                                      multiline=True, font_name="Courier New", font_size=12)
