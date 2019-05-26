import tensorflow as tf
from .supervised import SupervisedTrainer


class SequenceTrainer(SupervisedTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.debug_embeddings = config.is_debugging("debug_embeddings")
        self.debug_embedded = config.is_debugging("debug_embedded")

    def build_trainer(self):
        super().build_trainer()

        with self.variable_scope("sequence/"):
            self.build_embedding_images()

    def process_evaluate_results(self, results):
        num_labels = 1
        input_batch = self.dataset.decipher(results["X"][:num_labels])
        target_batch = self.dataset.decipher(results["Y"][:num_labels])
        predict_batch = self.dataset.decipher(results["predict"][:num_labels])

        with self.variable_scope("sequence/"):
            for i in range(num_labels):
                input_seq = " ".join([str(x) for x in input_batch[i]])
                target_seq = " ".join([str(x) for x in target_batch[i]])
                predict_seq = " ".join([str(x) for x in predict_batch[i]])

                # log in summary
                prediction_table = {
                    "input": input_seq,
                    "target": target_seq,
                    "predicted": predict_seq,
                }
                tensor = tf.stack([tf.convert_to_tensor([k, v])
                                   for k, v in prediction_table.items()])
                self.summary.write_text(f"prediction_{i}", tensor)

    def build_embedding_images(self):
        # render embeddings params
        if self.debug_embeddings:
            embedding = self.get_fetch("embedding")
            embedding_image = tf.expand_dims(embedding, axis=0)
            embedding_image = tf.expand_dims(embedding_image, axis=-1)
            self.summary.add_images("embedding", embedding_image)

        # render embedded representation of input
        if self.debug_embedded:
            embedded = self.get_fetch("embedded")
            embedded_image = embedded[:1]
            embedded_image = tf.expand_dims(embedded_image, axis=-1)
            self.summary.add_images("embedded", embedded_image)
