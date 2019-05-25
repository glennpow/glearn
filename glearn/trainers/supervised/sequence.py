import tensorflow as tf
from .supervised import SupervisedTrainer


class SequenceTrainer(SupervisedTrainer):
    def process_evaluate_results(self, results):
        num_labels = 1
        input_batch = self.dataset.decipher(results["X"][:num_labels])
        target_batch = self.dataset.decipher(results["Y"][:num_labels])
        predict_batch = self.dataset.decipher(results["predict"][:num_labels])

        with tf.variable_scope("sequence/"):
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
