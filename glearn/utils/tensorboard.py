import os
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from mpi4py import MPI


class TensorBoardWriter(object):
    def __init__(self, dir, prefix=None):
        self.dir = dir
        self.step = 1
        prefix = prefix or 'events'
        path = os.path.join(os.path.abspath(dir), prefix)
        self.writer = tf.summary.FileWriter(path)

    def writekvs(self, kvs, step=None, **_):
        if step is not None:
            self.step = step

        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            if MPI.COMM_WORLD.Get_rank() != 0:
                kwargs['node_name'] = 'proc_%d' % (MPI.COMM_WORLD.Get_rank(),)
            return tf.Summary.Value(**kwargs)
        summary = tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        self.writer.add_summary(summary.SerializeToString(),
                                global_step=self.step)
        self.writer.flush()

    def writehist(self, name2hist, step=None, **_):
        if step is not None:
            self.step = step

        def summary_hist(name, data):
            x, y = data
            x = np.array(x)
            y = np.array(y)
            histo = tf.HistogramProto(
                min=float(x.min()),
                max=float(x.max()),
                num=float(len(x)),
                sum=float(sum(y)),
                sum_squares=float(sum(y**2)),
                bucket_limit=x.tolist(),
                bucket=y.tolist()
            )
            node_name = None
            if MPI.COMM_WORLD.Get_rank() != 0:
                node_name = 'proc_%d' % (MPI.COMM_WORLD.Get_rank(),)
            return tf.Summary.Value(node_name=node_name, tag=name, histo=histo)

        summary = summary_pb2.Summary(value=[
            summary_hist(k, v) for k, v in name2hist.items()])
        self.writer.add_summary(summary.SerializeToString(),
                                global_step=self.step)
        self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None
