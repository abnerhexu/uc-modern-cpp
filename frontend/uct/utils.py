import numpy as np
import uctc.nn as nn

def parameter_data(*shape):
    assert len(shape) == 2, (
            "Shape must have 2 dimensions, instead has {}".format(len(shape)))
    assert all(isinstance(dim, int) and dim > 0 for dim in shape), (
            "Shape must consist of positive integers, got {!r}".format(shape))
    limit = np.sqrt(3.0 / np.mean(shape))
    data = np.random.uniform(low=-limit, high=limit, size=shape).astype(np.float32)
    return data

class Dataset(object):
    def __init__(self, x, y):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert np.issubdtype(x.dtype, np.floating)
        assert np.issubdtype(y.dtype, np.floating)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def iterate_once(self, batch_size):
        assert isinstance(batch_size, int) and batch_size > 0, (
            f"Batch size should be a positive integer, got {batch_size}")
        assert self.x.shape[0] % batch_size == 0, (
            f"Dataset size {self.x.shape[0]} is not divisible by batch size {batch_size}")
        index = 0
        while index < self.x.shape[0]:
            x = self.x[index:index + batch_size]
            y = self.y[index:index + batch_size]
            yield nn.Constant(x), nn.Constant(y)
            index += batch_size

    def iterate_forever(self, batch_size):
        while True:
            yield from self.iterate_once(batch_size)

    def get_validation_accuracy(self):
        raise NotImplementedError(
            "No validation data is available for this dataset. "
            "In this assignment, only the Digit Classification and Language "
            "Identification datasets have validation data.")