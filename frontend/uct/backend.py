import collections
import os
import time
import os

import matplotlib.pyplot as plt
import numpy as np

import nn

use_graphics = True

def maybe_sleep_and_close(seconds):
    if use_graphics and plt.get_fignums():
        time.sleep(seconds)
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            plt.close(fig)
            try:
                # This raises a TclError on some Windows machines
                fig.canvas.start_event_loop(1e-3)
            except:
                pass

def get_data_path(filename):
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise Exception("Could not find data file: {}".format(filename))
    return path

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
            "Batch size should be a positive integer, got {!r}".format(
                batch_size))
        assert self.x.shape[0] % batch_size == 0, (
            "Dataset size {:d} is not divisible by batch size {:d}".format(
                self.x.shape[0], batch_size))
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

class PerceptronDataset(Dataset):
    def __init__(self, model):
        points = 500
        x = np.hstack([np.random.randn(points, 2), np.ones((points, 1))])
        y = np.where(x[:, 0] + 2 * x[:, 1] - 1 >= 0, 1.0, -1.0)
        super().__init__(x, np.expand_dims(y, axis=1))

        self.model = model
        self.epoch = 0

        if use_graphics:
            fig, ax = plt.subplots(1, 1)
            limits = np.array([-3.0, 3.0])
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            positive = ax.scatter(*x[y == 1, :-1].T, color="red", marker="+")
            negative = ax.scatter(*x[y == -1, :-1].T, color="blue", marker="_")
            line, = ax.plot([], [], color="black")
            text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
            ax.legend([positive, negative], [1, -1])
            plt.show(block=False)

            self.fig = fig
            self.limits = limits
            self.line = line
            self.text = text
            self.last_update = time.time()

    def iterate_once(self, batch_size):
        self.epoch += 1

        for i, (x, y) in enumerate(super().iterate_once(batch_size)):
            yield x, y

            if use_graphics and time.time() - self.last_update > 0.01:
                w = self.model.get_weights().data.flatten()
                limits = self.limits
                if w[1] != 0:
                    self.line.set_data(limits, (-w[0] * limits - w[2]) / w[1])
                elif w[0] != 0:
                    self.line.set_data(np.full(2, -w[2] / w[0]), limits)
                else:
                    self.line.set_data([], [])
                self.text.set_text(
                    "epoch: {:,}\npoint: {:,}/{:,}\nweights: {}".format(
                        self.epoch, i * batch_size + 1, len(self.x), w))
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(1e-3)
                self.last_update = time.time()
