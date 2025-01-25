import numpy as np
import time
import os

import matplotlib.pyplot as plt
import uctc.nn as nn 

use_graphics = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 10
        self.input_features = 1
        self.output_features = 1
        self.hidden_f1 = 50
        self.lr = 0.01

        self.w1 = nn.Parameter([self.input_features, self.hidden_f1])
        self.b1 = nn.Parameter([1, self.hidden_f1])
        self.w2 = nn.Parameter([self.hidden_f1, self.output_features])
        self.b2 = nn.Parameter([1, self.output_features])

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        linear1 = nn.Linear(x, self.w1)
        bias1 = nn.AddBias(linear1, self.b1)
        act1 = nn.ReLU(bias1)
        linear2 = nn.Linear(act1, self.w2)
        bias2 = nn.AddBias(linear2, self.b2)
        return bias2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.run(x)
        return nn.SquareLoss(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        itera = 0
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                print(loss.data())
                g_w1, g_b1, g_w2, g_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                # print(g_w1.data())
                # print(g_b1.data())
                # print(g_w2.data())
                # print(g_b2.data())
                # assert 0
                self.w1.update(g_w1, self.lr)
                self.b1.update(g_b1, self.lr)
                self.w2.update(g_w2, self.lr)
                self.b2.update(g_b2, self.lr)
            if loss.data()[0] < 0.01:
                break
            itera += 1
            # if itera > 100:
            #     break

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
    
class RegressionDataset(Dataset):
    def __init__(self, model: RegressionModel):
        x = np.expand_dims(np.linspace(-2 * np.pi, 2 * np.pi, num=200), axis=1)
        np.random.RandomState(0).shuffle(x)
        self.argsort_x = np.argsort(x.flatten())
        y = np.sin(x)
        super().__init__(x, y)

        self.model = model
        self.processed = 0

        if use_graphics:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlim(-2 * np.pi, 2 * np.pi)
            ax.set_ylim(-1.4, 1.4)
            real, = ax.plot(x[self.argsort_x], y[self.argsort_x], color="blue")
            learned, = ax.plot([], [], color="red")
            text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
            ax.legend([real, learned], ["real", "learned"])
            plt.show(block=False)

            self.fig = fig
            self.learned = learned
            self.text = text
            self.last_update = time.time()

    def iterate_once(self, batch_size):
        for x, y in super().iterate_once(batch_size):
            yield x, y
            self.processed += batch_size

            if use_graphics and time.time() - self.last_update > 0.1:
                predicted = self.model.run(nn.Constant(self.x)).data()
                loss = self.model.get_loss(
                    nn.Constant(self.x), nn.Constant(self.y)).data()
                predicted = np.array(predicted)
                loss = loss[0]
                self.learned.set_data(self.x[self.argsort_x], predicted[self.argsort_x])
                self.text.set_text(f"processed: {self.processed}\nloss: {loss: .6f}")
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(1e-3)
                self.last_update = time.time()

model = RegressionModel()
dataset = RegressionDataset(model)
model.train(dataset)