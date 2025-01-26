import numpy as np
np.random.seed(42)
import time
import os

import matplotlib.pyplot as plt
import uctc.nn as nn 
from utils import parameter_data, Dataset

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
        self.w1 = nn.Parameter(parameter_data(self.input_features, self.hidden_f1))
        self.b1 = nn.Parameter(parameter_data(1, self.hidden_f1))
        self.w2 = nn.Parameter(parameter_data(self.hidden_f1, self.output_features))
        self.b2 = nn.Parameter(parameter_data(1, self.output_features))

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # uctc
        linear1 = nn.Linear(x, self.w1)
        bias1 = nn.AddBias(linear1, self.b1)
        act1 = nn.ReLU(bias1)
        linear2 = nn.Linear(act1, self.w2)
        bias2 = nn.AddBias(linear2, self.b2)

        # numpy
        # print(len(x.data()))
        _x = np.array(x.data()).reshape(-1, 1)
        _w1 = np.array(self.w1.data()).reshape(self.input_features, -1)
        _b1 = np.array(self.b1.data()).reshape(1, -1)
        _w2 = np.array(self.w2.data()).reshape(self.hidden_f1, -1)
        _b2 = np.array(self.b2.data()).reshape(1, -1)

        _linear1 = np.dot(_x, _w1) + _b1
        _act1 = np.maximum(0.0, _linear1)
        _linear2 = np.dot(_act1, _w2) + _b2
        
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
                g_w1, g_b1, g_w2, g_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(g_w1, self.lr)
                self.b1.update(g_b1, self.lr)
                self.w2.update(g_w2, self.lr)
                self.b2.update(g_b2, self.lr)
                itera += 1
            if loss.data()[0] < 0.01:
                break
            
    
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
                    x, y).data()
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