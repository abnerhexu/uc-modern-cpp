import numpy as np
import time
import os
import collections

import matplotlib.pyplot as plt
import uctc.nn as nn 

use_graphics = True

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.input_features = 784
        self.h1 = 200
        self.h2 = 100
        self.output_features = 10
        self.lr = 0.01
        self.batch_size = 100
        self.w1 = nn.Parameter([self.input_features, self.h1])
        self.b1 = nn.Parameter([1, self.h1])
        self.w2 = nn.Parameter([self.h1, self.h2])
        self.b2 = nn.Parameter([1, self.h2])
        self.w3 = nn.Parameter([self.h2, self.output_features])
        self.b3 = nn.Parameter([1, self.output_features])


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        l1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        l2 = nn.ReLU(nn.AddBias(nn.Linear(l1, self.w2), self.b2))
        l3 = nn.AddBias(nn.Linear(l2, self.w3), self.b3)
        return l3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                g_w1, g_b1, g_w2, g_b2, g_w3, g_b3 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(g_w1, self.lr)
                self.b1.update(g_b1, self.lr)
                self.w2.update(g_w2, self.lr)
                self.b2.update(g_b2, self.lr)
                self.w3.update(g_w3, self.lr)
                self.b3.update(g_b3, self.lr)
            accuracy = dataset.get_validation_accuracy()
            print(accuracy)
            if accuracy > 0.95:
                break

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

class DigitClassificationDataset(Dataset):
    def __init__(self, model):
        mnist_path = get_data_path("mnist.npz")

        with np.load(mnist_path) as data:
            train_images = data["train_images"]
            train_labels = data["train_labels"]
            test_images = data["test_images"]
            test_labels = data["test_labels"]
            assert len(train_images) == len(train_labels) == 60000
            assert len(test_images) == len(test_labels) == 10000
            self.dev_images = test_images[0::2]
            self.dev_labels = test_labels[0::2]
            self.test_images = test_images[1::2]
            self.test_labels = test_labels[1::2]

        train_labels_one_hot = np.zeros((len(train_images), 10))
        train_labels_one_hot[range(len(train_images)), train_labels] = 1

        super().__init__(train_images, train_labels_one_hot)

        self.model = model
        self.epoch = 0

        if use_graphics:
            width = 20  # Width of each row expressed as a multiple of image width
            samples = 100  # Number of images to display per label
            fig = plt.figure()
            ax = {}
            images = collections.defaultdict(list)
            texts = collections.defaultdict(list)
            for i in reversed(range(10)):
                ax[i] = plt.subplot2grid((30, 1), (3 * i, 0), 2, 1,
                                         sharex=ax.get(9))
                plt.setp(ax[i].get_xticklabels(), visible=i == 9)
                ax[i].set_yticks([])
                ax[i].text(-0.03, 0.5, i, transform=ax[i].transAxes,
                           va="center")
                ax[i].set_xlim(0, 28 * width)
                ax[i].set_ylim(0, 28)
                for j in range(samples):
                    images[i].append(ax[i].imshow(
                        np.zeros((28, 28)), vmin=0, vmax=1, cmap="Greens",
                        alpha=0.3))
                    texts[i].append(ax[i].text(
                        0, 0, "", ha="center", va="top", fontsize="smaller"))
            ax[9].set_xticks(np.linspace(0, 28 * width, 11))
            ax[9].set_xticklabels(
                ["{:.1f}".format(num) for num in np.linspace(0, 1, 11)])
            ax[9].tick_params(axis="x", pad=16)
            ax[9].set_xlabel("Probability of Correct Label")
            status = ax[0].text(
                0.5, 1.5, "", transform=ax[0].transAxes, ha="center",
                va="bottom")
            plt.show(block=False)

            self.width = width
            self.samples = samples
            self.fig = fig
            self.images = images
            self.texts = texts
            self.status = status
            self.last_update = time.time()

    def iterate_once(self, batch_size):
        self.epoch += 1

        for i, (x, y) in enumerate(super().iterate_once(batch_size)):
            yield x, y

            if use_graphics and time.time() - self.last_update > 1:
                dev_logits_raw = self.model.run(nn.Constant(self.dev_images)).data()
                dev_logits = np.array(dev_logits_raw).reshape(5000, 10)
                dev_predicted = np.argmax(dev_logits, axis=1)
                assert 0
                dev_probs = np.exp(nn.log_softmax(dev_logits_raw))
                dev_accuracy = np.mean(dev_predicted == self.dev_labels)

                self.status.set_text(
                    "epoch: {:d}, batch: {:d}/{:d}, validation accuracy: "
                    "{:.2%}".format(
                        self.epoch, i, len(self.x) // batch_size, dev_accuracy))
                for i in range(10):
                    predicted = dev_predicted[self.dev_labels == i]
                    probs = dev_probs[self.dev_labels == i][:, i]
                    linspace = np.linspace(
                        0, len(probs) - 1, self.samples).astype(int)
                    indices = probs.argsort()[linspace]
                    for j, (prob, image) in enumerate(zip(
                            probs[indices],
                            self.dev_images[self.dev_labels == i][indices])):
                        self.images[i][j].set_data(image.reshape((28, 28)))
                        left = prob * (self.width - 1) * 28
                        if predicted[indices[j]] == i:
                            self.images[i][j].set_cmap("Greens")
                            self.texts[i][j].set_text("")
                        else:
                            self.images[i][j].set_cmap("Reds")
                            self.texts[i][j].set_text(predicted[indices[j]])
                            self.texts[i][j].set_x(left + 14)
                        self.images[i][j].set_extent([left, left + 28, 0, 28])
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(1e-3)
                self.last_update = time.time()

    def get_validation_accuracy(self):
        dev_logits = self.model.run(nn.Constant(self.dev_images)).data
        dev_predicted = np.argmax(dev_logits, axis=1)
        dev_accuracy = np.mean(dev_predicted == self.dev_labels)
        return dev_accuracy

model = DigitClassificationModel()
dataset = DigitClassificationDataset(model)
model.train(dataset)