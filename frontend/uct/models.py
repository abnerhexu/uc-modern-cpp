import uctc.nn as nn 
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

        self.w1 = nn.Parameter(self.input_features, self.hidden_f1)
        self.w2 = nn.Parameter(self.hidden_f1, self.output_features)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.ReLU(nn.Linear(x, self.w1))
        layer2 = nn.Linear(layer1, self.w2)
        return layer2

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
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

test = RegressionModel()
print(test.get_weights())