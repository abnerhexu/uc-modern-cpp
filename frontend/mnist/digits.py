from uct.nn import Linear, relu, cross_entropy
from .dataloader import MNISTDataLoader
from uct.optim import optim

class DigitClassificationModel:
    def __init__(self):
        input_size = 28 * 28
        output_size = 10
        hidden_size = 256
        self.batch_size = 30
        self.input_layer = Linear(input_size, hidden_size)
        self.hidden_layer1 = Linear(hidden_size, hidden_size)
        self.hidden_layer2 = Linear(hidden_size, hidden_size)
        self.output_layer = Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = relu(self.input_layer(x))
        x = relu(self.hidden_layer1(x))
        x = relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x
    
    def get_loss(self, x, y):
        predict_y = self.forward(x)
        loss = cross_entropy(predict_y, y, num_class=10)
        return loss
    
    def train(self, dataset):
        dataloader = MNISTDataLoader(dataset, batch_size=self.batch_size)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        while dataset.get_validation_accuracy() <= 0.95:
            for data in dataloader:
                optimizer.zero_grad()
                x, y = data
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()