import uctc.nn as nn
import std_model as stdnn
import numpy as np
np.random.seed(42)
class LinearTestModel:
    def __init__(self, input_features, hidden_features, output_features):
        self.w1 = nn.Parameter([input_features, hidden_features])
        self.b1 = nn.Parameter([1, hidden_features])
        self.w2 = nn.Parameter([hidden_features, output_features])
        self.b2 = nn.Parameter([1, output_features])
    
    def forward(self, x):
        layer_1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        prediction = nn.AddBias(nn.Linear(layer_1, self.w2), self.b2)
        # print(f"o1: {prediction.data()[:10]}")
        return prediction
    
    def get_loss(self, x, y):
        return nn.SquareLoss(self.forward(x), y)
    
    def backward(self, x, y):
        loss = self.get_loss(x, y)
        g_w1, g_b1, g_w2, g_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
        return g_w1.data(), g_b1.data(), g_w2.data(), g_b2.data()
    
    def update(self, x, y, lr):
        loss = self.get_loss(x, y)
        g_w1, g_b1, g_w2, g_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
        self.w1.update(g_w1, lr)
        self.b1.update(g_b1, lr)
        self.w2.update(g_w2, lr)
        self.b2.update(g_b2, lr)
        # print(g_w1.data())
        # print(g_b1.data())
        # print(g_w2.data())
        # print(g_b2.data())
        # return self.w1.data(), self.b1.data(), self.w2.data(), self.b2.data()
    
    def train(self):
        self.x = np.expand_dims(np.linspace(-2 * np.pi, 2 * np.pi, num=200), axis=1)
        # np.random.RandomState(0).shuffle(self.x)
        self.argsort_x = np.argsort(self.x.flatten())
        self.y = np.sin(self.x)
        for i in range(epoch):
            np.random.RandomState(0).shuffle(self.x)
            index = 0
            while index < self.x.shape[0]:
                x = self.x[index:index + batch_size]
                y = self.y[index:index + batch_size]
                cx = nn.Constant(x)
                cy = nn.Constant(y)
                self.update(cx, cy, 0.01)
                index += batch_size
                # break
            loss = self.get_loss(cx,cy)
            print(loss.data())


class StdLinerTestModel:
    def __init__(self, input_features, hidden_features, output_features, tmodel: LinearTestModel):
        self.w1 = stdnn.Parameter(input_features, hidden_features)
        self.b1 = stdnn.Parameter(1, hidden_features)
        self.w2 = stdnn.Parameter(hidden_features, output_features)
        self.b2 = stdnn.Parameter(1, output_features)
        # self.w1.data = np.array(tmodel.w1.data()).reshape(input_features, hidden_features)
        # self.b1.data = np.array(tmodel.b1.data()).reshape(1, hidden_features)
        # self.w2.data = np.array(tmodel.w2.data()).reshape(hidden_features, output_features)
        # self.b2.data = np.array(tmodel.b2.data()).reshape(1, output_features)
        # print(self.w1.data)
        

    def forward(self, x):
        layer_1 = stdnn.ReLU(stdnn.AddBias(stdnn.Linear(x, self.w1), self.b1))
        prediction = stdnn.AddBias(stdnn.Linear(layer_1, self.w2), self.b2)
        # print(f"o2: {prediction.data.flatten()[:10]}")
        return prediction
    
    def get_loss(self, x, y):
        return stdnn.SquareLoss(self.forward(x), y)
    
    def backward(self, x, y):
        loss = self.get_loss(x, y)
        g_w1, g_b1, g_w2, g_b2 = stdnn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
        return g_w1.data.flatten().tolist(), g_b1.data.flatten().tolist(), g_w2.data.flatten().tolist(), g_b2.data.flatten().tolist()
    
    def update(self, x, y, lr):
        # loss = self.get_loss(x, y)
        # g_w1, g_b1, g_w2, g_b2 = stdnn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
        loss = self.get_loss(x, y)
        g_w1, g_b1, g_w2, g_b2 = stdnn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
        self.w1.update(g_w1, -lr)
        self.b1.update(g_b1, -lr)
        self.w2.update(g_w2, -lr)
        self.b2.update(g_b2, -lr)
        # print(loss.data)
        # return self.w1.data.flatten().tolist(), self.b1.data.flatten().tolist(), self.w2.data.flatten().tolist(), self.b2.data.flatten().tolist()
    
    def train(self):
        self.x = np.expand_dims(np.linspace(-2 * np.pi, 2 * np.pi, num=200), axis=1)
        self.argsort_x = np.argsort(self.x.flatten())
        self.y = np.sin(self.x)
        for i in range(epoch):
            # np.random.RandomState(0).shuffle(self.x)
            index = 0
            while index < self.x.shape[0]:
                x = self.x[index:index + batch_size]
                y = self.y[index:index + batch_size]
                cx = stdnn.Constant(x)
                cy = stdnn.Constant(y)
                self.update(cx, cy, 0.01)
                index += batch_size
                break
            loss = self.get_loss(cx, cy)
            print(loss.data)

input_features = 1
hidden_features = 50
output_features = 1
batch_size = 10
epoch = 1

model = LinearTestModel(input_features, hidden_features, output_features)
smodel = StdLinerTestModel(input_features, hidden_features, output_features, model)

# model.train()


smodel.train()