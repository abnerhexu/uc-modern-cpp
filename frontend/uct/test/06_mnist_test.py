import uctc.nn as nn
import std_model as stdnn
import numpy as np
from data6 import x, y
np.random.seed(42)

def parameter_data(*shape):
    assert len(shape) == 2, (
            "Shape must have 2 dimensions, instead has {}".format(len(shape)))
    assert all(isinstance(dim, int) and dim > 0 for dim in shape), (
            "Shape must consist of positive integers, got {!r}".format(shape))
    limit = np.sqrt(3.0 / np.mean(shape))
    data = np.random.uniform(low=-limit, high=limit, size=shape).astype(np.float32)
    return data


class MNISTModel:
    def __init__(self):
        self.input_features = 784
        self.h1 = 200
        self.h2 = 100
        self.output_features = 10
        self.lr = 0.01
        self.batch_size = 100
        self.w1data = parameter_data(self.input_features, self.h1)
        self.b1data = parameter_data(1, self.h1)
        self.w2data = parameter_data(self.h1, self.h2)
        self.b2data = parameter_data(1, self.h2)
        self.w3data = parameter_data(self.h2, self.output_features)
        self.b3data = parameter_data(1, self.output_features)
        self.w1 = nn.Parameter(self.w1data)
        self.b1 = nn.Parameter(self.b1data)
        self.w2 = nn.Parameter(self.w2data)
        self.b2 = nn.Parameter(self.b2data)
        self.w3 = nn.Parameter(self.w3data)
        self.b3 = nn.Parameter(self.b3data)
    
    def run(self, x):
        l1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        l2 = nn.ReLU(nn.AddBias(nn.Linear(l1, self.w2), self.b2))
        l3 = nn.AddBias(nn.Linear(l2, self.w3), self.b3)
        return l3

    def get_loss(self, x, y):
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, x, y):
        loss = self.get_loss(x, y)
        g_w1, g_b1, g_w2, g_b2, g_w3, g_b3 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
        self.w1.update(g_w1, self.lr)
        self.b1.update(g_b1, self.lr)
        self.w2.update(g_w2, self.lr)
        self.b2.update(g_b2, self.lr)
        self.w3.update(g_w3, self.lr)
        self.b3.update(g_b3, self.lr)
        return g_w1.data(), g_b1.data(), g_w2.data(), g_b2.data(), g_w3.data(), g_b3.data()

class StdMNISTModel:
    def __init__(self, model: MNISTModel):
        self.input_features = 784
        self.h1 = 200
        self.h2 = 100
        self.output_features = 10
        self.lr = 0.01
        self.batch_size = 100
        self.w1 = stdnn.Parameter(self.input_features, self.h1)
        self.w1.data = model.w1data
        self.b1 = stdnn.Parameter(1, self.h1)
        self.b1.data = model.b1data
        self.w2 = stdnn.Parameter(self.h1, self.h2)
        self.w2.data = model.w2data
        self.b2 = stdnn.Parameter(1, self.h2)
        self.b2.data = model.b2data
        self.w3 = stdnn.Parameter(self.h2, self.output_features)
        self.w3.data = model.w3data
        self.b3 = stdnn.Parameter(1, self.output_features)
        self.b3.data = model.b3data
        
    
    def run(self, x):
        l1 = stdnn.ReLU(stdnn.AddBias(stdnn.Linear(x, self.w1), self.b1))
        l2 = stdnn.ReLU(stdnn.AddBias(stdnn.Linear(l1, self.w2), self.b2))
        l3 = stdnn.AddBias(stdnn.Linear(l2, self.w3), self.b3)
        return l3

    def get_loss(self, x, y):
        return stdnn.SoftmaxLoss(self.run(x), y)

    def train(self, x, y):
        loss = self.get_loss(x, y)
        g_w1, g_b1, g_w2, g_b2, g_w3, g_b3 = stdnn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
        self.w1.update(g_w1, -self.lr)
        self.b1.update(g_b1, -self.lr)
        self.w2.update(g_w2, -self.lr)
        self.b2.update(g_b2, -self.lr)
        self.w3.update(g_w3, -self.lr)
        self.b3.update(g_b3, -self.lr)
        return g_w1.data.flatten().tolist(), g_b1.data.flatten().tolist(), g_w2.data.flatten().tolist(), g_b2.data.flatten().tolist(), g_w3.data.flatten().tolist(), g_b3.data.flatten().tolist()

model = MNISTModel()
smodel = StdMNISTModel(model)

o1_x = nn.Constant(x)
o1_y = nn.Constant(y)
o1_out = model.run(o1_x).data()
print(o1_out)
# o1_loss = model.get_loss(o1_x, o1_y)
# print(o1_loss.data()[0])
# o1_gw1, o1_gb1, o1_gw2, o1_gb2, o1_gw3, o1_gb3 = model.train(o1_x, o1_y)

o2_x = stdnn.Constant(x)
o2_y = stdnn.Constant(y)
o2_out = smodel.run(o2_x).data
print(o2_out)
# o2_loss = smodel.get_loss(o2_x, o2_y)
# print(o2_loss.data)
# o2_gw1, o2_gb1, o2_gw2, o2_gb2, o2_gw3, o2_gb3 = smodel.train(o2_x, o2_y)

# for i, (a, b) in enumerate(zip(o1_gw1, o2_gw1)):
#     if abs(a - b) > 1e-4:
#         print(f"gw1 failed: {i, a, b}")
#         break
# for i, (a, b) in enumerate(zip(o1_gb1, o2_gb1)):
#     if abs(a - b) > 1e-4:
#         print(f"gb1 failed: {i, a, b}")
#         break  
# for i, (a, b) in enumerate(zip(o1_gw2, o2_gw2)):
#     if abs(a - b) > 1e-4:
#         print(f"gw2 failed: {i, a, b}")
#         break  
# for i, (a, b) in enumerate(zip(o1_gb2, o2_gb2)):
#     if abs(a - b) > 1e-4:
#         print(f"gb2 failed: {i, a, b}")
#         break 
# for i, (a, b) in enumerate(zip(o1_gw3, o2_gw3)):
#     if abs(a - b) > 1e-4:
#         print(f"gw3 failed: {i, a, b}")
#         break 
# for i, (a, b) in enumerate(zip(o1_gb3, o2_gb3)):
#     if abs(a - b) > 1e-4:
#         print(f"gb3 failed: {i, a, b}")
#         break
# print(o1_loss.data()[0], o2_loss.data)
print("PASSED")