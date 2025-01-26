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
        l1 = nn.Linear(x, self.w1)
        l2 = nn.AddBias(l1, self.b1)
        l3 = nn.ReLU(l2)
        l4 = nn.Linear(l3, self.w2)
        l5 = nn.AddBias(l4, self.b2)
        return l5
    
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
        print(g_w1.data())
        print(g_b1.data())
        print(g_w2.data())
        print(g_b2.data())
        return self.w1.data(), self.b1.data(), self.w2.data(), self.b2.data()


class StdLinerTestModel:
    def __init__(self, input_features, hidden_features, output_features, tmodel: LinearTestModel):
        self.w1 = stdnn.Parameter(input_features, hidden_features)
        self.b1 = stdnn.Parameter(1, hidden_features)
        self.w2 = stdnn.Parameter(hidden_features, output_features)
        self.b2 = stdnn.Parameter(1, output_features)
        self.w1.data = np.array(tmodel.w1.data()).reshape(input_features, hidden_features)
        self.b1.data = np.array(tmodel.b1.data()).reshape(1, hidden_features)
        self.w2.data = np.array(tmodel.w2.data()).reshape(hidden_features, output_features)
        self.b2.data = np.array(tmodel.b2.data()).reshape(1, output_features)
        

    def forward(self, x):
        l1 = stdnn.Linear(x, self.w1)
        l2 = stdnn.AddBias(l1, self.b1)
        l3 = stdnn.ReLU(l2)
        l4 = stdnn.Linear(l3, self.w2)
        l5 = stdnn.AddBias(l4, self.b2)
        return l5
    
    def get_loss(self, x, y):
        return stdnn.SquareLoss(self.forward(x), y)
    
    def backward(self, x, y):
        loss = self.get_loss(x, y)
        g_w1, g_b1, g_w2, g_b2 = stdnn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
        return g_w1.data.flatten().tolist(), g_b1.data.flatten().tolist(), g_w2.data.flatten().tolist(), g_b2.data.flatten().tolist()
    
    def update(self, x, y, lr):
        loss = self.get_loss(x, y)
        g_w1, g_b1, g_w2, g_b2 = stdnn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
        self.w1.update(g_w1, -lr)
        self.b1.update(g_b1, -lr)
        self.w2.update(g_w2, -lr)
        self.b2.update(g_b2, -lr)
        return self.w1.data.flatten().tolist(), self.b1.data.flatten().tolist(), self.w2.data.flatten().tolist(), self.b2.data.flatten().tolist()

input_features = 1
hidden_features = 50
output_features = 1
batch_size = 10
x = np.array([-5.146528720855713, 4.451905250549316, 0.4736069440841675, -0.09472138434648514, 4.8939385414123535, 5.209676265716553, -5.967447280883789, 2.9363629817962646, -5.525413990020752, 3.315248489379883]).reshape(batch_size, -1)
y = np.array([0.9072322249412537, -0.9662654995918274, 0.45609915256500244, -0.09457980841398239, -0.9835651516914368, -0.8788799047470093, 0.3105180263519287, 0.2037920206785202, 0.6873041391372681, -0.17278438806533813]).reshape(batch_size, -1)

model = LinearTestModel(input_features, hidden_features, output_features)
stdmodel = StdLinerTestModel(input_features, hidden_features, output_features, model)

test_x = nn.Constant(x)
predict_y = model.forward(test_x).data()
test_y = nn.Constant(y)
loss = model.get_loss(test_x, test_y).data()
g_w1, g_b1, g_w2, g_b2 = model.backward(test_x, test_y)
new_w1, new_b1, new_w2, new_b2 = model.update(test_x, test_y, 0)


std_test_x = stdnn.Constant(x)
std_predict_y = stdmodel.forward(std_test_x)
std_test_y = stdnn.Constant(y)
std_loss = stdmodel.get_loss(std_test_x, std_test_y)
std_g_w1, std_g_b1, std_g_w2, std_g_b2 = stdmodel.backward(std_test_x, std_test_y)
std_new_w1, std_new_b1, std_new_w2, std_new_b2 = stdmodel.update(std_test_x, std_test_y, 0)

# print(predict_y)
# print()
# print(std_predict_y.data.flatten().tolist())
# check forward
for x, y in zip(predict_y, std_predict_y.data.flatten().tolist()):
    if (abs(x-y) > 1e-4):
        assert 0, "Forward data mismatch!"

# print(loss, std_loss.data)
# check loss
if abs(loss[0] - std_loss.data) > 1e-4:
    assert 0, "Loss mismatch!"

# check backward
for i, (x, y) in enumerate(zip(g_w1, std_g_w1)):
    if (abs(x-y) > 1e-4):
        assert 0, f"Gradient w1 mismatch at position {i}, g_w1 is {x} while std g_w1 is {y}"
for i, (x, y) in enumerate(zip(g_b1, std_g_b1)):
    if (abs(x-y) > 1e-4):
        assert 0, f"Gradient b1 mismatch at position {i}, g_b1 is {x} while std g_b1 is {y}"
for i, (x, y) in enumerate(zip(g_w2, std_g_w2)):
    if (abs(x-y) > 1e-4):
        assert 0, f"Gradient w2 mismatch at position {i}, g_w2 is {x} while std g_w2 is {y}"
for i, (x, y) in enumerate(zip(g_b2, std_g_b2)):
    if (abs(x-y) > 1e-4):
        assert 0, f"Gradient b2 mismatch at position {i}, g_b2 is {x} while std g_b2 is {y}"

# check update
for i, (x, y) in enumerate(zip(new_b1, std_new_b1)):
    if (abs(x-y) > 1e-4):
        assert 0, f"Updated b1 mismatch at position {i}, new_b1 is {x} while std new_b1 is {y}"
for i, (x, y) in enumerate(zip(new_w1, std_new_w1)):
    if (abs(x-y) > 1e-4):
        assert 0, f"Updated w1 mismatch at position {i}, new_w1 is {x} while std new_w1 is {y}"
# for i, (x, y) in enumerate(zip(new_b2, std_new_b2)):
#     if (abs(x-y) > 1e-4):
#         assert 0, f"Updated b2 mismatch at position {i}, new_b2 is {x} while std new_b2 is {y}"
# for i, (x, y) in enumerate(zip(new_w2, std_new_w2)):
#     if (abs(x-y) > 1e-4):
#         assert 0, f"Updated w2 mismatch at position {i}, new_w2 is {x} while std new_w2 is {y}"
print("Test passed")