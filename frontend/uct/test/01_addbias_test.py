import uctc.nn as nn
import std_model as stdnn
import numpy as np

class LinearTestModel:
    def __init__(self, output_features):
        self.b1 = nn.Parameter([1, output_features])
    
    def forward(self, x):
        l2 = nn.AddBias(x, self.b1)
        return l2
    
    def get_loss(self, x, y):
        return nn.SquareLoss(self.forward(x), y)
    
    def backward(self, x, y):
        loss = self.get_loss(x, y)
        g_b1 = nn.gradients(loss, [self.b1])[0]
        return g_b1.data()

class StdLinerTestModel:
    def __init__(self, output_features, tmodel: LinearTestModel):
        self.b1 = stdnn.Parameter(1, output_features)
        self.b1.data = np.array(tmodel.b1.data()).reshape(1, output_features)

    def forward(self, x):
        l2 = stdnn.AddBias(x, self.b1)
        return l2
    
    def get_loss(self, x, y):
        return stdnn.SquareLoss(self.forward(x), y)
    
    def backward(self, x, y):
        loss = self.get_loss(x, y)
        g_b1 = stdnn.gradients(loss, [self.b1])[0]
        return g_b1.data.flatten().tolist()

output_features = 32
batch_size = 4
x = np.random.randn(batch_size, output_features).astype(np.float32)
y = np.random.randn(batch_size, output_features).astype(np.float32)

model = LinearTestModel(output_features)
test_x = nn.Constant(x)
predict_y = model.forward(test_x).data()
test_y = nn.Constant(y)
loss = model.get_loss(test_x, test_y).data()
g_b1 = model.backward(test_x, test_y)

stdmodel = StdLinerTestModel(output_features, model)
std_test_x = stdnn.Constant(x)
std_predict_y = stdmodel.forward(std_test_x)
std_test_y = stdnn.Constant(y)
std_loss = stdmodel.get_loss(std_test_x, std_test_y)
std_g_b1 = stdmodel.backward(std_test_x, std_test_y)

# check forward
for x, y in zip(predict_y, std_predict_y.data.tolist()[0]):
    if (abs(x-y) > 1e-4):
        assert 0, "Forward data mismatch!"

# check loss
if abs(loss[0] - std_loss.data) > 1e-4:
    assert 0, "Loss mismatch!"

# check backward
for i, (x, y) in enumerate(zip(g_b1, std_g_b1)):
    if (abs(x-y) > 1e-4):
        assert 0, f"Gradient b1 mismatch at position {i}, g_b1 is {x} while std g_b1 is {y}"


print("Test passed")