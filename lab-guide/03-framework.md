### 第三部分：进入人工智能的世界

相信你在前两部分中，已经积累了足够多的C++知识，也回忆起了足够多的高等数学知识。现在，我们要构造一个框架，这个框架可以接受一个矩阵作为输入，并且支持神经网络中的常见的网络层，例如

- 线性层（Linear）
- 激活层（Activation）
- 损失层（Loss）

#### 张量类

我们已经在`cc/tensor/tensor.h`中定义了张量类，这个类可以表示一个多维数组，并且支持常见的数学运算。我们可以在`cc/tensor/tensor.cc`中实现这些运算。当然，我们假定所有的张量都是二维的，这样你就不必考虑各种情况。

**[TASK 10]** 补全`cc/tensor/tensor.cc`中关于`Tensor::transpose()`的函数实现。它能够将一个张量进行转置。

**[TASK 11]** 补全`cc/tensor/tensor.cc`中关于`argmax(const std::shared_ptr<Tensor>& tensor, int axis)`的函数实现，它能够返回一个张量在指定维度上的最大值的索引。

#### 线性层

线性层是神经网络中最为常见的网络层，它接受一个输入张量，并且输出一个张量。输入两个张量`feature: (batch_size x input_features)`和`weight: (input_features x output_features)`，输出张量`output: (batch_size x output_features)`，实际上就是将`feature`矩阵和`weight`矩阵相乘。

用公式表示就是$y = Wx + b$。

**[TASK 12]** 补全`cc/operators/nn.h`中`Linear`类的构造函数和`forward`函数。

- 构造函数：构造函数接受两个参数`a`和`b`，它们都是`std::shared_ptr<Node>`类型的智能指针，分别表示输入特征和权重。构造函数调用基类`FunctionNode`的构造函数，并将`a`和`b`传递给它。在构造函数中，调用`this->forward()`方法，并将结果赋值给`this->data`。

- `forward`函数：参见有关线性层的介绍。


**[TASK 13]** 补全`cc/operators/nn.cc`中`Linear`类的`backward`函数。

- `backward()`函数实现反向传播，计算梯度并返回。它接受`std::shared_ptr<tensor::Tensor> gradient`作为输入，你需要计算`grad_features`和`grad_weights`，它们分别表示对`features`和`weights`的梯度。

> 数学Tips：`grad_features`是通过将`gradient`与`weights`的转置相乘得到的。`grad_weights`是通过将`features`的转置与`gradient`相乘得到的。

完成了这两个任务后，你应该可以在`cc/`下执行

```
cmake -S . -B build
cmake --build build
```

就能够编译你的代码。然后，你应当可以运行`frontend/uct/perception.py`，它将使用你实现的线性层来训练一个感知机。

#### 激活层

激活层是神经网络中常见的网络层，它接受一个输入张量，并且输出一个张量。输入一个张量`x`，输出一个张量`y`，实际上就是将`x`中的每个元素进行某种变换。

用公式表示就是$y = f(x)$。对于`ReLU`函数来说，$y = max(0, x)$。

**[TASK 14]** 补全`cc/operators/nn.h`中`ReLU`类的构造函数和`forward`函数。

- 构造函数：构造函数接受一个参数`a`，它是一个`std::shared_ptr<Node>`类型的智能指针，表示输入特征。构造函数调用基类`FunctionNode`的构造函数，并将`a`传递给它。在构造函数中，调用`this->forward()`方法，并将结果赋值给`this->data`。

- `forward`函数：参见有关激活层的介绍。

**[TASK 15]** 补全`cc/operators/nn.cc`中`ReLU`类的`backward`函数。

- `backward()`函数实现反向传播，计算梯度并返回。它接受`std::shared_ptr<tensor::Tensor> gradient`作为输入，你需要计算`grads`，它表示对`features`的梯度。

> 数学Tips：`grads`是通过将`gradient`与`x`中大于0的元素对应相乘得到的。

#### 偏置

线性层中，我们没有实现偏置项`b`，它是一个向量，它的维度与输出特征的维度相同。偏置项的作用是使得线性层的输出能够更好地拟合数据。

**[TASK 16]** 补全`cc/operators/nn.h`中`AddBias`类的构造函数和`forward`函数。

- 构造函数：构造函数接受两个参数`a`和`b`，它们都是`std::shared_ptr<Node>`类型的智能指针，分别表示输入特征和偏置。构造函数调用基类`FunctionNode`的构造函数，并将`a`和`b`传递给它。在构造函数中，调用`this->forward()`方法，并将结果赋值给 `this->data`。

- `forward`函数：`forward`方法实现前向传播，将偏置添加到输入特征上。`features`和`bias`分别从`this->objects`中获取，`features`的形状为`(batch_size x num_features)`，`bias`的形状为`(1 x num_features)`。在函数中，需要创建一个与`features`形状相同的输出张量`outNode`，使用嵌套循环将`features`的每个元素与`bias`的对应元素相加，结果存储在`outNode`中。最后，返回`outNode`。

**[TASK 17]** 补全`cc/operators/nn.cc`中`AddBias`类的`backward`函数。

- `backward()`函数实现反向传播，计算梯度并返回。它接受`std::shared_ptr<tensor::Tensor> gradient`作为输入，你需要计算`grad_features`和`grad_bias`，它们分别表示对`features`和`bias`的梯度。

> 数学Tips：`grad_features`和`grad_bias`都是`gradient`的拷贝。但是考虑到我们有`batch_size`的存在，因此，在计算`bias`的梯度时，需要将`gradient`的每一列相加，得到`grad_bias`的对应元素。

#### 损失层——均方误差损失函数

我们首先实现均方误差损失函数，它接受两个张量`y_pred`和`y_true`，它们分别表示预测值和真实值，输出一个标量，表示预测值与真实值之间的误差。

用公式表示就是$\displaystyle loss = \frac{1}{2} \sum_{i=1}^{n} (y_{pred} - y_{true})^2$。

**[TASK 18]** 补全`cc/operators/nn.h`中`SquareLoss`类的构造函数和`forward`函数。

- 构造函数：构造函数接受两个参数`a`和`b`，它们都是`std::shared_ptr<Node>`类型的智能指针，分别表示预测值和真实值。构造函数调用基类`FunctionNode`的构造函数，并将`a`和`b`传递给它。在构造函数中，调用`this->forward()`方法，并将结果赋值给`this->data`。

- `forward`函数用于计算损失。

**[TASK 19]** 补全`cc/operators/nn.cc`中`SquareLoss`类的`backward`函数。

- `backward`函数计算损失函数相对于输入`a`和`b`的梯度。`gradient`是损失函数对输出的梯度（是一个形状为(1, 1)的张量，可以直接认为其是一个向量`g`）。`grad_a`和`grad_b`分别存储`a`和`b`的梯度。对于每个元素，梯度计算为`g * (a->data->data[i] - b->data->data[i]) / a->data->size`。最终返回 grad_a 和 grad_b 的向量。

#### 损失层——SoftmaxLoss

接下来，我们实现Softmax损失函数，它接受两个张量`y_pred`和`y_true`，它们分别表示预测值和真实值，输出一个标量，表示预测值与真实值之间的误差。

用公式表示就是$\displaystyle loss = -\sum_{i=1}^{n} y_{true} \log(y_{pred})$。

**[TASK 20]** 补全`cc/operators/nn.h`中`SoftmaxLoss`类的构造函数，`forward`函数和`backward`函数。

完成上述内容后，你可以编译和运行`frontend/uct/regression.py`，使用线性网络来拟合`sin`函数。

### 手写体识别

补全代码中的其他标注有`TODO`的内容，最后编译运行，你就将能够训练一个手写体识别模型。可以运行`frontend/uct/mnist.py`来试一下吧！

> 是不是觉得运行得有点慢？考虑使用多线程来加速矩阵运算。（这已经超出了这门课的要求，对高性能计算/并行计算感兴趣的同学可以勇于尝试！）