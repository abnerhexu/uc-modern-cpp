### 第二部分：自动微分

#### 数值微分

有时候，我们无需知道一个函数具体的表达式，借助导数的定义，利用计算机可以求解出在某一点的导数值。这种方法称为数值微分。举个例子，对于任何一个$f(x)$，我们当然可以根据定义求出其在$x=x_0$处的导数，即

$$f'(x)|_{x=x_0} = \frac{f(x_0+\varepsilon)-f(x_0 - \varepsilon)}{2\varepsilon }$$

其中$\varepsilon$是一个很小的正数。但是，如果$f(x)$的表达式非常复杂，那么我们可能无法直接求出导数。此时，我们可以借助数值微分来求解导数值。下面我们以$f(x)=x^2$为例，演示如何使用数值微分求解导数值。

```python
import numpy as np

def f(x):
    return x**2

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

x = 5.0
```

当然，你现在需要用C++来完成这件事。

**[TASK 7]** 补全`operators/autodiff.h`中的`central_difference`函数，实现数值微分，求出$f(x_1, x_2, ..., x_n)$在第$arg$个参数处的导数值。


#### 高等数学中的导数

还记得$z = x + y$，对$x$和$y$分别求导的结果是什么吗？显然，根据多元函数的求导法则，有$\frac{\partial z}{\partial x}=1$，以及$\frac{\partial z}{\partial y}=1$。如果我们再考虑梯度，那么$z$的梯度就是$\nabla z = (1, 1)$。那么，对于更复杂的函数，比如$f(x, y) = x^2 + y^2$，其梯度$\nabla f$又是什么呢？

**[TASK 8]** 补全`operators/autodiff.h`中的`Add`类，能够对表达式$z = x + y$求导。

**[TASK 9]** 仿照`Add`类构造`operators/autodiff.h`中的`Mul`类，能够对表达式$z = x \cdot y$求导。