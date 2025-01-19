### Welcome to uct lab

> uct 是Undergraduate Computing Torch的简写。

欢迎你选择uct作为自己的大实验，在这个大实验中，我们将亲自动手使用C++搭建一个机器学习框架，并完成手写体数据集MNIST的识别。

注意：你不需要获得任何对于神经网络的前置知识，考虑到《大学计算（下）》面向的是本科一年级学生，我们设计了非常详细的实验指导书帮助你完成这个实验。

#### 安装构建工具

大型的C++项目显然不止是几个文件，而是成百上千个文件，因此我们需要一个工具来管理这些文件。有很多课程会使用到类似的工具（在《操作系统》课程上，你将会遇见Makefile；在《编译原理》、《并行编译与优化》上，你将会用到CMake），在这里我们选择CMake。

> CMake 是一个开源的跨平台构建系统生成工具，广泛用于管理软件构建过程。它通过生成标准的构建文件（如 Makefile、Visual Studio 项目文件等）来简化跨平台项目的构建流程。

> 对于经验丰富的同学，如果你喜欢使用别的构建工具（例如Bazel）也是可以的~

假如你也正在使用WSL(2)，运行下面的命令可以安装好所需要的工具和库

```bash
sudo apt update
sudo apt install -y build-essential cmake git gcc g++
```

#### 准备Python环境

首先，你需要在Linux下具备Python环境。相信在《大学计算（上）》中，你已经具备这样的技能。我们以使用WSL+VSCode为例介绍环境配置的具体方案。

在VSCode中连接WSL，打开对应目录。

使用`conda`创建一个环境（或使用已有环境），然后执行

```
pip install pybind11
```

而后，通过`pip show pybind11`可以找到`pybind11`的安装路径，将对应的头文件路径添加到`.vscode/c_cpp_properties.json`的`includePath`中。