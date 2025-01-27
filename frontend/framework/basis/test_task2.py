from uctc.framework import basis
import numpy as np
import math

binary_arguments = [
    (1.0, 2.0),
    (2.0, 1.0),
    (-1.0, 1.0),
    (2.0, -2.0),
    (1.0, 3.0),
    (3.0, -1.0),
    (3.0, 3.0),
    (-4.0, -5.0),
    (5.0, 4.0),
    (4.0, 4.0),
    (5.0, 5.0)
]

singular_arguments = [
    1.0, -3.2, 4.3, 5.5, -6.7, 4.8, 3.33, 2.22, 1.11
]

def is_close(x, y):
    return abs(x - y) < 1e-5

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))

def iterate_binary_arguments(func, std_func):
    for argument in binary_arguments:
        if not is_close(func(*argument), std_func(*argument)):
            print(f"\033[1;31mError: {func.__name__}({argument}) = {func(*argument)} != {std_func.__name__}({argument}) = {std_func(*argument)}\033[0m")
            exit(0)
    print(f"\033[1;34mPassed: {func.__name__} passed all tests\033[0m")
    return True

def iterate_singular_arguments(func, std_func):
    for argument in singular_arguments:
        if not is_close(func(argument), std_func(argument)):
            print(f"\033[1;31mError: {func.__name__}({argument}) = {func(argument)} != {std_func.__name__}({argument}) = {std_func(argument)}\033[0m")
            exit(0)
    print(f"\033[1;34mPassed: {func.__name__} passed all tests\033[0m")
    return True

# Test task 1
iterate_binary_arguments(basis.is_close, lambda x, y: 1.0*int(is_close(x, y)))
iterate_singular_arguments(basis.sigmoid, lambda x: sigmoid(x))
iterate_singular_arguments(basis.relu, lambda x: x if x > 0.0 else 0.0)
iterate_singular_arguments(basis.inv, lambda x: 1.0/x)
iterate_binary_arguments(basis.inv_back, lambda x, d: -d/(x*x))
iterate_binary_arguments(basis.relu_back, lambda x, d: d * 1.0 if x > 0.0 else 0.0)
print(f"\033[1;32m[PASSED] Task 2 finished!\033[0m")