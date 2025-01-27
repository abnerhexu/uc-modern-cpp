from uctc.framework import basis
import numpy as np
import math

binary_arguments = [
    (1, 2),
    (-2, 1),
    (1, 1),
    (2, -2),
    (1, 3),
    (3, 1),
    (-3, 3),
    (4, 5),
    (5, 4),
    (4, 4),
    (5, 5)
]

singular_arguments = [
    1, 2, 4, -32, 42, 28, 0, 100, -1000, 10000, -100000
]

def iterate_binary_arguments(func, std_func):
    for argument in binary_arguments:
        if func(*argument) != std_func(*argument):
            print(f"\033[1;31mError: {func.__name__}({argument}) = {func(*argument)} != {std_func.__name__}({argument}) = {std_func(*argument)}\033[0m")
            exit(0)
    print(f"\033[1;34mPassed: {func.__name__} passed all tests\033[0m")
    return True

def iterate_singular_arguments(func, std_func):
    for argument in singular_arguments:
        if func(argument) != std_func(argument):
            print(f"\033[1;31mError: {func.__name__}({argument}) = {func(argument)} != {std_func.__name__}({argument}) = {std_func(argument)}\033[0m")
            exit(0)
    print(f"\033[1;34mPassed: {func.__name__} passed all tests\033[0m")
    return True

# Test task 1
iterate_binary_arguments(basis.mul, lambda x, y: x * y)
iterate_singular_arguments(basis.id, lambda x: x)
iterate_binary_arguments(basis.add, lambda x, y: x + y)
iterate_singular_arguments(basis.neg, lambda x: -x)
iterate_binary_arguments(basis.lt, lambda x, y: int(x < y))
iterate_binary_arguments(basis.eq, lambda x, y: int(x == y))
iterate_binary_arguments(basis.max, lambda x, y: max(x, y))
print(f"\033[1;32m[PASSED] Task 1 finished!\033[0m")