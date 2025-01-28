from uctc.framework import basis
import numpy as np
from functools import reduce
import random

def is_close(x, y):
    return abs(x - y) < 1e-3

arr = [random.random() for i in range(128)]

test_x1 = basis.sumList(arr)

test_x2 = basis.prodList(arr)

test_y1 = reduce(lambda x, y: x + y, arr, 0.0)

test_y2 = reduce(lambda x, y: x * y, arr, 1.0)


if not is_close(test_x1, test_y1):
    print(f"\033[1;31mError: {basis.sumList.__name__} failed test... expects {test_y1} but gets {test_x1}\033[0m")
    exit(0)
print(f"\033[1;34mPassed: {basis.sumList.__name__} passed all tests\033[0m")

if not is_close(test_x2, test_y2):
    print(f"\033[1;31mError: {basis.prodList.__name__} failed test... expects {test_y2} but gets {test_x2}\033[0m")
    exit(0)
print(f"\033[1;34mPassed: {basis.prodList.__name__} passed all tests\033[0m")

print(f"\033[1;32m[PASSED] Task 3 finished!\033[0m")