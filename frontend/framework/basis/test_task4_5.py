from uctc.framework import basis
import numpy as np
import math
import random

def is_close(x, y):
    return abs(x - y) < 1e-5

arr_a = [random.random() for i in range(128)]
arr_b = [random.random() for i in range(128)]

test_x = basis.addLists(arr_a, arr_b)

test_y = [e1 + e2 for e1, e2 in zip(arr_a, arr_b)]

for i, (x, y) in enumerate(zip(test_x, test_y)):
    if not is_close(x, y):
        print(f"\033[1;31mError: {basis.addLists.__name__} failed test at position {i}, expects {y} but gets {x}\033[0m")
        exit(0)
print(f"\033[1;34mPassed: {basis.addLists.__name__} passed all tests\033[0m")
print(f"\033[1;32m[PASSED] Task 4 finished!\033[0m")