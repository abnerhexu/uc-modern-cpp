import sys
a = [1, 2, 3]
print(f"a's reference count: {sys.getrefcount(a)}")
b = a
print(f"a's reference count: {sys.getrefcount(a)}")
print(f"b's reference count: {sys.getrefcount(b)}")
del b
print(f"a's reference count: {sys.getrefcount(a)}")
del a
print(f"a's reference count: {sys.getrefcount(a)}")