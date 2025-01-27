import gc

class FooClass:
    def __init__(self):
        self.value = 1
    def __del__(self):
        print(f"FooClass {id(self)} destroyed!")

foo1 = FooClass()
foo2 = foo1
del foo1
print("Called del foo1")
gc.collect()
foo2.value += 1
print("Called foo2.value += 1")