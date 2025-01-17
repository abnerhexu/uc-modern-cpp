# define a higher order function
# note: this function receives a function as an argument
def apply_twice(func, x):
    # note: this function returns a function
    return func(func(x))

# define a simple function
def square(x):
    return x * x

# use this higher order function
result = apply_twice(square, 2)
print(result)  # output: 16