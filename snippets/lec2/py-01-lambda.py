a = [42, 23, -17, 11, -8, 15, -38, 23, 18, -1]

# sort by absolute value
a.sort(key=lambda x: abs(x))

print(a)