import matplotlib.pyplot as plt
import numpy as np

points = np.array([[], []])
print(points)

empty = np.empty([2])

a1 = np.arange(6)
a2 = np.arange(7, 13)

for a in a1, a2:
    a = a.reshape((3,2))
    empty = np.append(empty, a, axis=0)

# points = points.reshape((-1, 2))
print(empty)
