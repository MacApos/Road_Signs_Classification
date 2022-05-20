import numpy as np

x = np.array([[1, 0, 0],
              [3, 0, 4],
              [5, 0, 0]])
print(np.nonzero(x)[0], np.nonzero(x)[1])