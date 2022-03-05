import matplotlib.pyplot as plt
import numpy as np

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

plt.close('all')


# row and column sharing
f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(x, y)
axarr[0, 1].scatter(x, y)
axarr[1, 0].plot(x, y ** 2)
axarr[1, 1].scatter(x, y ** 2)
# # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
# plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
# plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)


plt.show()