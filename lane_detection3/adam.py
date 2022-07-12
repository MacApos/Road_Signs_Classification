import numpy as np
from matplotlib import pyplot as plt


def objective(x, y):
    return x**2 + y**2


def derivative(x, y):
    return x*2, y*2


def adam(bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
    # x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    random_x0 = np.random.uniform(-bounds[0], -bounds[1])
    random_x1 = np.random.uniform(bounds[0], bounds[1])
    x = np.array([random_x0,  random_x1])
    score = objective(x[0], x[1])
    solutions = []

    m = [0.0 for _ in range(x.shape[0])]
    v = [0.0 for _ in range(x.shape[0])]

    for t in range(n_iter):
        g = derivative(x[0], x[1])

        for i in range(x.shape[0]):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2

            mhat = m[i] / (1.0 - beta1 ** (t + 1))
            vhat = v[i] / (1.0 - beta2 ** (t + 1))

            x[i] = x[i] - alpha * mhat / (np.sqrt(vhat) + eps)

        score = objective(x[0], x[1])
        solutions.append(x.copy())

        # print('%d, %s, %.6f' % (t, x, score))

    return x, score, solutions


def solution(bounds, xaxis, yaxis):
    best, score, solutions = adam(bounds, n_iter, alpha, beta1, beta2, eps=1e-8)
    solutions = np.asarray(solutions)

    xaxis = np.linspace(xaxis[0], xaxis[1], 200)
    yaxis = np.linspace(yaxis[0], yaxis[1], 200)

    x, y = np.meshgrid(xaxis, yaxis)

    results = objective(x, y)

    return solutions, x, y, results


n_iter = 60
beta1 = 0.8
beta2 = 0.999
alpha = 0.02
eps = 1e-8

bounds = [0.5, 1.0]
xaxis = [-1, 0.25]
yaxis = [-0.25, 1]
solutions, x, y, results = solution(bounds, xaxis, yaxis)
zaxis = objective(solutions[:, 0], solutions[:, 1])

solutions_2d, x_2d, y_2d, results_2d = solution(bounds, [-1.0, 1.0], [-1.0, 1.0])

ax = plt.axes(projection='3d')
ax.plot(solutions[:, 0], solutions[:, 1], zaxis, color = 'black', alpha = 1, linewidth=2.5)
ax.plot_surface(x, y, results, cmap = 'jet', alpha=0.5)
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('L($w_1$, $w_2$)')

plt.show()

plt.contourf(x_2d, y_2d, results_2d, levels=200, cmap='jet')
plt.plot(solutions_2d[:, 0], solutions_2d[:, 1], '.-', color='w')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')

plt.show()
