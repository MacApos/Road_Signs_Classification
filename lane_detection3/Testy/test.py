import numpy as np
import cv2
# import os
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import tensorflow as tf

# data_npy = r'F:\krzysztof\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\320x120_data.npy'
# data = np.load(data_npy)

# print(data.shape)
# image = data[1]
#
# # image = cv2.imread(image)
# cv2.imshow('image', image)
# cv2.waitKey(0)
def make_input(message):
    print(message, ' [y/n]')
    x = input()
    x = x.lower()
    if x != 'y' and x != 'n':
        raise Exception('Invalid input')

    return x

# from PIL import Image
# import tensorflow as tf
# img_cv = cv2.imread(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pictures\perspective.jpg')
# img_plt = mpimg.imread(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pictures\perspective.jpg')
# print(type(img_plt))

# img = tf.keras.preprocessing.image.array_to_img(img_data)
# plt.imshow(img)
# plt.show()
#
# array = tf.keras.preprocessing.image.img_to_array(img)
# # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
# plt.imshow(array)
# plt.show()
#
# cv2.waitKey(0)

# from datetime import datetime
#
# epochs = [10, 20, 30, 40]
# learning_rate = 0.001
# batch_size = 16
# input_shape = (60, 160, 3)
#
# # path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
# root_path = os.path.dirname(__file__)
#
# dt = datetime.now().strftime('%d.%m_%H.%M')
# dir_path = os.path.join(path, 'output')
# output_path = os.path.join(dir_path, f'initialized_{dt}')
#
# data_path = os.path.join(path, 'train')
# pickles_path = os.path.join(root_path, 'Pickles')
#
# if not os.path.exists(dir_path):
#     os.mkdir(dir_path)

# input_shape = (2, 2, 1, 3)
# x = np.arange(np.prod(input_shape)).reshape(input_shape)
# print(x)
#
# y = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
# print(y)

# x = np.array([[4,2,3],
#               [1,0,3]])
# index_array_x = np.argmax(x, axis=0) # wiersz o największej wartości w każdej kolumnie
# index_array_y = np.argmax(x, axis=1) # kolumna o największej wartości w każdym wierszu
# print(index_array_x, index_array_y)
#
# num_filters = 3
# start = 16
# block2 = [start * 2 ** i for i in range(num_filters)]
# block3 = [block2[-1] // 2 ** i for i in range(num_filters + 1)]
#
# # for i in range(num_filters + 1):
# #     print(block3)
#
# print(block2, block3)

# arr = np.array([[ 95],
#                 [127],
#                 [159]])
#
#
# image = cv2.imread('../Pictures/original.jpg')
# cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), 50, (0, 255, 0), -2)
# # cv2.imshow('circle', circle)
# # cv2.waitKey(0)
# cv2.imwrite('../Pictures/original_with_circle.jpg', image)

# import plotly.graph_objects as go
# import numpy as np
# np.random.seed(1)
#
# N = 100
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# sz = np.random.rand(N) * 30
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=x,
#     y=y,
#     mode="markers",
#     marker=go.scatter.Marker(
#         size=sz,
#         color=colors,
#         opacity=0.6,
#         colorscale="Viridis"
#     )
# ))
#
# fig.show()
# fig.write_image('../Pictures/fig1.svg')

# a = np.ones((3, 1))
# a = np.resize(a, (a.shape[0], 10))
# print(a)
#
# b = np.ones((3,2))
# b.resize((b.shape[0], 10))
# print(b)

# c = np.ones((3,2))
# d = np.c_[c, np.zeros((c.shape[0], 10))]
# print(d)
# from matplotlib import pyplot as plt
#
# xaxis = np.linspace(0, 5, 200)
# yaxis = np.linspace(0, 5, 200)
# x, y = np.meshgrid(xaxis, yaxis)
#
# z = np.sin(x)*np.cos(y)
# dz = np.cos(x)*(-np.sin(y))
#
# ax = plt.axes(projection='3d')
# ax.plot_surface(x, y, z, cmap = 'jet', alpha=0.5)
# plt.show()
import random
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_regression
# from sklearn.linear_model import LinearRegression
#
# X, y, coefficients = make_regression(
#     n_samples=50,
#     n_features=1,
#     n_informative=1,
#     n_targets=1,
#     noise=5,
#     coef=True,
#     random_state=1
# )
#
# print(X, y, coefficients, X.shape)
#
# plt.scatter(X, y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# x = np.sort(np.array([random.uniform(-50, 50) for i in range(50)]))
#
# #Let y = 0.3x + 0.4x^2. The polyfit needs to discover the parameters 0.4 and 0.3.
#
# #Add some noise to the data. More the noise, more the
# #error from true parameters 0.4 and 0.3. You can try this yourself.
# noise_std_dev = 15.0
# print("Std Dev of noise in Data = ", noise_std_dev)
#
# y = np.array([random.gauss(0.4*i*i + 0.3*i, noise_std_dev) for i in x])
#
# #Let's try to fit a degree 2 polynomial
# z = np.polyfit(x,y,2)
# print("Parameters determined by polyfit = ", z)
#
# #The predicted values based on the linear regression.
# zy = np.array([z[0]*pow(i, 2) + z[1]*pow(i,1) + z[2]*pow(i,0)  for i in x])
#
# #Let's visualize the results
# xp = np.linspace(-60, 60, 100)
#Plot the original points and the curve

image = np.zeros((640, 1280, 3))+255

text = 'OpenCV'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
color = (0, 0, 0)
thickness = 3
(text_width, text_height), _ = cv2.getTextSize(text, font, fontScale, thickness)
org = ((image.shape[1] - text_width)//2, (image.shape[0] + text_height)//2)

image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)


cv2.imshow('image', image)
cv2.waitKey(0)