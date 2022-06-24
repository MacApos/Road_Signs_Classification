import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

x1 = np.ones((16,2)) * 20
r_offset= np.copy(x1)[:, 0] + 10

# x2 = np.concatenate(x1[:, 0], x1[:, 1] + 10, axis=1)
print(r_offset)