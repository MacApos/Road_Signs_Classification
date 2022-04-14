import os
import cv2
import math
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def brighten(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v>255] = 255

    out_hsv = cv2.merge((h, s, v))
    bright = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)

    return bright


def round_num(x, base=5):
    return base * round(x/base)


def threshold(image, T):
    _, img = cv2.threshold(image, T, 250, cv2.THRESH_BINARY)
    return img


def adaptive_threshold(image, block_size, C):
    img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,block_size,C)
    return img


def preprocess(image):
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    max_val = np.mean(np.amax(gray, axis=1)).astype(int)
    # max_val = round_num(max_val, base=10)

    # if max_val>110:
    #     max_val = int(max_val * 0.8)
    # else:
    #     max_val = int(max_val * 0.95)

    return gray, max_val


def arrange(start, steps_num, step):
    range_list = [i for i in range(start, start+steps_num*step, step)]
    return range_list


path = r'C:\Nowy folder\10\Praca\Datasets\Video_data\train_set'
# path = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
list_dir = os.listdir(path)
random_num = random.randint(0, len(list_dir)-1)
# random_num = 499
print(random_num)
image = cv2.imread(os.path.join(path, f'{random_num}.jpg'))
height = image.shape[0]
width = image.shape[1]

src = np.float32([[290,675],
                  [570,515],
                  [670,515],
                  [990,675]])

dst = np.float32([[0,height],
                  [0,0],
                  [width,0],
                  [width,height]])


# _, max_val = preprocess(image)
# values = [i+10 for i in range(70, 150, 10)]
# indices = []
#
# for val in values:
#     while max_val!=val:
#         random_num=random.randint(0, len(list_dir)-1)
#         image = cv2.imread(os.path.join(path, f'{random_num}.jpg'))
#         _, max_val = preprocess(image)
#
#     indices.append(random_num)

# pickle.dump(indices, open('Pickles/indices.p', "wb"))

infile = open('Pickles/indices.p', 'rb')
indices = pickle.load(infile)
infile.close()







normal, max_val = preprocess(image)
thresh = threshold(normal, max_val)

nonzerox = len(thresh.nonzero()[1])
print(nonzerox)

brightened = brighten(image, 25)
bright, _ = preprocess(brightened)

cv2.imshow('bright', bright)
cv2.waitKey(0)

# for idx in indices:
# print(idx)
# image = cv2.imread(os.path.join(path, f'{idx}.jpg'))


new_val = int((1-(0.5*math.log(max_val) - 2.13))*max_val)


new_thresh = threshold(normal, new_val)
print()
bright_thresh = threshold(bright, max_val)

blur = cv2.GaussianBlur(image,(5,5),0)
ret3,th3 = cv2.threshold(normal,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

plt.subplot(221),plt.imshow(thresh),plt.title('Old')
plt.subplot(222),plt.imshow(new_thresh),plt.title('New')
plt.subplot(223),plt.imshow(bright_thresh),plt.title('Bright')
plt.subplot(224),plt.imshow(th3),plt.title('Otsu')
plt.show()
#
# row = 4
# col = 4
# max = 17
#
# fig, axs = plt.subplots(row, col, figsize=(20,20))
# [ax.set_axis_off() for ax in axs.ravel()]
#
# thresh_range = arrange(max_val, row*col, -5)
# block_range = arrange(3, row*col//2, 2)
# C_range = arrange(0, row*col//2, 1)
#
# for idx in range(row*col):
#     if idx<row*col//2:
#         adaptive_thresh = adaptive_threshold(normal, block_range[idx], 2)
#         val = block_range[idx]
#         title = 'block size'
#     else:
#         adaptive_thresh = adaptive_threshold(normal, block_range[-1], C_range[idx-len(C_range)])
#         val = C_range[idx-len(C_range)]
#         title = 'C'
#
#     thresh = threshold(normal, thresh_range[idx])
#     val = thresh_range[idx]
#     title = 'thresh'
#
#     x = idx//col
#     y = (idx+col)%col
#     axs[x][y].imshow(thresh)
#     axs[x][y].set_title(f'{title}={val}')
#
# plt.show()