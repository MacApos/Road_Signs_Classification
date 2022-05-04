import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def warp_perspective(image, from_, to):
    M = cv2.getPerspectiveTransform(from_, to)
    warp = cv2.warpPerspective(image, M, (1280, 720), flags=cv2.INTER_LINEAR)
    return warp, M


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def display_channel(image, channel):
    zeros = np.zeros_like(image)
    zeros[:, :, channel] = image[:, :, channel]
    return cv2.cvtColor(zeros, cv2.COLOR_BGR2GRAY)


def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(hls[:,:,1])
    binary_output[(hls[:,:,1] > thresh[0]) & (hls[:,:,1] <= thresh[1])] = 255
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output

def white_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(gray)
    binary_output[(gray > thresh[0]) & (gray <= thresh[1])] = 255
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    return binary_output


path = r'C:\Nowy folder\10\Praca\Datasets\Video_data\video4'
list_dir = os.listdir(path)
random_img = random.sample(list_dir, 1)[0]
# random = 1320
print(random)

# for img in list_dir:
#     print(img)
img = cv2.imread(os.path.join(path, random_img))
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

height = img.shape[0]
width = img.shape[1]

template = [[300, 650], [550, 500]]

src = np.float32([template[0],
                  template[1],
                  [width - template[1][0], template[1][1]],
                  [width - template[0][0], template[0][1]]])

dst = np.float32([[0,height],
                  [0,0],
                  [width,0],
                  [width,height]])

warp, _ = warp_perspective(img, src, dst)
gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

max_val = max(np.amax(gray, axis=1)).astype(int)

mean_max_val = np.mean(np.amax(gray, axis=1))
mean_max_val = int(max_val * 0.85)

mask_white = cv2.inRange(gray, max_val*0.65, int(max_val))
mask_white_image = cv2.bitwise_and(gray, mask_white)

gray_binary = white_select(warp, (max_val*0.65, max_val))

cv2.imshow('gray_binary', gray_binary)
cv2.waitKey(0)



max_light = max(np.amax(img_hls[:, :, 2], axis=1)).astype(int)
print(img_hls[:, :, 1])
mask_light = cv2.inRange(gray, max_light*0.65, int(max_light))
mask_sat_image = cv2.bitwise_and(gray, mask_light)
hls_binary = hls_select(img_hls, thresh=(max_light*0.95, int(max_light)))

_, thresh = cv2.threshold(warp, mean_max_val, 250, cv2.THRESH_BINARY)

# col = 3
#
# fig, axs = plt.subplots(1, col, figsize=(20,20), squeeze=False)
# [ax.set_axis_off() for ax in axs.ravel()]

# for i in range(3):
#     # x = i//col
#     # y = (i+col)%col
#
#     axs[0][0].imshow(mask_white_image, cmap='gray')
#     axs[0][0].set_title(f'mask_white')
#
#     axs[0][1].imshow(thresh, cmap='gray')
#     axs[0][1].set_title(f'thresh')
#
#     axs[0][2].imshow(gray_binary, cmap='gray')
#     axs[0][2].set_title(f'mask_sat')
#
# plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

col = 3

fig, axs = plt.subplots(3, col, figsize=(20,20), squeeze=False)
[ax.set_axis_off() for ax in axs.ravel()]

bgr_name = 'BGR'
hsv_name = 'HSV'
hls_name = 'HLS'

for i in range(3):
    bgr = display_channel(img, i)
    hls_space = display_channel(img_hls, i)

    x = i//col
    y = (i+col)%col

    axs[x][y].imshow(bgr, cmap='gray')
    axs[x][y].set_title(f'{bgr_name[i]}')

    axs[x+1][y].imshow(hls_space, cmap='gray')
    axs[x+1][y].set_title(f'{hls_name[i]}')

plt.show()


