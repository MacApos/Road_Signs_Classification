import os
import cv2
import pickle
import random
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(filename):
    file = open(filename, 'rb')
    array = pickle.load(file)
    file.close()

    return array


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def color_threshold(image, s_thresh, v_thresh):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    hlv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hlv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1

    return output


def abs_sobel(image, orientation, sobel_thresh):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]

    if orientation == 'x':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    if orientation == 'y':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))

    output = np.zeros_like(abs_sobel)
    output[(abs_sobel >= sobel_thresh[0]) & (abs_sobel <= sobel_thresh[1])] = 1

    return output

def sobel(image):
    output = np.zeros_like(image[:, :, 1])
    sobel_x = abs_sobel(image, "x", (25, 255))
    sobel_y = abs_sobel(image, "y", (25, 255))
    c_binary = color_threshold(image, (25, 255), (25, 255))
    output[(sobel_x==1)&(sobel_y==1) | (c_binary==1)]=255

    return output

path = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
list_dir = os.listdir(path)
random = random.randint(0, len(list_dir)-1)
random = 861
print(random)
image = cv2.imread(os.path.join(path, f'{random}.jpg'))



src = np.float32([[0, 720],
                  [450, 300],
                  [850, 300],
                  [1280, 720]])

dst = np.float32([src[0],
                  [src[0][0], 0],
                  [src[-1][0], 0],
                  src[-1]])

height = image.shape[0]
width = image.shape[1]

def warp_perspective(image, from_=src, to=dst):
    M = cv2.getPerspectiveTransform(from_, to)
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warp, M

mtx = load('Pickles/mtx.p')
dist = load('Pickles/dist.p')

undist = cv2.undistort(image, mtx, dist, None, mtx)

warp, _ = warp_perspective(image)
# plt.show()

row = 2
col = 2

fig, axs = plt.subplots(row, col, figsize=(20,20))
[ax.set_axis_off() for ax in axs.ravel()]

range = np.linspace(0, 250-(250//(col*row)), col*row).astype(int)

# for idx, val in enumerate(range):
#     no_scale = np.zeros_like(image[:, :, 0])
#     no_scale_x, scale_x = abs_sobel(image, "x", (25, 255))
#     no_scale_y, scale_y = abs_sobel(image, "y", (25, 255))
#     c_binary = color_threshold(image, (25, 255), (25, 255))
#     no_scale[(no_scale_x==1)&(no_scale_y==1) | (c_binary==1)]=255
#
#     x = idx//col
#     y = (idx+col)%col
#     axs[x][y].imshow(rgb(no_scale))
#     axs[x][y].set_title(f'{val}')

# plt.show()

gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

max_val = np.mean(np.amax(gray, axis=1)).astype(int)
print(max_val)

if max_val > (255 * 0.75):
    max_val = int(max_val * 0.75)

sobel = sobel(warp)

_, thresh = cv2.threshold(gray, max_val, 255, cv2.THRESH_BINARY)
# output = preprocess(image)

contours_sobel, hierarchy_sobel = cv2.findContours(sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
minpix = 50
for contour in contours_sobel:
    area = cv2.contourArea(contour)
    if area < minpix:
        cv2.drawContours(sobel, [contour], -1, (0), -1)

contours_thresh, hierarchy_thresh = cv2.findContours(sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contour in contours_thresh:
    area = cv2.contourArea(contour)
    if area < minpix:
        cv2.drawContours(thresh, [contour], -1, (0), -1)

bitwise = cv2.bitwise_and(gray, sobel)

blur = cv2.GaussianBlur(gray,(5,5),0)
c_binary = color_threshold(image, (25, 255), (25, 255))
canny = cv2.Canny(blur, 50, 150)

lines = cv2.HoughLinesP(canny, 2, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=5)
for line in lines:
    x1, y1, x2, y2 = line.reshape(4)
    print(x1, y1, x2, y2)
    parameters = np.polyfit((x1, x2), (y1, y2), 1)
    slope = parameters[0]
    intercept = parameters[1]
    cv2.line(warp, (x1, y1), (x2, y2), (0, 255, 0), 4)

cv2.imshow('warp', warp)
cv2.waitKey(0)


max_val = np.mean(np.amax(bitwise, axis=1)).astype(int)
_, thresh1 = cv2.threshold(bitwise, 125, 255, cv2.THRESH_BINARY)

titles = ['gray', 'thresh', 'sobel', 'thresh1']
for idx, val in enumerate([gray, thresh, sobel, thresh1]):
    plt.subplot(2,2,idx+1)
    plt.imshow(rgb(val))
    plt.title(f'{titles[idx]}')
    plt.axis('off')

plt.show()
