import os
import cv2
import random
import numpy as np

# img = cv2.imread(r'F:\Nowy folder\10\Praca\Datasets\tu-simple\TEST\1580.jpg')
path = r'C:\Nowy folder\10\Praca\Datasets\Video_data\video4'
list_dir = os.listdir(path)
random = random.sample(list_dir, 1)[0]
# random = 3786
print(random)
img = cv2.imread(os.path.join(path, random))
height = img.shape[0]
width = img.shape[1]
clone = img.copy()
copy = img.copy()


def extract_coordinates(events, x, y, flags, parameters):
    global i
    global points
    global clone
    # add pont
    if events == cv2.EVENT_LBUTTONDOWN:
        if i <= 4:
            points.append((x, y))
        cv2.line(clone, points[i-1], points[-1], (255,0,255), 2)
        i += 1

    # accept
    if events == cv2.EVENT_RBUTTONDOWN:
        breaking = True


def x(elem):
    return elem[0]


def round_num(x, base=5):
    return base * round(x/base)


def draw_lines(image, arr, point_color=(255, 0, 0), line_color=(0, 255, 0)):
    arr = arr.astype(int)
    copy = np.copy(image)
    for i in range(arr.shape[0]):
        x, y = arr[i][0], arr[i][1]
        x_0, y_0 = arr[i - 1][0], arr[i - 1][1]
        cv2.circle(copy, (x, y), radius=1, color=point_color, thickness=10)
        cv2.line(copy, (x, y), (x_0, y_0), color=line_color, thickness=4)
    return copy


i = 0
points = []

cv2.namedWindow('img')
cv2.setMouseCallback('img', extract_coordinates)

while i <= 4:
    cv2.imshow('img', clone)
    if cv2.waitKey(1)==27:
        break

new_points = []
for point in points:
    new_point = []
    for coordinate in point:
        new_point.append(round_num(coordinate, 5))
    new_points.append(new_point)

sorted_points = sorted(new_points[:4], key = x)

width = img.shape[1]
# sorted_points[0] = [0, height]
# sorted_points[-1] = [width, height]

# Wyrównanie poziome
sorted_points[3][1] = sorted_points[0][1]
sorted_points[2][1] = sorted_points[1][1]

# Odbicie lustrzane
sorted_points[3][0] = width-sorted_points[0][0]
sorted_points[2][0] = width-sorted_points[1][0]

print(sorted_points)

for j in range(len(sorted_points)):
    cv2.line(copy, tuple(sorted_points[j]), tuple(sorted_points[j-1]), (0,0,255), 2)

cv2.imshow('img', copy)
cv2.waitKey(0)

src = np.float32(sorted_points)

# video1
# src = np.float32([[290,650], [570,525]])
# pickle.dump(src, open('Pickles/src_video1.p', "wb"))

# video2
# template = [[280, 650], [575, 500]]
# pickle.dump(src, open('Pickles/src_video2.p', "wb"))

# video3
# template = np.float32([[345, 660], [580, 490]])
# pickle.dump(src, open('Pickles/src_video3.p', "wb"))

# video4
template = np.float32([[340, 660], [565, 505]])
# pickle.dump(src, open('Pickles/src_video3.p', "wb"))

src = np.float32([template[0],
                  template[1],
                  [width-template[1][0], template[1][1]],
                  [width-template[0][0], template[0][1]]])

box = draw_lines(img, src, point_color=(255, 0, 0), line_color=(0, 255, 0))

cv2.imshow('box', box)
cv2.waitKey(0)

cv2.imwrite('Test_frames/video_test.jpg', img)

