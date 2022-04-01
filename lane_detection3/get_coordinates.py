import cv2
import pickle

img = cv2.imread('Test_frames/test2.jpg')
clone = img.copy()
copy = img.copy()


def extract_coordinates(events, x, y, flags, parameters):
    global i
    global points
    global clone
    # add pont
    if events == cv2.EVENT_LBUTTONDOWN:
        if i <= 4:
            points.append([x, y])
        cv2.line(clone, points[i-1], points[-1], (255,0,255), 2)
        i += 1

    # accept
    if events == cv2.EVENT_RBUTTONDOWN:
        i = 0
        points = []
        clone = img.copy()


def x(elem):
    return elem[0]


def round_num(x, base=5):
    return base * round(x/base)

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
sorted_points[0][1] = sorted_points[3][1]
sorted_points[1][1] = sorted_points[2][1]

for j in range(len(sorted_points)):
    cv2.line(copy, sorted_points[j], sorted_points[j-1], (0,0,255), 2)

cv2.imshow('img', copy)
cv2.waitKey(0)

print(sorted_points)
pickle.dump(sorted_points, open('src.p', "wb"))