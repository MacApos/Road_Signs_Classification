import os
import cv2
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Nowy folder\10\Praca\Datasets\Video_data\train_set'
# path = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
list_dir = os.listdir(path)
random = random.randint(0, len(list_dir)-1)
print(random)
image = cv2.imread(os.path.join(path, f'{random}.jpg'))

height = image.shape[0]
width = image.shape[1]

def draw_lines(image, arr, point_color=(255, 0, 0), line_color=(0, 255, 0)):
    arr = arr.astype(int)
    copy = np.copy(image)
    j = 0
    for i in range(arr.shape[0]):
        x, y = arr[i][0], arr[i][1]
        x_0, y_0 = arr[i - 1][0], arr[i - 1][1]
        cv2.circle(copy, (x, y), radius=1, color=point_color, thickness=5)
        cv2.line(copy, (x, y), (x_0, y_0), color=line_color, thickness=1)
        cv2.putText(copy, f'j={j}', (x + 50, y + 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        j += 1
    return copy

file = open('Pickles/src.p', 'rb')
src = pickle.load(file)
file.close()

pts1 = np.float32([[290,675],[570,525],[710,525],[990,675]])
# pts1 = np.float32([[290,675],[570,525],[690,525],[990,675]])
# pts1 = src
# pts1 = np.float32([[290, 670],
#                    [580, 500],
#                    [660, 500],
#                    [990, 670]])
print(pts1)
new_height = int(pts1[0][1] - pts1[2][1])
new_width = int(pts1[3][0] - pts1[0][0])

print(new_height, new_width)

pts2 = np.float32([[0,height],[0,0],[width,0],[width,height]])
# pts2 = np.float32([[0,300],[0,0],[800,0],[800,300]])

pts3 = np.add(pts1, pts2)

box = draw_lines(image, pts1)
box = draw_lines(box, pts2)

cv2.imshow('img', box)
cv2.waitKey(0)

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(image,M,(width,height))
plt.subplot(121),plt.imshow(image),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
