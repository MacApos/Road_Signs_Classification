import os
import cv2

path = 'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
img_name = os.path.join(path, '01.jpg')

print(img_name)

# if not os.path.exists(test):
#     print('not exists')
#     os.mkdir(test)


image = cv2.imread('C:/Users/macie/PycharmProjects/Road_Signs_Classification/lane_detection3/Pictures/gray.jpg')

# cv2.imshow(img_name, image)
# cv2.waitKey(0)

if not cv2.imwrite(img_name, image):
     raise Exception("Could not write image")
