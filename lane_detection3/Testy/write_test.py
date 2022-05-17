import os
import cv2

# path = r'F:\krzysztof\Maciej_Apostol\StopieńII\Video_data'
path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'

data_path = os.path.join(path, 'test')

video_path = os.path.join(path, r'Videos\Video1.mp4')

cap = cv2.VideoCapture(video_path)

print(video_path)

i = 0
while cap.isOpened():
    _, image = cap.read()

    img_path = data_path + fr'\{i:05d}.jpg'

    cv2.imwrite(img_path, image)

    i += 1