import os
import pickle

import cv2
import shutil
import numpy as np
from imutils import paths

path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
videos_path = os.path.join(path, 'Videos')
train_path = os.path.join(path, 'train')
test_path = os.path.join(path, 'test')

# if os.path.exists(folder_path):
#     shutil.rmtree(folder_path)

print('Delete previous data? [y/n]')
x = input()
x = x.lower()
if x != 'y' and x != 'n':
    raise Exception('Invalid input')

for folder in train_path, test_path:
    if os.path.exists(folder) and x == 'y':
        shutil.rmtree(folder)

    if not os.path.exists(folder):
        os.mkdir(folder)

fps = 30
interval = 30
video1 = {"name": "Video1.mp4",
          "batch0": (15*fps, 573*fps),
          "batch1": (1815*fps, 3805*fps)}

video2 = {"name": "Video2.mp4",
          "batch0": (380*fps, 1475*fps),
          "batch1": (2093 * fps, 2500 * fps),
          "batch2": (3185 * fps, 5620 * fps),
          "batch3": (5645 * fps, 5830 * fps)}

video3 = {"name": "Video3.mp4",
          "batch0": (17*fps, 2850*fps)}

video4 = {"name": "Video4.mp4",
          "batch0": (290*fps, 1655*fps),
          "batch1": (2125*fps, 2600*fps)}

video_list = [video1]

i = 0
for video in video_list:
    values = list(video.values())[1:]
    video_path = os.path.join(videos_path, video["name"])
    cap = cv2.VideoCapture(video_path)

    frames = 0
    for value in values:
        diff = (value[1] - value[0]) / interval
        frames += diff

    train = int(frames * 0.8)
    test = int(frames - train)

    print(train, test)

    j = 0
    k = 0
    while cap.isOpened():
        _, image = cap.read()
        cropped_img = image[260:, :, :]
        # cropped_img = cv2.copyMakeBorder(cropped_img, 20, 0, 0, 0, cv2.BORDER_REPLICATE)

        if i < train:
            img_path = train_path + fr'\{i:05d}.jpg'
        else:
            img_path = test_path + fr'\{i:05d}.jpg'

        if values[k][0] < j <= values[k][1] and j%interval==0:
            if np.any(image):
                if not os.path.exists(img_path):
                    cv2.imwrite(img_path, cropped_img)
                    print(f'{j}, {i:05d}, saving')
                    i += 1
                else:
                    print(f'{j}, {i:05d}, already exists')
                    i += 1
            else:
                print('corrupted image')

        if j > values[k][1]:
            print(f'k={k}')
            k += 1


        if j > values[-1][1]:
            print('break')
            break

        # if i == 30:
        #     break

        j += 1
    print('new dict')