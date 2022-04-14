import os
import cv2
import time
import numpy as np

path = r'C:\Nowy folder\10\Praca\Datasets\Video_data\Videos'
dst = r'C:\Nowy folder\10\Praca\Datasets\Video_data\train_set'

fps = 30
# video1 = {"name": "Video1.mp4",
#           "batch0": (15*fps, 573*fps)}
#
# # "batch1": (1815*fps, 3805*fps)
#
# video2 = {"name": "Video2.mp4",
#           "batch0": (380*fps, 1475*fps)}
#
# # "batch1": (2093 * fps, 2500 * fps),
# # "batch2": (3185 * fps, 5620 * fps),
# # "batch3": (5645 * fps, 5830 * fps)
#
# video3 = {"name": "Video3.mp4",
#           "batch0": (17*fps, 2850*fps)}
#
# video4 = {"name": "Video4.mp4",
#           "batch0": (281*fps, 1655*fps)}
#
# # "batch1": (2125*fps, 2600*fps)

video1 = {"name": "Video1.mp4",
          "batch0": (1*fps, 2*fps),
          "batch1": (3*fps, 5*fps)}

video2 = {"name": "Video2.mp4",
          "batch0": (1*fps, 2*fps),
          "batch1": (3*fps, 5*fps)}

video3 = {"name": "Video3.mp4",
          "batch0": (1*fps, 2*fps),
          "batch1": (3*fps, 5*fps)}

video4 = {"name": "Video4.mp4",
          "batch0": (1*fps, 2*fps),
          "batch1": (3*fps, 5*fps)}

video_dict = [video1, video2, video3, video4]

frames_num = 0
for video in video_dict:
    values = list(video.values())[1:]
    for value in values:
        frames_num += (value[1] - value[0])

print(frames_num)

i = 0
for video in video_dict:
    values = list(video.values())[1:]

    video_path = os.path.join(path, video["name"])
    cap = cv2.VideoCapture(video_path)

    j = 0
    k = 0
    while (cap.isOpened()):
        _, image = cap.read()
        if values[k][0] < j <= values[k][1] and j%15==0:
            img_path = dst + fr'\{i}.jpg'
            # if not os.path.exists(img_path):
            if np.any(image):
                cv2.imwrite(img_path, image)
                print('copying')
            # else:
            #     print('image already exists')
            i += 1
        elif j > values[k][1]:
            print('next')
            k += 1
        else:
            print(j)
        j += 1

        if j > values[-1][1]:
            break