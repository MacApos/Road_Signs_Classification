import os
import cv2
import glob
import time
import random
import shutil
import numpy as np

path = r'C:\Nowy folder\10\Praca\Datasets\Video_data\Videos'
train_path = r'C:\Nowy folder\10\Praca\Datasets\Video_data\train_set'
test_path = r'C:\Nowy folder\10\Praca\Datasets\Video_data\test_set'



fps = 30
interval = 15
# video1 = {"name": "Video1.mp4",
#           "batch0": (15*fps, 573*fps),
#           "batch1": (1815*fps, 3805*fps)}
#
# video2 = {"name": "Video2.mp4",
#           "batch0": (380*fps, 1475*fps),
#           "batch1": (2093 * fps, 2500 * fps),
#           "batch2": (3185 * fps, 5620 * fps),
#           "batch3": (5645 * fps, 5830 * fps)}
#
# video3 = {"name": "Video3.mp4",
#           "batch0": (17*fps, 2850*fps)}
#
# video4 = {"name": "Video4.mp4",
#           "batch0": (281*fps, 1655*fps),
#           "batch1": (2125*fps, 2600*fps)}

rand = []
diff = []

for a in range(0,4):
    temp_rand = [random.randint(0, 100) for b in range(0,2)]
    for c in sorted(temp_rand):
        rand.append(c*fps)

diff = [n+(random.randint(1, 3)*fps) for n in rand]

for d in range(0,8):
    if d%2==0:
        print()
    print(f'batch{d%2} = [{rand[d]}, {diff[d]}]')

video1 = {"name": "Video1.mp4",
          "batch0": (rand[0], diff[0]),
          "batch1": (rand[1], diff[1])}

video2 = {"name": "Video2.mp4",
          "batch0": (rand[2], diff[2]),
          "batch1": (rand[3], diff[3])}

video3 = {"name": "Video3.mp4",
          "batch0": (rand[4], diff[4]),
          "batch1": (rand[5], diff[5])}

video4 = {"name": "Video4.mp4",
          "batch0": (rand[6], diff[6]),
          "batch1": (rand[7], diff[7])}

video_dict = [video1, video2, video3, video4]

frames_num = 0
for video in video_dict:
    values = list(video.values())[1:]
    for value in values:
        frames_num += (value[1] - value[0])

# frames_num = frames_num//interval
# print(frames_num)
# indices = random.sample(range(frames_num), frames_num)
#
# train_size = int(frames_num*0.7)
#
# print(train_size, frames_num-train_size)

for data_path in train_path, test_path:
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        os.mkdir(data_path)
    else:
        os.mkdir(data_path)

i = 0
for video in video_dict:
    values = list(video.values())[1:]

    video_path = os.path.join(path, video["name"])
    cap = cv2.VideoCapture(video_path)

    j = 0
    k = 0
    flag = cap.isOpened()

    while flag:
        if i == frames_num:
            break
        _, image = cap.read()
        img_path = train_path + fr'\{i}.jpg'

        if values[k][0] < j <= values[k][1] and i%interval==0:
            # if not os.path.exists(img_path):
            if np.any(image):
                cv2.imwrite(img_path, image)
                print(f'{j}, {i} - copying')
                i += 1
        elif j > values[k][1]:
            k += 1
            print('next', k)
        # else:
        #     # print(j)
        j += 1
        if j > values[-1][1]:

            break
#     print('next_dict')
#
# def sort_path(path):
#     sorted_path = []
#     for file in os.listdir(path):
#         number = int(''.join(n for n in file if n.isdigit()))
#         sorted_path.append(number)
#
#     sorted_path = sorted(sorted_path)
#     return [path + fr'\{str(f)}.jpg' for f in sorted_path]
#
# sorted_train = sort_path(train_path)
#
# test_idx = 0
# for train_idx, train_img in enumerate(sorted_train):
#     if train_idx>=train_size:
#         test_img = test_path + fr'\{test_idx}.jpg'
#         # print(train_idx, train_img, '->', test_img)
#         os.replace(train_img, test_img)
#         test_idx += 1