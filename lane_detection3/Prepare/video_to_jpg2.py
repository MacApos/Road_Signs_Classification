import os
import cv2
import glob
import time
import random
import shutil
import numpy as np

path = r'F:\Nowy folder\10\Praca\Datasets\Video_data\Videos'
train_path = os.path.join(path, 'train_set')
print(train_path)
frames_path = r'F:\Nowy folder\10\Praca\Datasets\Video_data\frames'
train_path = r'F:\Nowy folder\10\Praca\Datasets\Video_data\train_set'
test_path = r'F:\Nowy folder\10\Praca\Datasets\Video_data\test_set'

fps = 30
interval = 15
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
          "batch0": (281*fps, 1655*fps),
          "batch1": (2125*fps, 2600*fps)}

# rand = []
# diff = []
#
# for a in range(4):
#     temp_rand = [random.randint(10, 100) for c in range(2)]
#     for d in sorted(temp_rand):
#         rand.append(d*fps)
#
# diff = [n+(random.randint(50, 100)*fps) for n in rand]
#
# for d in range(0,8):
#     if d%2==0:
#         print()
#     print(f'batch{d%2} = [{rand[d]}, {diff[d]}]')
#
# video1 = {"name": "Video1.mp4",
#           "batch0": (2490, 2700),
#           "batch1": (2730, 4560)}
#
# video2 = {"name": "Video2.mp4",
#           "batch0": (2430, 2880),
#           "batch1": (2910, 4920)}
#
# video3 = {"name": "Video3.mp4",
#           "batch0": (750, 2700),
#           "batch1": (2730, 4170)}
#
# video4 = {"name": "Video4.mp4",
#           "batch0": (1650, 4350),
#           "batch1": (4380, 5340)}

# video1 = {"name": "Video1.mp4",
#           "batch0": (rand[0], diff[0]),
#           "batch1": (rand[1], diff[1])}
#
# video2 = {"name": "Video2.mp4",
#           "batch0": (rand[2], diff[2]),
#           "batch1": (rand[3], diff[3])}
#
# video3 = {"name": "Video3.mp4",
#           "batch0": (rand[4], diff[4]),
#           "batch1": (rand[5], diff[5])}
#
# video4 = {"name": "Video4.mp4",
#           "batch0": (rand[6], diff[6]),
#           "batch1": (rand[7], diff[7])}

video_dict = [video1]

frames_num = 0
for video in video_dict:
    values = list(video.values())[1:]
    for value in values:
        frames_num += (value[1] - value[0])
#
frames_num = frames_num//interval

indices = random.sample(range(frames_num), frames_num)
train_size = 1400
# int(frames_num*0.7)

print(frames_num, train_size, frames_num-train_size)

# for data_path in train_path, test_path:
if os.path.exists(frames_path):
    shutil.rmtree(frames_path)
    os.mkdir(frames_path)
else:
    os.mkdir(frames_path)

i = 0
for video in video_dict:
    values = list(video.values())[1:]

    video_path = os.path.join(path, video["name"])
    cap = cv2.VideoCapture(video_path)

    j = 0
    k = 0
    while cap.isOpened():
        _, image = cap.read()
        img_path = frames_path + fr'\{i}.jpg'

        if values[k][0] < j <= values[k][1] and j%interval==0:
            if np.any(image):
                if not os.path.exists(img_path):
                    cv2.imwrite(img_path, image)
                    print(f'{j}, {i}, train - copy')
                    i += 1
                else:
                    print('already exists')
                    pass

        if j > values[-1][1] or i==2000:
            print('break')
            break

        elif j > values[k][1]:
            print('k+1, k=', k)
            k += 1
        j += 1


def sort_path(path):
    sorted_path = []
    for file in os.listdir(path):
        number = int(''.join(n for n in file if n.isdigit()))
        sorted_path.append(number)

    sorted_path = sorted(sorted_path)
    return [path + fr'\{str(f)}.jpg' for f in sorted_path]

sorted_train = sort_path(frames_path)

# test_idx = 0
# for train_idx, train_img in enumerate(sorted_train):
#     if train_idx>=train_size:
#         test_img = test_path + fr'\{test_idx}.jpg'
#         print(train_idx, train_img, '->', test_img)
#         os.replace(train_img, test_img)
#         test_idx += 1
