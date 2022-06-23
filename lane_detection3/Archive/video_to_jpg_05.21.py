import os
import cv2
import shutil
import numpy as np
from imutils import paths

path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
videos_path = os.path.join(path, 'Videos')
data_path = os.path.join(path, 'data')

# if os.path.exists(data_path):
    # shutil.rmtree(data_path)

if not os.path.exists(data_path):
    os.mkdir(data_path)

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

video_list = [video1, video2]

limit = 100
i = 0
for video in video_list:
    # j = 0
    # single_video_path = os.path.join(path, video["name"].split('.mp4')[0])

    # if not os.path.exists(single_frames_path):
    #     os.mkdir(single_frames_path)
    # else:
    #     # shutil.rmtree(frames_path)
    #     # os.mkdir(frames_path)
    #     pass

    # if not os.path.exists(single_video_path):
    #     # shutil.rmtree(single_frames_path)
    #     os.mkdir(single_video_path)
    #
    # for src in list(paths.list_images(data_path))[i:i+limit]:
    #     dst = single_video_path + fr'\{j:05d}.jpg'
    #     print(src, '    ->  ',dst)
    #     shutil.copy(src, dst)
    #     j += 1
    #
    # i += limit
    # print()

    # single_frames_path = os.path.join(path, video["name"].split('.mp4')[0])
    # for src in list(paths.list_images(single_frames_path)):
    #     dst = data_path + fr'\{i:05d}.jpg'
    #     print(dst)
    #     shutil.copy(src, dst)
    #     i += 1

    frames_num = 0
    values = list(video.values())[1:]
    for value in values:
        frames_num += (value[1] - value[0])

    frames_num = frames_num // interval
    print(frames_num)

    video_path = os.path.join(videos_path, video["name"])
    cap = cv2.VideoCapture(video_path)

    print(video_path)
    j = 0
    k = 0
    while cap.isOpened():
        _, image = cap.read()
        resized_img = image[260:, :, :]
        resized_img = cv2.copyMakeBorder(resized_img, 20, 0, 0, 0, cv2.BORDER_REPLICATE)

        img_path = data_path + fr'\{i:05d}.jpg'

        if values[k][0] < j <= values[k][1] and j%interval==0:
            if np.any(image):
                if not os.path.exists(img_path):
                    cv2.imwrite(img_path, resized_img)

                    print(f'{j}, {i:05d}, copy')
                    i += 1
                else:
                    print(f'{j}, {i:05d}, already exists')
                    i += 1
            else:
                print('corrupt image')

        if j > values[k][1]:
            print('k+1, k=', k)
            print(f'{values[k][0]}')
            k += 1


        if j > values[-1][1]:
            print('break')
            break

        if i == 30:
            break

        j += 1
    print('new dict')


def sort_path(path):
    sorted_path = []
    for file in os.listdir(path):
        number = int(''.join(n for n in file if n.isdigit()))
        sorted_path.append(number)

    sorted_path = sorted(sorted_path)
    return [path + fr'\{str(f)}.jpg' for f in sorted_path]

# i = 0
# for src in sort_path(video1_path):
#     dst = os.path.join(data_path, f'{i:05d}.jpg')
#     shutil.copy(src, dst)
#     i += 1

# test_idx = 0
# for train_idx, train_img in enumerate(sorted_train):
#     if train_idx>=train_size:
#         test_img = test_path + fr'\{test_idx}.jpg'
#         print(train_idx, train_img, '->', test_img)
#         os.replace(train_img, test_img)
#         test_idx += 1
