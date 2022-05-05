import os
import cv2
import shutil
from imutils import paths

path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
labels_path = os.path.join(path, 'labels')
frames_path = os.path.join(path, 'frames')
video_name = os.path.join(path, 'label_video.avi')

if not os.path.exists(frames_path):
    os.mkdir(frames_path)
else:
    shutil.rmtree(frames_path)
    os.mkdir(frames_path)
    pass

fps = 15
seconds = 10

start = 2175
stop = start + fps*seconds

labels_list = list(paths.list_images(labels_path))
for src in labels_list[start:stop]:
    dst = os.path.join(frames_path, src.split('\\')[-1])
    shutil.copyfile(src, dst)

label_list = list(paths.list_images(frames_path))

frame = cv2.imread(label_list[0])
height, width, layers = frame.shape


video = cv2.VideoWriter(video_name, 0, fps, (width,height))


for label in label_list[:fps*seconds]:
    video.write(cv2.imread(label))

cv2.destroyAllWindows()
video.release()