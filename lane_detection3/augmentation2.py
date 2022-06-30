import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import os
import cv2
from imutils import paths

path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

save_path = '../lane_detection3/Pictures'
if not os.path.exists(save_path):
    os.mkdir(save_path)

M = np.load('Pickles/M_video1.npy')
M_inv = np.load('Pickles/M_inv_video1.npy')
image = cv2.imread(test_list[0])

width = image.shape[1]
image = cv2.resize(image, (width, width//2))
warp = cv2.warpPerspective(image, M, (width, width//2), flags=cv2.INTER_LINEAR)
img = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)

#
# cv2.imwrite(save_path + '/road.jpg', image)
#
# img = load_img(test_list[0])
#
data = img_to_array(img)

samples = np.expand_dims(data, 0)

datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.5,
                             rotation_range=10,
                             brightness_range=[0.6, 1.1],
                             shear_range=20,
                             zoom_range=0.3,
                             horizontal_flip=True
                             )
iterator = datagen.flow(samples, batch_size=1)

plt.figure(figsize=(16, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = iterator.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()
