import cv2
import imageio
from imutils import paths

path = r'F:\Zalman\BCHperform\Audi\Render\gif'
save_path = path + r'\Render.gif'

images_list = list(paths.list_images(path))

images = []
i = 0
for image in images_list:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow(f'{i}', img)
    # cv2.waitKey(0)
    images.append(img)
    i += 1


with imageio.get_writer(save_path, mode='I', fps=1) as writer:
    for image in images:
        print('saving')
        writer.append_data(image)