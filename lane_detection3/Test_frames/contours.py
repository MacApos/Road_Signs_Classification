from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def prepare_image(image):
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(5,5), sigmaX=0)
    edges = cv2.Canny(image=blurred, threshold1=70, threshold2=70)
    thresh = cv2.threshold(src=edges, thresh=200, maxval=255, type=cv2.THRESH_BINARY)[1]
    return thresh


# ap = argparse.ArgumentParser()
# ap.add_argument('-j', '--image', required=True, help='path to image')
# args = vars(ap.parse_args())
#
# print(f'OpenCV version: {cv2.__version__}')

answer_key = {0: 1, 1: 3, 2: 0, 3: 2, 4: 1, 5: 3, 6: 4, 7: 1, 8: 3, 9: 0}
questions = len(answer_key)
answer = 5

img_path = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST\450.jpg'
# Wczytywanie obrazu
img = cv2.imread(img_path)
# img = cv2.imread(args['image'])

# Przetwarzanie obrazu
thresh = prepare_image(img)

# Znajdowanie konturów
cnts = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

# Przetwarzanie konturów
cnts = imutils.grab_contours(cnts)

# Wizualizacja konturów
img_copy = img.copy()

for cnt in cnts:
    cv2.drawContours(image=img_copy, contours=[cnt], contourIdx=-1, color=(255, 0, 0), thickness=1)
cv2.imshow('contours', img_copy)


# Selekcja konturów
question_contours = []
#
for cnt in cnts:
    (x, y, h, w) = cv2.boundingRect(cnt)
    area = w/h

    if 0.9 <= area <= 1.0:
        cv2.drawContours(image=img, contours=[cnt], contourIdx=-1, color=(0, 255, 0), thickness=1)
        question_contours.append(cnt)

cv2.imshow('contours', img)
cv2.waitKey(0)
#
# print(f'Dostępnych jest {len(question_contours)} odpowiedzi.')
#
# # Porządkowanie konturów
# correct = 0
#
# question_top_bottom = imutils.contours.sort_contours(cnts=question_contours, method='top-to-bottom')[0]
#
# for idx, val in enumerate(range(0, len(question_contours), 5)):
#     marked = None
#
#     five_contours = question_top_bottom[val:val+answer]
#     five_contours = imutils.contours.sort_contours(cnts=five_contours, method='left-to-right')[0]
#
#     for cnt_idx, cnt_val in enumerate(five_contours):
#         mask = np.zeros(thresh.shape, dtype='uint8')
#
#         cv2.drawContours(image=mask, contours=[cnt_val], contourIdx=-1, color=(255, 255, 255), thickness=-2)
#
#         mask = cv2.bitwise_and(src1=thresh, src2=thresh, mask=mask)
#
#         total = cv2.countNonZero(src=mask)
#         if marked is None or total > marked[0]:
#             marked = (total, cnt_idx)
#
#     color = (0, 0, 255)
#
#     key = answer_key[idx]
#
#     if key == marked[1]:
#         color = (0, 255, 0)
#         correct += 1
#
#     cv2.drawContours(image=img, contours=[five_contours[key]], contourIdx=-1, color=color, thickness=2)
#
# # Przygotowanie wyniku
# score = correct/questions * 100
#
# if score >= 50:
#     color = (0, 255, 0)
#     text = f'{score}% Passed :)'
# else:
#     color = (0, 0, 255)
#     text = f'{score}% Failed :('
#
# checked = cv2.copyMakeBorder(src=img, top=50, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT,
#                              value=(255, 255, 255))
# checked = cv2.putText(img=checked, org=(30, 30), text=text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.9,
#                       color=color, thickness=2)
#
# # cv2.imshow('checked', checked)
# cv2.waitKey(0)