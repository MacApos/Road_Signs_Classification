import cv2
import numpy as np
import matplotlib.pyplot as plt

with open('threshold.npy', 'rb') as file:
    image = np.load(file)

# Sumowanie pikseli z połowy obrazu względem osi pionowej.
# np.sum([[0, 1],
#         [0, 5]], axis=0)
# >> array([0, 6])
histogram = np.sum(image[image.shape[0]//2:, :], axis=0)

# Ustawianie zdjęć w stos względem głębokości. Na jednym kanale nie było by widać kolorowych prostokątów.
out_img = np.dstack((image, image, image)) * 255
# print(out_img.shape)

# Punkty przesuwającego się okna
midpoint = int(histogram.shape[0] // 2)
# Największ wartość wykryta od początku do środka (lewa linia)
leftx_base = np.argmax(histogram[:midpoint])
# Największ wartość wykryta od środka do końca (prawa linia)
rightx_base = np.argmax(histogram[midpoint:])+midpoint
# Image
#  _______________________________________
# |                                       |
# | leftx_base    midpoint    rightx_base |
# |_______________________________________|
# print(leftx_base, '-', midpoint, '-', rightx_base)

# Liczba okien
nwindows = 9
# Margines
margain = 100
# ???
minpix = 50

window_height = image.shape[0] // nwindows


nonzero = image.nonzero()

nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# print(len(nonzero[1]), '\n')

leftx_current = leftx_base
rightx_current = rightx_base

left_lane_inds = []
right_lane_inds = []

# Prostokąty, narazie wszystkie są równe w pionie
for window in range(nwindows):
    win_y_low = image.shape[0] - (window+1) * window_height
    win_y_high = image.shape[0] - window * window_height
    win_xleft_low = leftx_current - margain
    win_xleft_high = leftx_current + margain
    win_xright_low = rightx_current - margain
    win_xright_high = rightx_current + margain

    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 4)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 4)

    # Indeksy niezerowych pikseli wewnątrz okna
    # Tam gdzie nonzeroy są większe lub równe dolnej ganicy okna i mniejsze od górnej, tak samo nonzerox
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                      (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                       (nonzerox < win_xright_high)).nonzero()[0]

    # Przypisanie indeksów do listy spoza pętli. Tworzy się lista list, trzeba ją potem połączyć.

    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    # Jeżeli liczba znalezionych pikseli jest większa niż założona wartość (50px) wtedy granice okna się aktualizują.
    # Zachodzi to tak, że niezerowe piksele z orginalnego obrazu o indeksach z niezerowych pikseli z okna są uśredniane.
    # Wszystko się dzieje w osi x bo okna są przesuwane tylko w poziomie, w pionie mają stałą wartość.
    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_right_inds]))

# Lączenie listy list
try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
except AttributeError:
    pass

leftx = nonzerox[left_lane_inds]
rightx = nonzerox[right_lane_inds]
lefty = nonzeroy[left_lane_inds]
righty = nonzeroy[right_lane_inds]

# cv2.imshow('image', out_img)
cv2.waitKey(0)
plt.hist(histogram)
# plt.show()