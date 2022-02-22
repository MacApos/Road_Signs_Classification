import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_lane_pixels(image):
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
    margin = 100

    minpix = 50

    window_height = int(image.shape[0] // nwindows)

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
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 4)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 4)

        # Indeksy niezerowych pikseli wewnątrz okna
        # Tam gdzie nonzeroy są większe lub równe dolnej ganicy okna i mniejsze od górnej, tak samo nonzerox. Zwraca
        # Fałsz lub Prawdę (0 lub 1),  za pomocą nonzero wybiera się tylko 1.
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Przypisanie indeksów do listy spoza pętli. Tworzy się lista list, trzeba ją potem połączyć.

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Jeżeli liczba znalezionych pikseli jest większa niż założona wartość (50px) wtedy granice okna się aktualizują.
        # Zachodzi to tak, że niezerowe piksele z orginalnego obrazu o indeksach niezerowych pikseli z okna są uśredniane.
        # Wszystko się dzieje w osi x bo okna są przesuwane tylko w poziomie, w pionie mają stałą wartość.
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Lączenie listy list
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except AttributeError:
        pass

    # Wyciągnięcie pikseli przedstawiających linie ze zdjęcia
    leftx = nonzerox[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

# find_lane_pixels(image)

# np.polyfit – Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y).
# Aproksymacja zbioru danych linią. Jeżeli stopień będzie określony na 2 zwróci współczynniki równania kwadratowego
# y=ax^2+bx+c – zwróci a, b i c
# Ma to na celu ukierunkowanie dalszego wyszukiwanie w dobrze znanych obszarach, czyli prostokątach.
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Zakres zbioru danych
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Równanie kwadratowe
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty

def search_around_poly(image):
    # Te wartości trzeba zdefiniować jeszcze raz bo są w innej funkcji
    margin = 100

    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(image)

    # Ukierunkowane wyszukiwanie
    if (len(leftx) == 0 or len(rightx) == 0) or (len(righty) == 0 or len(lefty) == 0):
        out_img = np.dstack((image, image, image))*255
        left_curverad = 0
        right_curverad = 0
        print('0')

    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        left_left_poly = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin/2
        right_left_poly = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin/2

        left_right_poly = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin/2
        right_right_poly = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin/2


        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        print(np.array_equal(leftx, nonzeroy))

        left_fitx, right_fitx, ploty = fit_poly(image.shape, leftx, lefty, rightx, righty)

        ax = plt.subplot()
        ax.scatter(leftx, -lefty, c='g')
        ax.scatter(rightx, -righty, c='r')
        ax.plot(left_left_poly, -nonzeroy, c='b')
        ax.plot(right_left_poly, -nonzeroy, c='b')
        ax.plot(left_right_poly, -nonzeroy, c='m')
        ax.plot(right_right_poly, -nonzeroy, c='m')
        ax.plot(left_fitx, -ploty, c='c')
        ax.plot(right_fitx, -ploty, c='c')

        plt.show()

        # Konwersja wartości pikseli na dane rzeczywiste, założono długóśc równą 30m, a szerokość 3.7m
        ym_per_px = 30 / 720 # metry na piksel w osi y
        xm_per_px = 3.7 / 650  # metry na piksel w osi x

        # Obliczanie krzywizny
        left_fit_cr = np.polyfit(ploty * ym_per_px, left_fitx * xm_per_px, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_px, right_fitx * xm_per_px, 2)
        y_eval = np.max(ploty)



        # Promień krzywsizny:
        # R = ((1 + (dy/dx) ** 2) ** 3/2) / (d2y/dx2)
        # y = ax ** 2 + bx + c
        # dy/dx = 2*ax + b
        # d2y/dx2 = 2*a
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_px + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_px + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])

        # Wizualizacja
        out_img = np.dstack((image, image, image)) * 255
        window_img = np.zeros_like(out_img)
        # Linie w kolorze
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

        left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        points = np.hstack((left, right))
        out_img = cv2.fillPoly(out_img, np.int_(points), (0, 255, 0))

    return out_img, left_curverad, right_curverad


with open(r'test/threshold.npy', 'rb') as file:
    image = np.load(file, allow_pickle=True)


leftx, lefty, rightx, righty, lanes = find_lane_pixels(image)
img_shape = image.shape
left_fitx, right_fitx, ploty = fit_poly(img_shape, leftx, lefty, rightx, righty)
out_img, left_curverad, right_curverad = search_around_poly(image)
print(left_curverad, right_curverad)
# cv2.imshow('image', image)
# cv2.imshow('lanes', lanes)
cv2.imshow('poly', out_img)
cv2.waitKey(0)