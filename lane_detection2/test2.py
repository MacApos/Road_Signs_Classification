import numpy as np
import cv2

img = np.zeros((600, 1000, 3), np.uint8)

# setup text
font = cv2.FONT_HERSHEY_SIMPLEX
text = "Hello Joseph!!"

# get boundary of this text
textsize = cv2.getTextSize(text, font, 1, 2)[0]

# get coords based on boundary
textX = (img.shape[1] - textsize[0]) // 2
textY = (img.shape[0] + textsize[1]) // 2

# add text centered on image
cv2.putText(img, text, (textX, textY ), font, 1, (255, 255, 255), 2)

# display image
cv2.imshow('image', img)
cv2.waitKey(0)