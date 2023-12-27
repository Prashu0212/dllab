import cv2
import numpy as np
from PIL import Image

img = cv2.imread("sample.jpg",0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))

cv2.imshow('Image',res)

cv2.waitKey(0)
cv2.destroyWindow()