import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\DELL\PycharmProjects\DeepLearningLAB\Basic_Image_Processing\anantha.jpg")
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
plt.imshow(erosion, cmap='gray')
plt.show()

dilation = cv2.dilate(img,kernel,iterations = 1)
plt.imshow(dilation, cmap='gray')
plt.show()

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
plt.imshow(opening, cmap='gray')
plt.show()

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing, cmap='gray')
plt.show()

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing, cmap='gray')
plt.show()

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
plt.imshow(tophat, cmap='gray')
plt.show()