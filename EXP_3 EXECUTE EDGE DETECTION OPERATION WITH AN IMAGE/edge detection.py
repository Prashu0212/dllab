from PIL import Image, ImageFilter

image = Image.open("sample.jpg")
image = image.convert("L")
image = image.filter(ImageFilter.FIND_EDGES)
image.save("output.png")
image.show("output.png")

import cv2

img = cv2.imread("sample.jpg")  # Read image

# Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

# Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)

cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()