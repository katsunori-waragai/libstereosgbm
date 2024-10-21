import cv2
import matplotlib.pyplot as plt
import skimage

image = cv2.imread("test/test-imgs/left/left_motorcycle.png", cv2.IMREAD_GRAYSCALE)

edge = skimage.filters.sobel(image)

plt.imshow(edge)
plt.show()
