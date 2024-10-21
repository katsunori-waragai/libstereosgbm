import cv2
import matplotlib.pyplot as plt
import skimage
import numpy as np

import stereosgbm


left_image = cv2.imread("test/test-imgs/left/left_motorcycle.png", cv2.IMREAD_GRAYSCALE)
left_edge = skimage.filters.sobel(left_image)

print(f"{left_edge.shape=}")

right_image = cv2.imread("test/test-imgs/right/right_motorcycle.png", cv2.IMREAD_GRAYSCALE)
right_edge = skimage.filters.sobel(right_image)

print(f"{right_edge.shape=}")

max1 = np.max(left_edge.flatten())
max2 = np.max(right_edge.flatten())
maxv = max((max1, max2))
left_edge_normalized = ((left_edge / maxv) * 255).astype(np.uint8)
right_edge_normalized = ((right_edge / maxv) * 255).astype(np.uint8)

window_size = 3
min_disp = 0
max_disp = 128

disparity_caluculator = stereosgbm.DisparityCalculator(
    window_size=window_size, min_disp=min_disp, max_disp=max_disp
)

disparity = disparity_caluculator.predict(left_edge_normalized, right_edge_normalized)

plt.imshow(disparity)
plt.show()
