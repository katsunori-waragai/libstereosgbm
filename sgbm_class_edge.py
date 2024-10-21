import cv2
import matplotlib.pyplot as plt
import skimage
import numpy as np

from dataclasses import dataclass, field

import stereosgbm

if __name__ == "__main__":
    window_size = 3
    min_disp = 0
    max_disp = 128

    left_imgs = ["test/test-imgs/left/left_motorcycle.png"]
    right_imgs = ["test/test-imgs/right/right_motorcycle.png"]

    edge_disparity_calculator = stereosgbm.EdgeBasedDisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )

    disparity_calculator = stereosgbm.DisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )

    for left_name, right_name in zip(left_imgs, right_imgs):
        left_image = cv2.imread(left_name, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_name, cv2.IMREAD_GRAYSCALE)
        edge_disparity = edge_disparity_calculator.predict(left_image, right_image)
        disparity = disparity_calculator.predict(left_image, right_image)

        plt.subplot(1, 2, 1)
        plt.imshow(disparity)
        plt.subplot(1, 2, 2)
        plt.imshow(edge_disparity)
        plt.show()
