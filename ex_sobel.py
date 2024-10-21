import cv2
import matplotlib.pyplot as plt
import skimage
import numpy as np

from dataclasses import dataclass, field

import stereosgbm

@dataclass
class EdgeBasedDisparityCalculator:
    window_size: int = 3
    min_disp: int = 0
    max_disp: int = 320
    sgbm_disparity_calculator: stereosgbm.DisparityCalculator = field(default=None)

    def __post_init__(self):
        self.sgbm_disparity_calculator = stereosgbm.DisparityCalculator(window_size=self.window_size, min_disp=self.min_disp, max_disp=self.max_disp)

    def predict(self, grayL, grayR):
        left_edge = skimage.filters.sobel(grayL)
        right_edge = skimage.filters.sobel(grayR)
        max1 = np.max(left_edge.flatten())
        max2 = np.max(right_edge.flatten())
        maxv = max((max1, max2))
        left_edge_normalized = ((left_edge / maxv) * 255).astype(np.uint8)
        right_edge_normalized = ((right_edge / maxv) * 255).astype(np.uint8)
        return self.sgbm_disparity_calculator.predict(left_edge_normalized, right_edge_normalized)

if __name__ == "__main__":
    window_size = 3
    min_disp = 0
    max_disp = 128

    left_imgs = ["test/test-imgs/left/left_motorcycle.png"]
    right_imgs = ["test/test-imgs/right/right_motorcycle.png"]

    edge_disparity_calculator = EdgeBasedDisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )

    for left_name, right_name in zip(left_imgs, right_imgs):
        left_image = cv2.imread(left_name, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_name, cv2.IMREAD_GRAYSCALE)
        edge_disparity = edge_disparity_calculator.predict(left_image, right_image)

        plt.imshow(edge_disparity)
        plt.show()
