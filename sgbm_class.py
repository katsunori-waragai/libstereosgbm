from dataclasses import dataclass, field

import numpy as np
import cv2
from matplotlib import pyplot as plt

print('loading images...')
bgrL = cv2.imread('test/test-imgs/left/left_motorcycle.png')
bgrR = cv2.imread('test/test-imgs/right/right_motorcycle.png')

@dataclass
class DisparityCalculator:
    window_size = 3
    min_disp = 0
    num_disp = 320 - min_disp
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 3,
    # P1 = 8 * 3 * window_size ** 2,
    # P2 = 32 * 3 * window_size ** 2,
    # disp12MaxDiff = 1,
    # uniquenessRatio = 10,
    # speckleWindowSize = 100,
    # speckleRange = 32

    opencv_sgbm: cv2.StereoSGBM = field(default=None)

    def __post_init__(self):
        self.opencv_sgbm = cv2.StereoSGBM_create(minDisparity=min_disp,
                                       numDisparities=num_disp,
                                       blockSize=3,
                                       P1=8 * 3 * window_size ** 2,
                                       P2=32 * 3 * window_size ** 2,
                                       disp12MaxDiff=1,
                                       uniquenessRatio=10,
                                       speckleWindowSize=100,
                                       speckleRange=32
                                       )

    def calc_by_gray(self, grayL: np.ndarray, grayR: np.ndarray):
        disparity = self.opencv_sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
        return disparity

    def calc_by_brg(self, bgrL: np.ndarray, bgrR: np.ndarray) -> np.ndarray:
        grayL = cv2.cvtColor(bgrL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(bgrR, cv2.COLOR_BGR2GRAY)
        return self.calc_by_gray(grayL, grayR)

# disparity range tuning
# https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html
window_size = 3
min_disp = 0
num_disp = 320 - min_disp

disparity_caluculator = DisparityCalculator()
disparity = disparity_caluculator.calc_by_brg(bgrL, bgrR)

print("saving disparity as disparity_image_sgbm.txt")
np.savetxt("data/disparity_image_sgbm.txt", disparity, fmt = '%3.2f', delimiter = ' ', newline = '\n')

# plt.imshow(imgL, 'gray')
plt.imshow(disparity, 'gray')
# plt.imshow('disparity', (disparity - min_disp) / num_disp)
plt.show()

