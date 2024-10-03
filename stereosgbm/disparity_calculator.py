from dataclasses import dataclass, field

import numpy as np
import cv2

@dataclass
class DisparityCalculator:
    """
    cv2.StereoSGBM based disparity calculator
    """
    window_size: int = 3
    min_disp: int  = 0
    max_disp: int = 320
    opencv_sgbm: cv2.StereoSGBM = field(default=None)

    def __post_init__(self):
        self.opencv_sgbm = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.max_disp - self.min_disp,
            blockSize=3,
            P1=8 * 3 * self.window_size**2,
            P2=32 * 3 * self.window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

    def calc_by_gray(self, grayL: np.ndarray, grayR: np.ndarray) -> np.ndarray:
        disparity = self.opencv_sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
        return disparity

    def calc_by_brg(self, bgrL: np.ndarray, bgrR: np.ndarray) -> np.ndarray:
        grayL = cv2.cvtColor(bgrL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(bgrR, cv2.COLOR_BGR2GRAY)
        return self.calc_by_gray(grayL, grayR)
