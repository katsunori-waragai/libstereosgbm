"""
wrapper library for cv2.StereoSGBM
"""

from dataclasses import dataclass, field

import numpy as np
import cv2
import skimage

import stereosgbm


@dataclass
class DisparityCalculator:
    """
    cv2.StereoSGBM based disparity calculator

    SEE ALSO:
        cv::StereoSGBM Class Reference
        https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
    """

    window_size: int = 3
    min_disp: int = 0
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

    def predict_by_gray(self, grayL: np.ndarray, grayR: np.ndarray) -> np.ndarray:
        """
        return disparity by gray image pair
        """
        disparity = self.opencv_sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
        return disparity

    def predict_by_bgr(self, bgrL: np.ndarray, bgrR: np.ndarray) -> np.ndarray:
        """
        return disparity by BGR image pair
        """
        grayL = cv2.cvtColor(bgrL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(bgrR, cv2.COLOR_BGR2GRAY)
        return self.predict_by_gray(grayL, grayR)

    def predict(self, imageL: np.ndarray, imageR: np.ndarray) -> np.ndarray:
        """
        return disparity by image pair
        """
        assert imageL.shape == imageR.shape
        if len(imageL.shape) == 2:
            return self.predict_by_gray(imageL, imageR)
        else:
            return self.predict_by_bgr(imageL, imageR)


@dataclass
class EdgeBasedDisparityCalculator:
    window_size: int = 3
    min_disp: int = 0
    max_disp: int = 320
    sgbm_disparity_calculator: DisparityCalculator = field(default=None)

    def __post_init__(self):
        self.sgbm_disparity_calculator = DisparityCalculator(
            window_size=self.window_size, min_disp=self.min_disp, max_disp=self.max_disp
        )

    def predict(self, grayL: np.ndarray, grayR: np.ndarray) -> np.ndarray:
        left_edge = skimage.filters.sobel(grayL)
        right_edge = skimage.filters.sobel(grayR)
        max1 = np.max(left_edge.flatten())
        max2 = np.max(right_edge.flatten())
        maxv = max((max1, max2))
        left_edge_normalized = ((left_edge / maxv) * 255).astype(np.uint8)
        right_edge_normalized = ((right_edge / maxv) * 255).astype(np.uint8)
        return self.sgbm_disparity_calculator.predict(left_edge_normalized, right_edge_normalized)
