import cv2
import numpy as np

import stereosgbm


def test_sgbm_edge_gray():
    grayL = cv2.imread("../test/test-imgs/left/left_motorcycle.png", cv2.IMREAD_GRAYSCALE)
    grayR = cv2.imread("../test/test-imgs/right/right_motorcycle.png", cv2.IMREAD_GRAYSCALE)

    window_size = 3
    min_disp = 0
    max_disp = 320

    disparity_calculator = stereosgbm.EdgeBasedDisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )
    disparity = disparity_calculator.predict_by_gray(grayL, grayR)

    assert disparity.shape[:2] == grayL.shape[:2]
    assert len(disparity.shape) == 2
    assert disparity.dtype in (np.float32, np.float64)

    disparity = disparity_calculator.predict(grayL, grayR)

    assert disparity.shape[:2] == grayL.shape[:2]
    assert len(disparity.shape) == 2
    assert disparity.dtype in (np.float32, np.float64)
