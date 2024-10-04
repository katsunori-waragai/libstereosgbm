import cv2
import numpy as np

import stereosgbm


def test_sgbm_class():
    bgrL = cv2.imread("../test/test-imgs/left/left_motorcycle.png")
    bgrR = cv2.imread("../test/test-imgs/right/right_motorcycle.png")

    window_size = 3
    min_disp = 0
    max_disp = 320

    disparity_caluculator = stereosgbm.DisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )
    disparity = disparity_caluculator.calc_by_bgr(bgrL, bgrR)

    assert disparity.shape[:2] == bgrL.shape[:2]
    assert len(disparity.shape) == 2
    assert disparity.dtype in (np.float32, np.float64)
