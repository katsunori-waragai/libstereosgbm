import cv2
import numpy as np

import stereosgbm


def test_sgbm_bgr():
    bgrL = cv2.imread("../test/test-imgs/left/left_motorcycle.png")
    bgrR = cv2.imread("../test/test-imgs/right/right_motorcycle.png")

    window_size = 3
    min_disp = 0
    max_disp = 320

    disparity_caluculator = stereosgbm.DisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )
    disparity = disparity_caluculator.predict_by_bgr(bgrL, bgrR)

    assert disparity.shape[:2] == bgrL.shape[:2]
    assert len(disparity.shape) == 2
    assert disparity.dtype in (np.float32, np.float64)

    disparity = disparity_caluculator.predict(bgrL, bgrR)

    assert disparity.shape[:2] == bgrL.shape[:2]
    assert len(disparity.shape) == 2
    assert disparity.dtype in (np.float32, np.float64)


def test_sgbm_gray():
    grayL = cv2.imread("../test/test-imgs/left/left_motorcycle.png", cv2.IMREAD_GRAYSCALE)
    grayR = cv2.imread("../test/test-imgs/right/right_motorcycle.png", cv2.IMREAD_GRAYSCALE)

    window_size = 3
    min_disp = 0
    max_disp = 320

    disparity_caluculator = stereosgbm.DisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )
    disparity = disparity_caluculator.predict_by_gray(grayL, grayR)

    assert disparity.shape[:2] == grayL.shape[:2]
    assert len(disparity.shape) == 2
    assert disparity.dtype in (np.float32, np.float64)

    disparity = disparity_caluculator.predict(grayL, grayR)

    assert disparity.shape[:2] == grayL.shape[:2]
    assert len(disparity.shape) == 2
    assert disparity.dtype in (np.float32, np.float64)
