import cv2
import numpy as np
# from matplotlib import pyplot as plt

import stereosgbm

def test_sgbm_class():

    print("loading images...")
    bgrL = cv2.imread("../test/test-imgs/left/left_motorcycle.png")
    bgrR = cv2.imread("../test/test-imgs/right/right_motorcycle.png")

    window_size = 3
    min_disp = 0
    max_disp = 320

    disparity_caluculator = stereosgbm.DisparityCalculator(window_size=window_size, min_disp=min_disp, max_disp=max_disp)
    disparity = disparity_caluculator.calc_by_bgr(bgrL, bgrR)

    assert disparity.shape[:2] == bgrL.shape[:2]
    assert len(disparity.shape) == 2
    assert disparity.dtype in (np.float32, np.float64)

    print("saving disparity as disparity_image_sgbm.txt")

    np.savetxt(
        "data/disparity_image_sgbm.txt",
        disparity,
        fmt="%3.2f",
        delimiter=" ",
        newline="\n",
    )

    # plt.imshow(disparity, "gray")
    # plt.show()

if __name__ == "__main__":
    test_sgbm_class()
