import numpy as np
import cv2

import stereosgbm

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    print("loading images...")
    bgrL = cv2.imread("../test/test-imgs/left/left_motorcycle.png")
    bgrR = cv2.imread("../test/test-imgs/right/right_motorcycle.png")

    window_size = 3
    min_disp = 0
    max_disp = 320

    disparity_caluculator = stereosgbm.DisparityCalculator(window_size=window_size, min_disp=min_disp, max_disp=max_disp)
    disparity = disparity_caluculator.calc_by_brg(bgrL, bgrR)

    print("saving disparity as disparity_image_sgbm.txt")

    np.savetxt(
        "data/disparity_image_sgbm.txt",
        disparity,
        fmt="%3.2f",
        delimiter=" ",
        newline="\n",
    )

    plt.imshow(disparity, "gray")
    plt.show()
