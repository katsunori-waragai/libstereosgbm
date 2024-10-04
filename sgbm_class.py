import glob
from pathlib import Path

import numpy as np
import cv2

import stereosgbm

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    window_size = 3
    min_disp = 0
    max_disp = 320

    disparity_caluculator = stereosgbm.DisparityCalculator(window_size=window_size, min_disp=min_disp, max_disp=max_disp)
    outdir = Path("test/test-imgs") / "disparity"
    outdir.mkdir(exist_ok=True)
    left_imgs = sorted(glob.glob("test/test-imgs/left/*.png", recursive=True))
    right_imgs = sorted(glob.glob("test/test-imgs/right/*.png", recursive=True))

    for left_name, right_name in zip(left_imgs, right_imgs):
        bgrL = cv2.imread(left_name)
        bgrR = cv2.imread(right_name)

        disparity = disparity_caluculator.calc_by_bgr(bgrL, bgrR)
        left_name = Path(left_name)
        oname = outdir / left_name.name.replace("left_", "disparity_").replace(".png", ".npy")
        print(f"saving disparity as {oname}")
        np.save(str(oname), disparity)

        plt.imshow(disparity, "gray")
        plt.show()
