import glob
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

import stereosgbm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_numpy", default=True, help="save output as numpy arrays")

    parser.add_argument(
        "-l", "--left_imgs", help="path to all first (left) frames", default="test/test-imgs/left/*.png"
    )
    parser.add_argument(
        "-r", "--right_imgs", help="path to all second (right) frames", default="test/test-imgs/right/*.png"
    )

    parser.add_argument("--output_directory", help="directory to save output", default="./demo-output/")
    parser.add_argument("--max_disp", type=int, default=320, help="max disp of geometry encoding volume")

    args = parser.parse_args()

    from matplotlib import pyplot as plt

    window_size = 3
    min_disp = 0
    max_disp = args.max_disp
    outdir = Path(args.output_directory)
    outdir.mkdir(exist_ok=True)

    disparity_caluculator = stereosgbm.DisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )
    left_imgs = sorted(glob.glob(args.left_imgs, recursive=True))
    right_imgs = sorted(glob.glob(args.right_imgs, recursive=True))

    for left_name, right_name in tqdm(list(zip(left_imgs, right_imgs))):
        bgrL = cv2.imread(left_name)
        bgrR = cv2.imread(right_name)

        disparity = disparity_caluculator.predict_by_bgr(bgrL, bgrR)
        left_name = Path(left_name)
        stem = left_name.stem.replace("left_", "")
        oname = outdir / f"disparity_{stem}.npy"
        print(f"saving disparity as {oname}")
        np.save(str(oname), disparity)

        plt.imshow(disparity, "gray")
        plt.show()
