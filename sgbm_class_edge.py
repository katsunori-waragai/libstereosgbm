from pathlib import Path
import glob

import cv2
import matplotlib.pyplot as plt
import skimage
import numpy as np


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

    left_imgs = sorted(glob.glob(args.left_imgs, recursive=True))
    right_imgs = sorted(glob.glob(args.right_imgs, recursive=True))

    edge_disparity_calculator = stereosgbm.EdgeBasedDisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )
    left_imgs = sorted(glob.glob(args.left_imgs, recursive=True))
    right_imgs = sorted(glob.glob(args.right_imgs, recursive=True))

    for left_name, right_name in zip(left_imgs, right_imgs):
        left_image = cv2.imread(left_name, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_name, cv2.IMREAD_GRAYSCALE)
        edge_disparity = edge_disparity_calculator.predict(left_image, right_image)
        left_name = Path(left_name)
        stem = left_name.stem.replace("left_", "")
        oname = outdir / f"edge_disparity_{stem}.npy"
        print(f"saving disparity as {oname}")
        np.save(str(oname), edge_disparity)

        plt.imshow(edge_disparity)
        plt.show()
