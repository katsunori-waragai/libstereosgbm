"""
Script for capture as usb camera and estimate disparity.
"""

import argparse

import cv2
import numpy as np

import stereosgbm


def resize_image(frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    return cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="disparity tool for ZED2i camera as usb camera")
    parser.add_argument("--calc_disparity", action="store_true", help="calc disparity")
    parser.add_argument("--max_disp", type=int, default=320, help="max disp of geometry encoding volume")
    parser.add_argument("video_num", help="number in /dev/video")
    args = parser.parse_args()

    calc_disparity = args.calc_disparity
    video_num = int(args.video_num)

    if calc_disparity:
        window_size = 3
        min_disp = 0
        max_disp = args.max_disp

        disparity_calculator = stereosgbm.DisparityCalculator(
            window_size=window_size, min_disp=min_disp, max_disp=max_disp
        )
    cap = cv2.VideoCapture(video_num)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        _, frame = cap.read()
        frame = resize_image(frame)
        H, W = frame.shape[:2]
        half_W = W // 2
        left = frame[:, :half_W, :]
        right = frame[:, half_W:, :]

        cv2.imshow("left and right", frame)

        if calc_disparity:
            disparity = disparity_calculator.calc_by_bgr(left.copy(), right.copy())
            disp = np.round(disparity * 256).astype(np.uint16)
            colored = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.01), cv2.COLORMAP_JET)
            cv2.imshow("IGEV", colored)
        key = cv2.waitKey(100)
        if key == ord("q"):
            exit()
