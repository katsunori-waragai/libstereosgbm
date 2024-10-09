"""
Sample script to view depth images
- ZED-SDK depth
- Depth-Anything depth(native)
This is developing code for depth-anything with zed sdk.
"""

import pyzed.sl as sl
import argparse

import cv2
import numpy as np

import stereosgbm

MAX_ABS_DEPTH, MIN_ABS_DEPTH = 0.0, 2.0  # [m]

def finitemax(depth: np.ndarray):
    return np.nanmax(depth[np.isfinite(depth)])


def finitemin(depth: np.ndarray):
    return np.nanmin(depth[np.isfinite(depth)])


def depth_as_colorimage(depth_raw: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """
    apply color mapping with vmin, vmax
    """
    if vmin is None:
        vmin = finitemin(depth_raw)
    if vmax is None:
        vmax = finitemax(depth_raw)
    depth_raw = (depth_raw - vmin) / (vmax - vmin) * 255.0
    depth_raw = depth_raw.astype(np.uint8)
    return cv2.applyColorMap(depth_raw, cv2.COLORMAP_INFERNO)



def parse_args(init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
            and len(ip_str.split(":")) == 2
        ):
            init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif ip_str.replace(":", "").replace(".", "").isdigit() and len(ip_str.split(".")) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Unvalid IP format. Using live stream")
    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")


def resize_image(image: np.ndarray, rate: float) -> np.ndarray:
    H, W = image.shape[:2]
    return cv2.resize(image, (int(W * rate), int(H * rate)))


def as_matrix(chw_array):
    H_, W_ = chw_array.shape[-2:]
    return np.reshape(chw_array, (H_, W_))


def main(opt):
    calc_disparity = True
    video_num = 0

    zed = sl.Camera()
    init_params = sl.InitParameters()
    parse_args(init_params)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD1080

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        exit(1)

    window_size = 3
    min_disp = 0
    max_disp = 320

    disparity_calculator = stereosgbm.DisparityCalculator(
        window_size=window_size, min_disp=min_disp, max_disp=max_disp
    )

    left_image = sl.Mat()
    right_image = sl.Mat()

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    runtime_parameters.confidence_threshold = opt.confidence_threshold
    print(f"### {runtime_parameters.confidence_threshold=}")

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT, sl.MEM.CPU)
            cv_left_image = left_image.get_data()
            assert cv_left_image.shape[2] == 4  # ZED SDK dependent.
            cv_left_image = cv_left_image[:, :, :3].copy()
            cv_left_image = np.ascontiguousarray(cv_left_image)

            cv_right_image = right_image.get_data()
            cv_right_image = cv_right_image[:, :, :3].copy()
            cv_right_image = np.ascontiguousarray(cv_right_image)
        else:
            continue
        assert cv_left_image.shape[2] == 3
        assert cv_left_image.dtype == np.uint8
        disparity_raw = disparity_calculator.calc_by_bgr(cv_left_image, cv_right_image)

        depth_any = depth_as_colorimage(disparity_raw)
        results = np.concatenate((cv_left_image, depth_any), axis=1)
        H_, W_ = results.shape[:2]
        results = cv2.resize(results, (W_ // 2, H_ // 2))
        cv2.imshow("Depth", results)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="depth-anything(native) with zed camera")
    parser.add_argument(
        "--input_svo_file",
        type=str,
        help="Path to an .svo file, if you want to replay it",
        default="",
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        help="IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup",
        default="",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA",
        default="",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="depth confidence_threshold(0 ~ 100)",
        default=100,
    )
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)