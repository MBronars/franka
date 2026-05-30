#!/usr/bin/env python
"""
Visual sanity check for the camera<->robot extrinsic.

What it does
------------
- Opens the RealSense color stream via pyrealsense2 (no ROS needed for this
  side; matches the Dex4D / pyrealsense2 path you'll be using at run time).
- Reads the live end-effector pose from frankapy.
- Transforms the EE position from `panda_link0` -> camera optical frame using
  the loaded T_base_camera, then projects to pixel coords with the RealSense
  intrinsics.
- Draws a crosshair at the projected pixel + a small RGB triad showing the
  EE orientation.

If the calibration is good, the crosshair sticks to the actual EE in the
image as you move the robot.  A constant offset of >~1 cm in the image plane
means the extrinsic is off.  Pure observation -- this script does NOT command
the robot, so you can run keyboard_teleop or guide_mode in another terminal
to move the EE around while this watches.

Inputs
------
--yaml   easy_handeye's saved file
         (default ~/.ros/easy_handeye/franka_realsense_eob_eye_on_base.yaml)
--npz    alternative: a numpy file with key "T_base_camera" (4x4)

Loads ONE of those; --yaml wins if both passed.

Usage
-----
  python check_calibration.py
  python check_calibration.py --yaml ~/.ros/easy_handeye/<name>.yaml
  python check_calibration.py --npz T_base_camera.npz
"""

import argparse
import os

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from frankapy import FrankaArm


DEFAULT_YAML = os.path.expanduser(
    '~/.ros/easy_handeye/franka_realsense_eob_eye_on_base.yaml')

TRIAD_LEN = 0.05    # m, length of each axis arrow drawn at the EE


# ----------------------------------------------------------------------------
# Loading the calibration.
# ----------------------------------------------------------------------------
def quat_wxyz_to_R(qw, qx, qy, qz):
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def load_T_base_camera(yaml_path, npz_path):
    if yaml_path is not None:
        with open(yaml_path) as f:
            d = yaml.safe_load(f)
        tf = d['transformation']
        R = quat_wxyz_to_R(tf['qw'], tf['qx'], tf['qy'], tf['qz'])
        t = np.array([tf['x'], tf['y'], tf['z']])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    if npz_path is not None:
        data = np.load(npz_path)
        key = 'T_base_camera' if 'T_base_camera' in data else list(data.keys())[0]
        T = data[key]
        assert T.shape == (4, 4), f'{key} must be 4x4, got {T.shape}'
        return T
    raise SystemExit('Pass --yaml or --npz')


# ----------------------------------------------------------------------------
# RealSense pipeline + intrinsics.
# ----------------------------------------------------------------------------
def start_realsense(width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1],
    ])
    dist = np.array(intr.coeffs, dtype=np.float64)
    return pipeline, K, dist


def project(P_cam, K, dist):
    """3xN points in camera frame -> Nx2 pixel coords; returns None for points behind camera."""
    if P_cam[2] <= 0:
        return None
    pts = P_cam.reshape(1, 1, 3).astype(np.float64)
    pix, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
    return pix[0, 0]


def draw_triad(img, T_cam_ee, K, dist):
    """Draw a small XYZ triad at the EE in the image (red=x, green=y, blue=z)."""
    origin_cam = T_cam_ee[:3, 3]
    R_cam_ee = T_cam_ee[:3, :3]

    o_pix = project(origin_cam, K, dist)
    if o_pix is None:
        return
    o = (int(o_pix[0]), int(o_pix[1]))

    axes_colors = [
        (R_cam_ee[:, 0] * TRIAD_LEN, (0, 0, 255)),   # +x red   (BGR)
        (R_cam_ee[:, 1] * TRIAD_LEN, (0, 255, 0)),   # +y green
        (R_cam_ee[:, 2] * TRIAD_LEN, (255, 0, 0)),   # +z blue
    ]
    for axis_vec, color in axes_colors:
        end_pix = project(origin_cam + axis_vec, K, dist)
        if end_pix is None:
            continue
        e = (int(end_pix[0]), int(end_pix[1]))
        cv2.line(img, o, e, color, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Main loop.
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yaml', default=DEFAULT_YAML if os.path.exists(DEFAULT_YAML) else None)
    ap.add_argument('--npz', default=None)
    args = ap.parse_args()

    T_base_camera = load_T_base_camera(args.yaml, args.npz)
    T_camera_base = np.linalg.inv(T_base_camera)

    cam_pos_in_base = T_base_camera[:3, 3]
    print('Loaded T_base_camera:')
    print(T_base_camera)
    print('Camera position in robot base frame: '
          f'x={cam_pos_in_base[0]:+.3f}  y={cam_pos_in_base[1]:+.3f}  '
          f'z={cam_pos_in_base[2]:+.3f}   (|.|={np.linalg.norm(cam_pos_in_base):.3f} m)')
    if not (0.2 < np.linalg.norm(cam_pos_in_base) < 3.0):
        print('  ! sanity check WARN: camera-to-base distance is unusual.')

    print('Connecting to the robot (read-only)...')
    fa = FrankaArm()
    print('Starting RealSense...')
    pipeline, K, dist = start_realsense()

    cv2.namedWindow('calibration check', cv2.WINDOW_AUTOSIZE)
    print('\nMove the EE (guide mode or keyboard_teleop in another terminal).')
    print('The green crosshair should stick to the EE; the triad should align')
    print('with the gripper axes.  Press q to quit.\n')

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data()).copy()

            ee_pose = fa.get_pose()
            P_base = np.asarray(ee_pose.translation, dtype=np.float64)
            R_base_ee = np.asarray(ee_pose.rotation, dtype=np.float64)

            # EE pose expressed in camera frame.
            T_base_ee = np.eye(4)
            T_base_ee[:3, :3] = R_base_ee
            T_base_ee[:3, 3] = P_base
            T_cam_ee = T_camera_base @ T_base_ee
            P_cam = T_cam_ee[:3, 3]

            pix = project(P_cam, K, dist)
            if pix is not None:
                u, v = int(pix[0]), int(pix[1])
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.drawMarker(img, (u, v), (0, 255, 0),
                                   cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)
                    cv2.circle(img, (u, v), 10, (0, 255, 0), 1, cv2.LINE_AA)
                    draw_triad(img, T_cam_ee, K, dist)
                hud_msg = f'EE in base : ({P_base[0]:+.3f}, {P_base[1]:+.3f}, {P_base[2]:+.3f}) m'
                pix_msg = f'projected px: ({u}, {v})   depth_in_cam={P_cam[2]:+.3f} m'
            else:
                hud_msg = 'EE projects BEHIND the camera -- extrinsic may be wrong or arm out of view'
                pix_msg = ''

            for i, line in enumerate([hud_msg, pix_msg, 'q = quit']):
                if line:
                    cv2.putText(img, line, (10, 25 + 22 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, line, (10, 25 + 22 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('calibration check', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
