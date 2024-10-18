import argparse
import os
import cv2
import random
import re
import numpy as np
import time
import math
from pathlib import Path
import pyrealsense2 as rs


class MultiRealSenseCamera:
    def __init__(self):
        super().__init__()
        # set initial pipelines and configs
        self.serial_numbers, self.device_idxs = self.get_serial_numbers()
        self.total_cam_num = len(self.serial_numbers)
        self.pipelines = [None] * self.total_cam_num
        self.configs = [None] * self.total_cam_num

        # set resolutions and fps
        self.fps = 30

        # set pipelines and configs
        for i, serial_number in zip(range(0, self.total_cam_num), self.serial_numbers):
            self.pipelines[i] = rs.pipeline()
            self.configs[i] = rs.config()
            self.configs[i].enable_device(serial_number)
            self.configs[i].enable_stream(
                rs.stream.depth,
                640,
                480,
                rs.format.z16,
                self.fps,
            )
            self.configs[i].enable_stream(
                rs.stream.color,
                640,
                480,
                rs.format.rgb8,
                self.fps,
            )

        # Start streaming
        self.sensors = [None] * self.total_cam_num
        self.cfgs = [None] * self.total_cam_num

        # set master & slave
        master_or_slave = 1
        for i in range(0, self.total_cam_num):
            depth_sensor = self.ctx.devices[self.device_idxs[i]].first_depth_sensor()
            color_sensor = self.ctx.devices[self.device_idxs[i]].first_color_sensor()
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
            if i == 0:
                depth_sensor.set_option(rs.option.inter_cam_sync_mode, master_or_slave)
                master_or_slave = 2
            else:
                depth_sensor.set_option(rs.option.inter_cam_sync_mode, master_or_slave)

            self.cfgs[i] = self.pipelines[i].start(self.configs[i])
            depth_scale = (
                1 / self.cfgs[i].get_device().first_depth_sensor().get_depth_scale()
            )

            # sensor = self.pipelines[i].get_active_profile().get_device().query_sensors()[1]
            # sensor.set_option(rs.option.exposure, 330)

    def undistorted_rgbd(self):
        depth_frame = [None] * self.total_cam_num
        color_frame = [None] * self.total_cam_num
        depth_image = [None] * self.total_cam_num
        color_image = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            frame = self.pipelines[i].wait_for_frames()
            align_frame = rs.align(rs.stream.color).process(frame)
            depth_frame[i] = align_frame.get_depth_frame()
            color_frame[i] = align_frame.get_color_frame()
            depth_image[i] = np.asanyarray(depth_frame[i].get_data())
            color_image[i] = np.asanyarray(color_frame[i].get_data())
        return color_image, depth_image

    def undistorted_rgb(self):
        color_frame = [None] * self.total_cam_num
        color_image = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            frame = self.pipelines[i].wait_for_frames()
            align_frame = rs.align(rs.stream.color).process(frame)
            color_frame[i] = align_frame.get_color_frame()
            color_image[i] = np.asanyarray(color_frame[i].get_data())
        return color_image

    def get_serial_numbers(self):
        serial_numbers = []
        device_idxs = []
        self.ctx = rs.context()
        if len(self.ctx.devices) > 0:
            for j, d in enumerate(self.ctx.devices):
                name = d.get_info(rs.camera_info.name)
                serial_number = d.get_info(rs.camera_info.serial_number)
                print(f"Found device: {name} {serial_number}")
                serial_numbers.append(serial_number)
                device_idxs.append(j)
        else:
            print("No Intel Device connected")
        return serial_numbers, device_idxs

    def get_intrinsic_color(self):
        intrinsic = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            profile = self.cfgs[i].get_stream(rs.stream.color).as_video_stream_profile()
            intr = profile.get_intrinsics()
            intrinsic[i] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
            }
        return intrinsic

    def get_intrinsic_depth(self):
        intrinsic = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            profile = self.cfgs[i].get_stream(rs.stream.depth).as_video_stream_profile()
            intr = profile.get_intrinsics()
            intrinsic[i] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
            }
        return intrinsic


def save_colors(images, images_dir, frame_num):
    tmp_colors, tmp_depths = images
    now_time = frame_num
    assert len(tmp_colors) == len(tmp_depths)
    # print(len(tmp_colors))
    os.makedirs(images_dir / f"color", exist_ok=True)
    cv2.imwrite(
        str(images_dir / f"color" / f"00000{now_time}.png"),
        cv2.cvtColor(tmp_colors[0], cv2.COLOR_RGB2BGR),
    )


def save_qpos(qpos, qpos_dir, frame_num):
    os.makedirs(qpos_dir / f"qpos", exist_ok=True)
    qpos_filename = qpos_dir / f"qpos" / f"00000{frame_num}.txt"
    np.savetxt(qpos_filename, qpos)


def save_k(multi_camera, base_dir):
    intr_colors = multi_camera.get_intrinsic_color()
    fx = intr_colors[0]["fx"]
    print(fx)
    fy = intr_colors[0]["fy"]
    ppx = intr_colors[0]["ppx"]
    ppy = intr_colors[0]["ppy"]
    intrinsic = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    qpos_filename = base_dir / f"K.txt"
    np.savetxt(qpos_filename, intrinsic)


base_dir = Path("data/xarm6_offline/example_right_back_test")
os.makedirs(base_dir, exist_ok=True)
multi_camera = MultiRealSenseCamera()
speed = math.radians(30)
time.sleep(2)

from xarm.wrapper import XArmAPI

save_k(multi_camera, base_dir)

print("Starting robot")
ip = "192.168.1.212"
arm = XArmAPI(ip, is_radian=True)
arm.motion_enable(enable=True)
arm.clean_error()
arm.set_gripper_enable(True)
arm.set_mode(2)
arm.set_state(0)
# 自动化拍摄和保存数据
num_shots = 5  # 拍摄次数

for i in range(num_shots):
    code, (qpos, qvel, qeff) = arm.get_joint_states()
    qpos.append(0)
    _, gripper_pos = arm.get_gripper_position()
    save_colors(multi_camera.undistorted_rgbd(), base_dir, i)
    save_qpos(qpos, base_dir, i)
    input("Press Enter to continue...")

arm.set_mode(0)
arm.set_state(0)
