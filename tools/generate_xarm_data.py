import argparse
import os
import cv2
import mplib
import numpy as np
import time
import math
from pathlib import Path
import pyrealsense2 as rs

SERIAL_NUMBERS = ["317622070866", "317622075882", "f1422097", "215222072518"]
CAMPOS_TO_SERIAL_NUMBERS = {
    "left_front": "317622070866",
    "right_front": "317622075882",
    "left_back": "f1422097",
    "right_back": "215222072518",
}
SERIAL_NUMBERS_TO_CAMPOS = {v: k for k, v in CAMPOS_TO_SERIAL_NUMBERS.items()}


class MultiRealSenseCamera:
    def __init__(self):
        super().__init__()
        # set initial pipelines and configs
        (
            self.serial_numbers,
            self.product_lines,
            self.device_idxs,
        ) = self.get_serial_numbers()
        assert all(
            sn in SERIAL_NUMBERS for sn in self.serial_numbers
        ), f"All serial numbers must be in {SERIAL_NUMBERS}"
        self.total_cam_num = len(self.serial_numbers)
        self.pipelines = [None] * self.total_cam_num
        self.configs = [None] * self.total_cam_num

        # set fps
        self.fps = 30

        # set pipelines and configs
        for i, serial_number, product_line in zip(
            range(0, self.total_cam_num), self.serial_numbers, self.product_lines
        ):
            self.pipelines[i] = rs.pipeline()
            self.configs[i] = rs.config()
            self.configs[i].enable_device(serial_number)
            if product_line == "L500":
                self.configs[i].enable_stream(
                    rs.stream.depth,
                    # 768,
                    1024,
                    768,
                    rs.format.z16,
                    self.fps,
                )
                self.configs[i].enable_stream(
                    rs.stream.color,
                    1920,
                    1080,
                    rs.format.rgb8,
                    self.fps,
                )
            else:
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

            if self.product_lines[i] != "L500":
                if master_or_slave == 1:
                    depth_sensor.set_option(rs.option.inter_cam_sync_mode, 1)  # Master
                    master_or_slave = 2
                else:
                    depth_sensor.set_option(rs.option.inter_cam_sync_mode, 2)  # Slave
            else:
                print(
                    f"Camera {i} is an L500 series and does not support inter-camera synchronization."
                )

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
        product_lines = []
        device_idxs = []
        self.ctx = rs.context()
        if len(self.ctx.devices) > 0:
            for j, d in enumerate(self.ctx.devices):
                name = d.get_info(rs.camera_info.name)
                serial_number = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                print(f"Found device: {name} {serial_number}")
                serial_numbers.append(serial_number)
                product_lines.append(product_line)
                device_idxs.append(j)
        else:
            print("No Intel Device connected")
        return serial_numbers, product_lines, device_idxs

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


def save_colors(images, data_dir, frame_num, campos_list):
    tmp_colors, tmp_depths = images
    assert len(tmp_colors) == len(tmp_depths)
    for i, campos in enumerate(campos_list):
        images_dir = data_dir / f"{campos}"
        os.makedirs(images_dir / f"color", exist_ok=True)
        cv2.imwrite(
            str(images_dir / f"color" / f"00000{frame_num}.png"),
            cv2.cvtColor(tmp_colors[i], cv2.COLOR_RGB2BGR),
        )


def save_qpos(images, qpos, data_dir, frame_num, campos_list):
    tmp_colors, tmp_depths = images
    assert len(tmp_colors) == len(tmp_depths)
    for i, campos in enumerate(campos_list):
        qpos_dir = data_dir / f"{campos}"
        os.makedirs(qpos_dir / f"qpos", exist_ok=True)
        qpos_filename = qpos_dir / f"qpos" / f"00000{frame_num}.txt"
        np.savetxt(qpos_filename, qpos)


def save_k(multi_camera, base_dir, campos_list):
    intr_colors = multi_camera.get_intrinsic_color()
    for i, campos in enumerate(campos_list):
        data_dir = base_dir / f"{campos}"
        os.makedirs(data_dir, exist_ok=True)
        intrinsic = np.array(
            [
                [intr_colors[i]["fx"], 0, intr_colors[i]["ppx"]],
                [0, intr_colors[i]["fy"], intr_colors[i]["ppy"]],
                [0, 0, 1],
            ]
        )
        qpos_filename = data_dir / "K.txt"
        np.savetxt(qpos_filename, intrinsic)


INITIAL_ARM_JOINT_POS = [0.8, -36.7, -22.6, -0.1, 58.4, 0.3, 0]
speed = 30

if __name__ == "__main__":
    print("Creating mplib planner")
    urdf_path = "./assets/xarm6_with_gripper.urdf"
    srdf_path = "./assets/xarm6_with_gripper.srdf"
    planner = mplib.Planner(
        urdf=urdf_path,
        srdf=srdf_path,
        move_group="link_tcp",
        joint_vel_limits=np.ones(6) * 0.9,
        joint_acc_limits=np.ones(6) * 0.9,
    )

    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"./data/xarm6_offline/{current_datetime}")
    os.makedirs(base_dir, exist_ok=True)
    multi_camera = MultiRealSenseCamera()
    speed = math.radians(30)
    time.sleep(3)

    from xarm.wrapper import XArmAPI

    campos_list = [SERIAL_NUMBERS_TO_CAMPOS[sn] for sn in multi_camera.serial_numbers]
    save_k(multi_camera, base_dir, campos_list)

    print("Starting robot")
    ip = "192.168.1.212"
    arm = XArmAPI(ip, is_radian=True)
    arm.motion_enable(enable=True)
    arm.clean_error()
    arm.set_gripper_enable(True)
    arm.set_gripper_position(800, wait=True)
    arm.set_gripper_enable(False)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_servo_angle(
        angle=INITIAL_ARM_JOINT_POS, speed=30, is_radian=False, wait=True
    )

    num_shots = 18

    calibration_dir = Path("./xarm6_calib_qpos")
    for i in range(num_shots):
        qpos_file = calibration_dir / f"00000{i}.txt"

        if qpos_file.exists():
            
            # qpos = np.loadtxt(qpos_file)
            # arm.set_servo_angle(angle=qpos.tolist()[:-2], is_radian=True, wait=True, speed=speed)

            qpos = np.loadtxt(qpos_file)
            current_qpos = np.concatenate(
                [np.array(arm.get_servo_angle(is_radian=True)[1])[:-1], qpos[-2:]]
            )
            result = planner.plan_qpos_to_qpos(
                [qpos],
                current_qpos,
                time_step=0.1,
            )
            planned_qpos_traj = np.array(result["position"])
            planned_qpos_traj = np.concatenate(
                [planned_qpos_traj, np.zeros((planned_qpos_traj.shape[0], 1))],
                axis=-1,
            )
            for qpos in planned_qpos_traj:
                arm.set_servo_angle(angle=qpos.tolist(), is_radian=True, wait=True, speed=speed)
                time.sleep(0.1)

            time.sleep(3)  # wait for the arm to be stable
        else:
            print(f"canot find the qpos file: {qpos_file}")
            continue
        _, gripper_pos = arm.get_gripper_position()
        save_colors(multi_camera.undistorted_rgbd(), base_dir, i, campos_list)
        save_qpos(multi_camera.undistorted_rgbd(), qpos, base_dir, i, campos_list)

    arm.set_mode(0)
    arm.set_state(0)

    # get by bash script
    print("--------------------------------")
    print("Generated data path:")
    print(current_datetime)
