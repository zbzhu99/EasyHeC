import cv2
import os.path as osp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from easyhec.utils import render_api, utils_3d, plt_utils
from easyhec.utils.vis3d_ext import Vis3D

data_dir = "data/xarm6_offline/20241022_171204/right_front"
qpos = np.loadtxt(osp.join(data_dir, "qpos/000001.txt"))
# qpos = np.zeros_like(qpos)

K = np.loadtxt(osp.join(data_dir, "K.txt"))
color = imageio.imread_v2(osp.join(data_dir, "color/000001.png"))
H, W, _ = color.shape
vis3d = Vis3D(
    xyz_pattern=("x", "y", "z"),
    out_folder="dbg",
    sequence="test_franka_init",
)

qpos = np.concatenate([qpos, np.zeros(1)])
vis3d.add_xarm(qpos)
#  you may need to tune these numbers manually.
Tb_b2c = utils_3d.calc_pose_from_lookat(np.radians([60]), np.radians([-20]), 1, 1.2)[0]
vis3d.add_camera_trajectory(Tb_b2c[None])
# you may need to tune these numbers manually.
Tb_b2c[0, 3] += 0.4
Tb_b2c[1, 3] += 0.4
# Tb_b2c[2, 3] += 0.2
Tc_c2b = np.linalg.inv(Tb_b2c)
# Tc_c2b = np.array([[ 0.8777, -0.4787,  0.0212, -0.2526],
#         [-0.1906, -0.3894, -0.9011,  0.2816],
#         [ 0.4396,  0.7869, -0.4330,  0.7745],
#         [ 0.0000,  0.0000,  0.0000,  1.0000]])
Tc_c2b = np.array([[-0.9271,  0.3749, -0.0061,  0.3839],
        [ 0.1821,  0.4359, -0.8814,  0.0084],
        [-0.3278, -0.8182, -0.4724,  1.1687],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])



mask = render_api.nvdiffrast_render_xarm_api(
    "assets/xarm6_with_gripper.urdf", Tc_c2b, qpos.tolist(), H, W, K
)
mask_image = (mask * 255).astype(np.uint8)
imageio.imwrite("xarm_mask.png", mask_image)

# scale_x = 640 / W
# scale_y = 480 / H
# K_new = np.array([
#     [K[0, 0] * scale_x, 0, K[0, 2] * scale_x],
#     [0, K[1, 1] * scale_y, K[1, 2] * scale_y],
#     [0, 0, 1],
# ])

# mask_1 = render_api.nvdiffrast_render_xarm_api(
#     "assets/xarm6_with_gripper.urdf", Tc_c2b, qpos.tolist(), 480, 640, K_new
# )
# mask_image_1 = (mask_1 * 255).astype(np.uint8)
# imageio.imwrite("xarm_mask_1.png", mask_image_1)

# mask_image_resized = cv2.resize(mask_image, (640, 480), interpolation=cv2.INTER_LINEAR)
# imageio.imwrite("xarm_mask_resized.png", mask_image_resized)
# diff_image = np.abs(mask_image_resized.astype(int) - mask_image_1.astype(int)).astype(np.uint8)
# imageio.imwrite("xarm_mask_diff.png", diff_image)
# print(diff_image.sum())

# tune until the mask has reasonable overlap with the RGB image.
plt.imshow(plt_utils.vis_mask(color, mask))
plt.savefig("xarm_init.png")
# plt.show()

print(np.array2string(Tc_c2b, separator=", "))
print("put this into the yaml file: model.rbsolver.init_Tc_c2b")
