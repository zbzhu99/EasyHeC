import os
import os.path as osp
import glob
import numpy as np
import imageio

from easyhec.utils.pointrend_api import pointrend_api
from easyhec.utils.point_drawer import PointDrawer

use_sam = False

if __name__ == "__main__":
    POINTREND_DIR = osp.join(
        osp.abspath("."), "third_party/detectron2/projects/PointRend"
    )
    pointrend_cfg_file = "configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_xarm7_finetune.yaml"
    pointrend_model_weight = "models/xarm7_finetune/model_0084999.pth"
    config_file = osp.join(POINTREND_DIR, pointrend_cfg_file)
    model_weight = osp.join(POINTREND_DIR, pointrend_model_weight)

    # image_path = osp.join("000000.png")
    image_dir = "./data/xarm6_offline/20241020_164336/left_back/color"
    output_mask_dir = "./data/xarm6_offline/20241020_164336/left_back/mask_autoseg"
    if not osp.exists(output_mask_dir):
        os.makedirs(output_mask_dir, exist_ok=True)

    for image_path in glob.glob(osp.join(image_dir, "*.png")):
        image_name = osp.basename(image_path)
        out_path = osp.join(output_mask_dir, image_name)

        if use_sam:
            rgb = imageio.imread_v2(image_path)[..., :3]
            point_drawer = PointDrawer(
                screen_scale=1.75,
                sam_checkpoint="third_party/segment_anything/sam_vit_h_4b8939.pth",
            )
            _, _, binary_mask = point_drawer.run(rgb)
            pred_binary_mask = binary_mask.astype(np.uint8)
        else:
            pred_binary_mask = pointrend_api(config_file, model_weight, image_path)

        imageio.imsave(out_path, (pred_binary_mask * 255)[:, :, None].repeat(3, axis=-1))
