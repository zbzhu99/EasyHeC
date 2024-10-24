#!/bin/bash
DATA_PATH=$(python tools/generate_xarm_data.py | tail -n 1)
# DATA_PATH="20241022_171204"

echo "Running prompt_drawer.py for all cameras..."
CAMERAS=("left_front" "left_back" "right_front" "right_back")
for CAM in "${CAMERAS[@]}"
do
    python easyhec/utils/prompt_drawer.py \
    --img_paths data/xarm6_offline/${DATA_PATH}/${CAM}/color/*.png \
    --output_dir data/xarm6_offline/${DATA_PATH}/${CAM}/mask
done

echo "Running run_easyhec.py for all cameras..."
for CAM in "${CAMERAS[@]}"
do
    CONFIG="./configs/xarm6/${CAM}.yaml"
    sed -i "s|train:.*|train: (\"xarm6_real/xarm6_offline/${DATA_PATH}/${CAM}\",)|g" "$CONFIG"
    python tools/run_easyhec.py -c "$CONFIG"
done
