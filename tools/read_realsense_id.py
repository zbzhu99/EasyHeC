import cv2
import numpy as np
import pyrealsense2 as rs


def get_realsense_serial_numbers():
    # Create a context object to manage RealSense devices
    ctx = rs.context()

    # Initialize an empty list to store serial numbers and device info
    device_info = []

    # Iterate through all connected devices
    for dev in ctx.devices:
        # Get the serial number and product line of each device
        serial_number = dev.get_info(rs.camera_info.serial_number)
        product_line = dev.get_info(rs.camera_info.product_line)
        device_info.append((serial_number, product_line))

    return device_info


def capture_and_save_images(serial_number):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    if product_line == "L500":
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Save the color image
        filename = f"realsense_{serial_number}.png"
        cv2.imwrite(filename, color_image)
        print(f"Image saved as {filename}")

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    # Get the serial numbers of all connected RealSense cameras
    serials = get_realsense_serial_numbers()

    # Print the results and capture images
    if serials:
        print("Connected RealSense camera serial numbers:")
        for i, (serial, product_line) in enumerate(serials, 1):
            print(f"Camera {i}: {serial} ({product_line})")
            capture_and_save_images(serial)
    else:
        print("No RealSense cameras detected.")
