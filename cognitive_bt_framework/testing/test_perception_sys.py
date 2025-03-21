import numpy as np
import cv2
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig
from cognitive_bt_framework.src.vision.realsense import Camera
import time
import torch


def main():
    # Initialize camera
    camera = Camera(width=640, height=480)
    if not camera.start():
        print("Failed to start camera")
        return

    try:
        # Wait for camera to warm up
        print("Warming up camera...")
        time.sleep(2)

        # Get camera matrix from RealSense intrinsics
        camera_matrix = np.array([
            [camera.intrinsics.fx, 0, camera.intrinsics.ppx],
            [0, camera.intrinsics.fy, camera.intrinsics.ppy],
            [0, 0, 1]
        ])

        # Initialize perception system
        fast_sam_config = FastSAMConfig(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        perception = PerceptionSystem(
            fast_sam_config=fast_sam_config,
            camera_matrix=camera_matrix,
            depth_scale=camera.depth_scale
        )

        # List of objects to detect
        target_objects = ["pen"]
        current_target_idx = 0

        while True:
            # Get frames from camera
            frames = camera.get_frames()
            if frames is None:
                print("Failed to get frames")
                continue

            color_image, depth_image = frames
            target_object = target_objects[current_target_idx]

            # Detect object
            print(f"\nLooking for: {target_object}")
            object_info = perception.detect_object(
                target_object=target_object,
                image=color_image,
                depth_image=depth_image,
                confidence_threshold=0.5
            )

            if object_info is not None:
                print(f"Found {target_object}!")
                print(f"Confidence: {object_info.confidence:.2f}")
                print(f"Bounding box: {object_info.bbox}")
                
                if object_info.pose is not None:
                    x, y, z, roll, pitch, yaw = object_info.pose
                    print(f"Position: ({x:.3f}, {y:.3f}, {z:.3f}) meters")
                    print(f"Orientation: ({np.rad2deg(roll):.1f}°, {np.rad2deg(pitch):.1f}°, {np.rad2deg(yaw):.1f}°)")

                # Visualize detection
                vis_image = perception.visualize_detection(
                    image=color_image,
                    object_info=object_info,
                    show_pose=True
                )
            else:
                print(f"No {target_object} found")
                vis_image = color_image.copy()

            # Show depth colormap alongside RGB
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Stack images horizontally
            display_image = np.hstack((vis_image, depth_colormap))
            
            # Add text overlay with current target
            cv2.putText(
                display_image,
                f"Looking for: {target_object}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Show result
            cv2.imshow("Perception Test", display_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Switch to next target object
                current_target_idx = (current_target_idx + 1) % len(target_objects)

    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()