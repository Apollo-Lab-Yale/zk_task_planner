import numpy as np
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision.mapping.voxel_map_3d import EnvironmentMapper
import pyrealsense2 as rs
import cv2
import time
from scipy.spatial.transform import Rotation
import signal
import sys


class MappingApplication:
    def __init__(self, voxel_size=0.05):
        self.camera = Camera()
        self.mapper = EnvironmentMapper(voxel_size=voxel_size)
        self.running = True
        self.camera_matrix = None

        # Configuration
        self.center = np.array([0.0, 0.0, 0.0])
        self.radius = 1.0
        self.vis_update_interval = 0.1  # 10 Hz visualization update
        self.depth_scale = 0.03  # Scale factor for depth visualization

        # Initialize timing variables
        self.start_time = None
        self.last_vis_update = None

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\nSignal received, cleaning up...")
        self.running = False

    def initialize(self):
        """Initialize camera and get intrinsics"""
        if not self.camera.start():
            print("Failed to start camera")
            return False

        try:
            # Get camera intrinsics
            profile = self.camera.pipeline.get_active_profile()
            depth_profile = profile.get_stream(rs.stream.depth)
            intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

            self.camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])

            # Initialize visualization
            self.mapper.initialize_visualizer()

            # Initialize timing
            self.start_time = time.time()
            self.last_vis_update = time.time()

            return True

        except Exception as e:
            print(f"Error during initialization: {e}")
            return False

    def update_robot_pose(self):
        """Update robot pose based on time"""
        t = time.time() - self.start_time
        angle = t * 0.5

        position = self.center + self.radius * np.array([
            np.cos(angle),
            np.sin(angle),
            0.0
        ])

        direction = self.center - position
        orientation = Rotation.from_rotvec(
            [0, 0, np.arctan2(direction[1], direction[0])]
        )

        self.mapper.update_robot_pose(position, orientation)

    def process_frames(self):
        """Process camera frames and update visualizations"""
        frames = self.camera.get_frames()
        if not frames:
            return True

        color_image, depth_image = frames

        # Update mapping
        self.update_robot_pose()
        self.mapper.update_map(depth_image, self.camera_matrix)

        # Visualize raw depth
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=self.depth_scale),
            cv2.COLORMAP_JET
        )
        cv2.imshow('Raw Depth', depth_colormap)

        # Visualize color
        cv2.imshow('Color', color_image)

        # Update 3D visualization at controlled rate
        current_time = time.time()
        if current_time - self.last_vis_update >= self.vis_update_interval:
            self.mapper.visualize(block=False)
            self.last_vis_update = current_time

        # Check for exit command
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord('q'), 27)

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up resources...")
        try:
            self.mapper.close_visualizer()
            self.camera.stop()
            cv2.destroyAllWindows()
            self.mapper.save_map('environment_map.ply')
            print("Cleanup complete")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def run(self):
        """Main application loop"""
        if not self.initialize():
            return

        try:
            while self.running:
                if not self.process_frames():
                    break
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()


def main():
    app = MappingApplication(voxel_size=0.05)
    app.run()


if __name__ == "__main__":
    main()