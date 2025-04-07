import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import queue
import time
from typing import Optional, Tuple, Union


class Camera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, debug=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.debug=debug
        # Core RealSense components
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.point_cloud = rs.pointcloud()
        # Configure streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Threading components
        self._frame_queue = queue.Queue(maxsize=5)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Camera parameters
        self.depth_scale: Optional[float] = None

    def start(self) -> bool:
        """Start the camera stream and processing thread."""
        if self._running:
            return False

        try:
            profile = self.pipeline.start(self.config)
            self.intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self._running = True
            self._thread = threading.Thread(target=self._process_frames, daemon=True)
            self._thread.start()
            return True

        except RuntimeError as e:
            print(f"Failed to start camera: {e}")
            return False

    def stop(self) -> None:
        """Stop the camera stream and processing thread."""
        self._running = False
        if self._thread:
            self._thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def _process_frames(self) -> None:
        """Process frames in background thread."""
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)

                if not self._frame_queue.full():
                    self._frame_queue.put(aligned_frames)
                else:
                    # Discard oldest frame if queue is full
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put(aligned_frames)
                    except queue.Empty:
                        pass

            except RuntimeError as e:
                if self.debug:
                    print(f"Frame acquisition error: {e}")
                time.sleep(0.1)
    
    def get_point_cloud(self, depth_frame):
        aligned_frames = self._frame_queue.get(timeout=1.0)

        depth_frame = aligned_frames.get_depth_frame()
        points =  self.point_cloud.calculate(depth_frame)
        
        return points

    
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get latest color and depth frames."""
        try:
            aligned_frames = self._frame_queue.get(timeout=1.0)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image

        except queue.Empty:
            return None, None

    def process_frames(self, clipping_distance_m: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Process frames with background removal and depth colorization."""
        frames = self.get_frames()
        if not frames or not self.depth_scale:
            return None

        color_image, depth_image = frames

        # Remove background
        clipping_distance = clipping_distance_m / self.depth_scale
        depth_image_3d = np.dstack((depth_image,) * 3)
        bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            153,  # Grey color
            color_image
        )

        # Colorize depth map
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        return bg_removed, depth_colormap

    def get_closest_blob(self) -> Optional[Tuple[float, Tuple[int, int]]]:
        """Returns the distance and coordinates of the closest blob to the camera."""
        frames = self.get_frames()
        if frames is None or self.depth_scale is None:
            return None

        _, depth_image = frames

        # Mask out zero values (which indicate no depth information)
        non_zero_depths = np.where(depth_image > 0, depth_image, np.inf)
        min_distance = np.min(non_zero_depths)

        # If no valid depth was found, return None
        if min_distance == np.inf:
            return None

        # Find coordinates of the closest point
        min_coords = np.unravel_index(np.argmin(non_zero_depths), non_zero_depths.shape)

        # Convert distance to meters
        min_distance_m = min_distance * self.depth_scale

        return min_distance_m, min_coords

    def visualize(self, clipping_distance_m: float = 1.0) -> None:
        """Display processed frames."""
        result = self.process_frames(clipping_distance_m)
        if result:
            bg_removed, depth_colormap = result
            combined = np.hstack((bg_removed, depth_colormap))
            cv2.imshow('RealSense Camera', combined)
            cv2.waitKey(1)


if __name__ == "__main__":
    camera = Camera()
    if camera.start():
        try:
            while True:
                camera.visualize()

                # Example usage of get_closest_blob method
                closest_blob = camera.get_closest_blob()
                if closest_blob:
                    distance, (x, y) = closest_blob
                    print(f"Closest blob is at ({x}, {y}) with a distance of {distance:.2f} meters")

                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        finally:
            camera.stop()