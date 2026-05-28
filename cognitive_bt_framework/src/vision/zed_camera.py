import pyzed.sl as sl
import numpy as np
import cv2
import threading
import queue
import time
import collections
import os
from datetime import datetime
from typing import Optional, Tuple, Deque


class Camera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, 
                 depth_averaging_frames: int = 1, debug=True):
        """
        ZED camera wrapper with EXACT same interface as RealSense wrapper
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.debug = debug
        self.depth_averaging_frames = depth_averaging_frames
        
        # Core ZED components (matching RealSense structure)
        self.pipeline = None  # Not used in ZED but kept for interface compatibility
        self.config = None    # Not used in ZED but kept for interface compatibility
        self.align = None     # Not used in ZED but kept for interface compatibility
        self.point_cloud = None  # Will be initialized in start()
        
        # ZED-specific components
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.runtime_params = sl.RuntimeParameters()
        
        # ZED matrices for data storage
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.depth_display_mat = sl.Mat()
        self.point_cloud_mat = sl.Mat()

        # Threading components
        self._frame_queue = queue.Queue(maxsize=5)
        self._running = False
        self._thread = None
        
        # Simple depth frame buffer
        self._depth_buffer = collections.deque(maxlen=depth_averaging_frames)
        self._buffer_lock = threading.Lock()

        # Camera parameters
        self.depth_scale = 0.001  # ZED depth is in mm, convert to meters
        self.intrinsics = None
        
        # Filter placeholders (for interface compatibility with RealSense)
        self.hole_filling_filter = None
        self.spatial_filter = None
        
        # Visualization settings
        self.display_mode = 0  # 0: background removal, 1: normalized depth, 2: raw depth
        self.colormap = cv2.COLORMAP_JET
        self.colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA]
        self.colormap_index = 0
        self.clipping_distance_m = 5.0  # ZED has longer range than RealSense
        
        # Mouse callback variables
        self.mouse_x = 0
        self.mouse_y = 0
        self.current_color_image = None
        self.current_depth_image = None
        
        # Frame counter
        self.frame_count = 0
        
        # ZED-specific attributes
        self.camera_info = None
        self._resolution_map = {
            (2208, 1242): sl.RESOLUTION.HD2K,
            (1920, 1080): sl.RESOLUTION.HD1080,
            (1280, 720): sl.RESOLUTION.HD720,
            (672, 376): sl.RESOLUTION.VGA
        }

    def _init_minimal_filters(self):
        """Initialize filters - kept for interface compatibility, no-op for ZED"""
        # ZED doesn't use the same filter system as RealSense
        # This is kept for interface compatibility
        if self.debug:
            print("ZED camera: filter initialization (no-op)")

    def _get_best_zed_resolution(self):
        """Get the best matching ZED resolution for requested width/height"""
        target_res = (self.width, self.height)
        
        # Exact match
        if target_res in self._resolution_map:
            return self._resolution_map[target_res], target_res
        
        # Find closest resolution
        best_match = None
        best_diff = float('inf')
        
        for (w, h), zed_res in self._resolution_map.items():
            diff = abs(w - self.width) + abs(h - self.height)
            if diff < best_diff:
                best_diff = diff
                best_match = (zed_res, (w, h))
        
        if self.debug and best_match:
            actual_res = best_match[1]
            print(f"Requested: {self.width}x{self.height}, using closest: {actual_res[0]}x{actual_res[1]}")
        
        return (sl.RESOLUTION.HD720, (1280, 720)) #best_match if best_match else (sl.RESOLUTION.HD720, (1280, 720))

    def start(self) -> bool:
        """Start camera with verified resolution"""
        if self._running:
            return False

        try:
            # Configure ZED parameters
            zed_resolution, actual_resolution = self._get_best_zed_resolution()
            self.init_params.camera_resolution = zed_resolution
            self.init_params.camera_fps = self.fps
            self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
            self.init_params.coordinate_units = sl.UNIT.MILLIMETER
            self.init_params.depth_stabilization = 1
            
            # Runtime parameters
            self.runtime_params.confidence_threshold = 50
            self.runtime_params.texture_confidence_threshold = 100
            
            # Open camera
            err = self.zed.open(self.init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"Failed to open ZED camera: {err}")
                return False
            
            # Get camera information
            self.camera_info = self.zed.get_camera_information()
            res = self.camera_info.camera_configuration.resolution
            actual_width, actual_height = res.width, res.height
            
            # Update our resolution to match what we actually got
            if actual_width != self.width or actual_height != self.height:
                if self.debug:
                    print(f"Resolution adjusted: requested {self.width}x{self.height}, got {actual_width}x{actual_height}")
                self.width = actual_width
                self.height = actual_height
            
            # Get camera intrinsics
            calib_params = self.camera_info.camera_configuration.calibration_parameters.left_cam
            
            # Create intrinsics object compatible with RealSense format
            class Intrinsics:
                def __init__(self, fx, fy, cx, cy, coeffs):
                    self.fx = fx
                    self.fy = fy
                    self.ppx = cx  # RealSense uses ppx/ppy instead of cx/cy
                    self.ppy = cy
                    self.coeffs = coeffs
            
            self.intrinsics = Intrinsics(
                fx=calib_params.fx,
                fy=calib_params.fy,
                cx=calib_params.cx,
                cy=calib_params.cy,
                coeffs=[calib_params.disto[i] for i in range(5)]
            )
            
            # Initialize point cloud object for compatibility
            self.point_cloud = sl.Camera()  # Placeholder
            
            if self.debug:
                print(f"ZED Camera started successfully:")
                print(f"  Resolution: {self.width}x{self.height}")
                print(f"  FPS: {self.fps}")
                print(f"  Depth scale: {self.depth_scale}")
                print(f"  Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            
            # Initialize filters for compatibility
            self._init_minimal_filters()
            
            # Start processing thread
            self._running = True
            self._thread = threading.Thread(target=self._process_frames, daemon=True)
            self._thread.start()
            
            return True

        except Exception as e:
            print(f"Failed to start ZED camera: {e}")
            return False

    def _apply_minimal_processing(self, depth_frame):
        """Apply minimal processing - kept for interface compatibility"""
        # ZED doesn't need the same processing as RealSense
        # This method is kept for interface compatibility
        return depth_frame

    def _process_frames(self):
        """Frame processing thread"""
        while self._running:
            try:
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve images
                    self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
                    
                    # Convert to numpy arrays
                    color_data = self.image_mat.get_data()
                    depth_data = self.depth_mat.get_data()
                    
                    # Handle NaN values in depth data
                    depth_cleaned = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Resize if needed to match requested resolution
                    if color_data.shape[:2] != (self.height, self.width):
                        color_data = cv2.resize(color_data, (self.width, self.height))
                    if depth_cleaned.shape != (self.height, self.width):
                        depth_cleaned = cv2.resize(depth_cleaned, (self.width, self.height))
                    
                    # Verify shape periodically
                    if self.debug and self.frame_count % 100 == 0:
                        print(f"Frame {self.frame_count}: Color {color_data.shape}, Depth {depth_cleaned.shape}")
                    
                    # Add to depth buffer
                    with self._buffer_lock:
                        self._depth_buffer.append(depth_cleaned.astype(np.float32))
                    
                    # Create frames object for compatibility
                    frames = {
                        'color': color_data,
                        'depth': depth_cleaned
                    }
                    
                    if not self._frame_queue.full():
                        self._frame_queue.put(frames)
                    else:
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put(frames)
                        except queue.Empty:
                            pass
                    
                    self.frame_count += 1
                else:
                    time.sleep(0.01)

            except Exception as e:
                if self.debug:
                    print(f"Frame acquisition error: {e}")
                time.sleep(0.1)

    def get_frames(self, use_averaging=False):
        """Get frames with guaranteed correct resolution"""
        try:
            frames = self._frame_queue.get(timeout=1.0)
            
            color_image = frames['color']
            
            if use_averaging:
                depth_image = self._get_averaged_depth()
                if depth_image is None:
                    depth_image = frames['depth']
            else:
                depth_image = frames['depth']
            
            # Convert color from RGBA to BGR for consistency with RealSense
            if len(color_image.shape) == 3:
                if color_image.shape[2] == 4:  # RGBA
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)
                elif color_image.shape[2] == 3:  # RGB
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            # Final shape verification
            if color_image.shape[:2] != (self.height, self.width):
                if self.debug:
                    print(f"Unexpected color shape: {color_image.shape}")
            
            if depth_image.shape != (self.height, self.width):
                if self.debug:
                    print(f"Unexpected depth shape: {depth_image.shape}")

            return color_image, depth_image

        except queue.Empty:
            return None, None

    def _get_averaged_depth(self):
        """Simple depth averaging without artifacts"""
        with self._buffer_lock:
            if not self._depth_buffer:
                return None
                
            # Simple median-based averaging to avoid artifacts
            stacked_depths = np.stack(list(self._depth_buffer), axis=0)
            
            # Use median instead of mean to avoid noise amplification
            result_depth = np.median(stacked_depths, axis=0)
            
            return result_depth.astype(self._depth_buffer[0].dtype)

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to get depth values at clicked positions."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x, self.mouse_y = x, y
            
            # Adjust coordinates if this is the right side of a side-by-side display
            actual_x = x
            if x > self.width:  # Clicked on right side (depth image)
                actual_x = x - self.width
            
            if (self.current_depth_image is not None and 
                0 <= actual_x < self.width and 0 <= y < self.height):
                
                depth_value_raw = self.current_depth_image[y, actual_x]
                if depth_value_raw > 0:
                    depth_value_m = depth_value_raw * self.depth_scale
                    print(f"Depth at ({actual_x}, {y}): {depth_value_raw:.1f} units ({depth_value_m*1000:.1f} mm, {depth_value_m:.3f} m)")
                else:
                    print(f"No valid depth data at ({actual_x}, {y})")

    def create_depth_colormap(self, depth_image):
        """Create a colorized version of the depth image."""
        # Convert to proper format for OpenCV colormap
        if len(depth_image.shape) == 3:
            depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
        else:
            depth_gray = depth_image
        
        # Ensure we have the right data type
        if depth_gray.dtype != np.uint8:
            # Normalize to 0-255 range and convert to uint8
            depth_normalized = cv2.normalize(depth_gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
        else:
            depth_uint8 = depth_gray
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, self.colormap)
        return depth_colored

    def save_data(self):
        """Save current depth map and RGB image."""
        if self.current_color_image is None or self.current_depth_image is None:
            print("No current frames to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        output_dir = "zed_captures"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save RGB image (convert BGR to RGB for proper saving)
        rgb_filename = f"{output_dir}/rgb_{timestamp}.png"
        rgb_image = cv2.cvtColor(self.current_color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        # Save raw depth data as numpy array
        depth_raw_filename = f"{output_dir}/depth_raw_{timestamp}.npy"
        np.save(depth_raw_filename, self.current_depth_image)
        
        # Save colorized depth image
        depth_colored = self.create_depth_colormap(self.current_depth_image)
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_colored_filename = f"{output_dir}/depth_colored_{timestamp}.png"
        cv2.imwrite(depth_colored_filename, cv2.cvtColor(depth_colored_rgb, cv2.COLOR_RGB2BGR))
        
        # Save depth in millimeters as 16-bit PNG
        depth_mm = np.clip(self.current_depth_image, 0, 65535).astype(np.uint16)
        depth_mm_filename = f"{output_dir}/depth_mm_{timestamp}.png"
        cv2.imwrite(depth_mm_filename, depth_mm)
        
        print(f"Data saved to {output_dir}/ with timestamp {timestamp}")
        print(f"  RGB image saved in proper color order (matching RealSense format)")

    def display_info_overlay(self, image):
        """Add information overlay to the image."""
        height, width = image.shape[:2]
        
        # Create info text
        colormap_names = {
            cv2.COLORMAP_JET: "JET",
            cv2.COLORMAP_TURBO: "TURBO", 
            cv2.COLORMAP_VIRIDIS: "VIRIDIS",
            cv2.COLORMAP_PLASMA: "PLASMA"
        }
        colormap_name = colormap_names.get(self.colormap, "UNKNOWN")
        
        mode_names = ["BG Removal", "Normalized", "Raw Depth"]
        mode_name = mode_names[self.display_mode] if self.display_mode < len(mode_names) else "Unknown"
        
        info_text = [
            f"ZED: {self.width}x{self.height} @ {self.fps}fps",
            f"Mode: {mode_name}",
            f"Colormap: {colormap_name}",
            f"Clip: {self.clipping_distance_m:.1f}m",
            f"Mouse: ({self.mouse_x}, {self.mouse_y})",
            f"Frame: {getattr(self, 'frame_count', 0)}",
            "Controls: q=quit, s=save, d=mode, c=colormap, +/-=clip"
        ]
        
        # Add text overlay with background for better readability
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 25
            # Add background rectangle
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (5, y_pos - 20), (15 + text_size[0], y_pos + 5), (0, 0, 0), -1)
            # Add text
            cv2.putText(image, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image

    def visualize(self, use_averaging=True):
        """Advanced visualization with multiple modes and interactive features"""
        frames = self.get_frames(use_averaging=use_averaging)
        if not frames:
            return
        
        color_image, depth_image = frames
        
        # Convert BGR to RGB for consistent color representation
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.current_color_image = color_image.copy()  # Keep original BGR for saving
        self.current_depth_image = depth_image.copy()
        
        try:
            if self.display_mode == 0:
                # Background removal mode
                clipping_distance = self.clipping_distance_m / self.depth_scale  # Convert to mm
                depth_image_3d = np.dstack((depth_image,) * 3)
                viz_left = np.where(
                    (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                    153,  # Grey color
                    color_image_rgb
                )
                
                # Simple depth colorization
                depth_normalized = np.clip(depth_image.astype(np.float32) * self.depth_scale * 50, 0, 255).astype(np.uint8)
                viz_right = cv2.applyColorMap(depth_normalized, self.colormap)
                viz_right = cv2.cvtColor(viz_right, cv2.COLOR_BGR2RGB)
                
            elif self.display_mode == 1:
                # Normalized depth mode
                viz_left = color_image_rgb.copy()
                
                # Normalize depth for better visualization
                valid_mask = depth_image > 0
                if np.any(valid_mask):
                    depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
                    valid_depths = depth_image[valid_mask]
                    depth_normalized[valid_mask] = np.clip(
                        ((valid_depths - valid_depths.min()) / 
                         (valid_depths.max() - valid_depths.min()) * 255), 0, 255
                    ).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
                
                viz_right = cv2.applyColorMap(depth_normalized, self.colormap)
                viz_right = cv2.cvtColor(viz_right, cv2.COLOR_BGR2RGB)
                
            else:  # display_mode == 2
                # Raw depth values mode
                viz_left = color_image_rgb.copy()
                
                # Use raw depth values with clipping (ZED depth is in mm)
                depth_clipped = np.clip(depth_image.astype(np.float32), 0, 10000)  # Clip to 10m
                depth_norm = (depth_clipped / 10000 * 255).astype(np.uint8)
                viz_right = cv2.applyColorMap(depth_norm, self.colormap)
                viz_right = cv2.cvtColor(viz_right, cv2.COLOR_BGR2RGB)
            
            # Mark invalid regions in depth image
            invalid_mask = depth_image == 0
            viz_right[invalid_mask] = [0, 0, 0]
            
            # Create side-by-side display
            combined = np.hstack([viz_left, viz_right])
            
            # Add info overlay
            combined = self.display_info_overlay(combined)
            
            # Convert back to BGR for OpenCV display
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            
            # Display the image
            cv2.imshow('ZED Advanced Visualization', combined_bgr)
            
        except Exception as e:
            if self.debug:
                print(f"Visualization error: {e}")
            # Fallback to simple display
            simple_combined = np.hstack([color_image, cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)])
            cv2.imshow('ZED Advanced Visualization', simple_combined)

    def run_advanced_visualization(self, use_averaging=True):
        """Run interactive visualization with keyboard controls"""
        if not self._running:
            print("Camera not started!")
            return
        
        print("\nStarting ZED advanced visualization...")
        print("Controls:")
        print("  'q' or ESC: Quit")
        print("  's': Save current data")
        print("  'd': Toggle display mode (BG removal -> Normalized -> Raw)")
        print("  'c': Cycle through colormaps")
        print("  '+'/'-': Adjust clipping distance")
        print("  Click on image: Print depth value")
        print()
        
        # Set up OpenCV window
        cv2.namedWindow("ZED Advanced Visualization", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("ZED Advanced Visualization", self.mouse_callback)
        
        try:
            while True:
                self.visualize(use_averaging=use_averaging)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):
                    self.save_data()
                elif key == ord('d'):
                    self.display_mode = (self.display_mode + 1) % 3
                    mode_names = ["background removal", "normalized depth", "raw depth"]
                    print(f"Switched to {mode_names[self.display_mode]} mode")
                elif key == ord('c'):
                    self.colormap_index = (self.colormap_index + 1) % len(self.colormaps)
                    self.colormap = self.colormaps[self.colormap_index]
                    colormap_names = ["JET", "TURBO", "VIRIDIS", "PLASMA"]
                    print(f"Switched to {colormap_names[self.colormap_index]} colormap")
                elif key == ord('+') or key == ord('='):
                    self.clipping_distance_m = min(20.0, self.clipping_distance_m + 0.5)
                    print(f"Clipping distance: {self.clipping_distance_m:.1f}m")
                elif key == ord('-'):
                    self.clipping_distance_m = max(0.5, self.clipping_distance_m - 0.5)
                    print(f"Clipping distance: {self.clipping_distance_m:.1f}m")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()
            print("ZED advanced visualization stopped")

    def get_closest_blob(self, use_averaging=True):
        """Simple closest blob detection"""
        frames = self.get_frames(use_averaging=use_averaging)
        if frames is None:
            return None

        _, depth_image = frames

        # Simple minimum finding
        valid_mask = depth_image > 0
        if np.sum(valid_mask) < 100:
            return None

        valid_depths = depth_image[valid_mask] * self.depth_scale
        min_depth = np.min(valid_depths)
        
        # Find coordinates
        min_coords = np.unravel_index(
            np.argmin(np.where(depth_image > 0, depth_image, np.inf)), 
            depth_image.shape
        )

        return min_depth, min_coords[::-1]  # Return as (x, y)

    def get_point_cloud(self, use_averaging=True, manual_calculation=False):
        """
        Get point cloud from depth data
        
        Args:
            use_averaging: Whether to use averaged depth frames
            manual_calculation: If True, calculate manually using intrinsics.
                               If False, use ZED built-in point cloud.
        
        Returns:
            numpy array of 3D points (N, 3) where each row is [x, y, z]
        """
        try:
            if manual_calculation:
                return self._get_point_cloud_manual(use_averaging)
            else:
                return self._get_point_cloud_zed(use_averaging)
        except Exception as e:
            if self.debug:
                print(f"Error getting point cloud: {e}")
            return np.array([])

    def _get_point_cloud_zed(self, use_averaging=True):
        """Get point cloud using ZED built-in calculation"""
        try:
            # For ZED, we need to retrieve the point cloud directly
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZ)
                points = self.point_cloud_mat.get_data()
                
                # Reshape and filter valid points
                h, w = points.shape[:2]
                pts = points.reshape(-1, 3)
                
                # Filter out invalid points (NaN, inf, or zero)
                valid_mask = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 0.1)
                pts_filtered = pts[valid_mask]
                
                # Convert to meters (ZED uses millimeters)
                pts_filtered = pts_filtered * self.depth_scale
                
                return pts_filtered
            else:
                if self.debug:
                    print("Failed to grab ZED frame for point cloud")
                return self._get_point_cloud_manual(use_averaging)
                
        except Exception as e:
            if self.debug:
                print(f"ZED point cloud failed: {e}, falling back to manual calculation")
            return self._get_point_cloud_manual(use_averaging)

    def _get_point_cloud_manual(self, use_averaging=True):
        """Manually calculate point cloud using camera intrinsics and depth data"""
        try:
            frames = self.get_frames(use_averaging=use_averaging)
            if not frames or not self.intrinsics:
                return np.array([])
            
            color_image, depth_image = frames
            h, w = depth_image.shape
            
            # Create coordinate grids
            i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
            
            # Convert depth to meters
            depth_m = depth_image.astype(np.float32) * self.depth_scale
            
            # Filter valid depths
            valid_mask = (depth_m > 0.1) & (depth_m < 20.0)  # ZED has longer range
            
            if np.sum(valid_mask) == 0:
                return np.array([])
            
            # Get valid coordinates and depths
            i_valid = i[valid_mask]
            j_valid = j[valid_mask]
            depth_valid = depth_m[valid_mask]
            
            # Convert to 3D coordinates using pinhole camera model
            x_3d = (i_valid - self.intrinsics.ppx) * depth_valid / self.intrinsics.fx
            y_3d = (j_valid - self.intrinsics.ppy) * depth_valid / self.intrinsics.fy
            z_3d = depth_valid
            
            # Stack coordinates
            points_3d = np.column_stack((x_3d, y_3d, z_3d))
            
            return points_3d
            
        except Exception as e:
            if self.debug:
                print(f"Manual point cloud calculation failed: {e}")
            return np.array([])

    def get_point_cloud_with_colors(self, use_averaging=True):
        """
        Get point cloud with corresponding RGB colors
        
        Returns:
            Tuple of (points, colors) where:
            - points: (N, 3) array of [x, y, z] coordinates
            - colors: (N, 3) array of [r, g, b] values (0-255) - consistent RGB format
        """
        try:
            frames = self.get_frames(use_averaging=use_averaging)
            if not frames or not self.intrinsics:
                return np.array([]), np.array([])
            
            color_image, depth_image = frames
            h, w = depth_image.shape
            
            # Create coordinate grids
            i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
            
            # Convert depth to meters
            depth_m = depth_image.astype(np.float32) * self.depth_scale
            
            # Filter valid depths
            valid_mask = (depth_m > 0.1) & (depth_m < 20.0)
            
            if np.sum(valid_mask) == 0:
                return np.array([]), np.array([])
            
            # Get valid coordinates, depths, and colors
            i_valid = i[valid_mask]
            j_valid = j[valid_mask]
            depth_valid = depth_m[valid_mask]
            
            # Get corresponding colors - convert BGR to RGB for consistency
            colors_bgr = color_image[j_valid, i_valid]  # Note: j,i for row,col indexing
            colors_rgb = colors_bgr[:, [2, 1, 0]]  # Convert BGR to RGB for consistent format
            
            # Convert to 3D coordinates
            x_3d = (i_valid - self.intrinsics.ppx) * depth_valid / self.intrinsics.fx
            y_3d = (j_valid - self.intrinsics.ppy) * depth_valid / self.intrinsics.fy
            z_3d = depth_valid
            
            # Stack coordinates
            points_3d = np.column_stack((x_3d, y_3d, z_3d))
            
            return points_3d, colors_rgb
            
        except Exception as e:
            if self.debug:
                print(f"Colored point cloud failed: {e}")
            return np.array([]), np.array([])

    def visualize_point_cloud(self, use_averaging=True, max_points=10000):
        """
        Simple 3D visualization of point cloud (requires matplotlib)
        
        Args:
            use_averaging: Use averaged depth frames
            max_points: Maximum number of points to display (for performance)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            points, colors = self.get_point_cloud_with_colors(use_averaging=use_averaging)
            
            if len(points) == 0:
                if self.debug:
                    print("No valid points for visualization")
                return
            
            # Subsample for performance if needed
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points with colors
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors/255.0, s=1, alpha=0.6)
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Point Cloud ({len(points)} points)')
            
            # Set equal aspect ratio
            max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                                 points[:, 1].max()-points[:, 1].min(),
                                 points[:, 2].max()-points[:, 2].min()]).max() / 2.0
            mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.show()
            
        except ImportError:
            print("matplotlib not available for 3D visualization")
        except Exception as e:
            if self.debug:
                print(f"Point cloud visualization failed: {e}")

    def save_point_cloud_ply(self, filename, use_averaging=True, include_colors=True):
        """
        Save point cloud to PLY file format
        
        Args:
            filename: Output PLY file path
            use_averaging: Use averaged depth frames
            include_colors: Whether to include RGB colors
        """
        try:
            if include_colors:
                points, colors = self.get_point_cloud_with_colors(use_averaging=use_averaging)
            else:
                points = self.get_point_cloud(use_averaging=use_averaging, manual_calculation=True)
                colors = None
            
            if len(points) == 0:
                if self.debug:
                    print("No points to save")
                return False
            
            # Write PLY file
            with open(filename, 'w') as f:
                # PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                
                if include_colors and colors is not None:
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                
                f.write("end_header\n")
                
                # Write vertex data
                for i in range(len(points)):
                    x, y, z = points[i]
                    if include_colors and colors is not None:
                        r, g, b = colors[i].astype(int)
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                    else:
                        f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            
            if self.debug:
                print(f"Saved {len(points)} points to {filename}")
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Failed to save PLY file: {e}")
            return False

    def stop(self):
        """Stop camera"""
        self._running = False
        if self._thread:
            self._thread.join()
        
        self.zed.close()
        cv2.destroyAllWindows()
        
        if self.debug:
            print("ZED camera stopped")


# Ultra-simple fallback (for interface compatibility)
class UltraSimpleCamera:
    def __init__(self, width=640, height=480, fps=30, debug=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.debug = debug
        
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.runtime_params = sl.RuntimeParameters()
        
        # Bare minimum configuration
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        
        self.depth_scale = 0.001
        self._running = False
        
        # ZED matrices
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()

    def start(self):
        try:
            err = self.zed.open(self.init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"Failed to open ZED: {err}")
                return False
            
            self._running = True
            
            if self.debug:
                print(f"Ultra-simple ZED camera started: {self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"Failed to start ultra-simple ZED camera: {e}")
            return False

    def get_frames(self):
        if not self._running:
            return None
        
        try:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
                
                color_image = self.image_mat.get_data()
                depth_image = self.depth_mat.get_data()
                
                # Convert color from RGBA to BGR if needed
                if len(color_image.shape) == 3 and color_image.shape[2] == 4:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)
                
                # Handle NaN values
                depth_image = np.nan_to_num(depth_image, nan=0.0)
                
                return color_image, depth_image
            else:
                return None
                
        except Exception as e:
            if self.debug:
                print(f"Frame error: {e}")
            return None

    def stop(self):
        self._running = False
        self.zed.close()


# Test the aligned ZED camera wrapper
if __name__ == "__main__":
    print("Testing aligned ZED camera wrapper...")
    
    # Test the main camera with same interface as RealSense
    camera = Camera(
        width=640, 
        height=480, 
        fps=30, 
        depth_averaging_frames=3,
        debug=True
    )
    
    if camera.start():
        print("ZED camera started! Testing advanced visualization...")
        
        # Test a few frames first
        for i in range(3):
            frames = camera.get_frames()
            if frames:
                color, depth = frames
                print(f"Frame {i}: Color {color.shape}, Depth {depth.shape}")
                valid_pixels = np.sum(depth > 0)
                total_pixels = depth.size
                print(f"  Valid depth pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
            time.sleep(0.1)
        
        print("\nRunning advanced interactive visualization...")
        print("Use mouse to click for depth values, keyboard for controls!")
        
        # Run the advanced visualization
        camera.run_advanced_visualization(use_averaging=True)
        
        camera.stop()
    else:
        print("Failed to start ZED camera")