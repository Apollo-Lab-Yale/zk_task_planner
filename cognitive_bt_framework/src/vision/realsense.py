import pyrealsense2 as rs
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
                 depth_averaging_frames: int = 5, debug=True, config_file: str = None,
                 exposure: int = None, gain: int = None, laser_power: int = None, 
                 auto_exposure: bool = None, use_viewer_defaults: bool = True,
                 trigger_calibration: bool = False):
        """
        Simple camera that just works - now with RealSense Viewer quality defaults
        
        Args:
            width: Camera width resolution
            height: Camera height resolution  
            fps: Frames per second
            depth_averaging_frames: Number of frames to average for depth
            debug: Enable debug output
            config_file: Path to config file to load/save parameters
            exposure: Initial exposure time in microseconds (None = auto)
            gain: Initial gain value (None = auto)
            laser_power: Initial laser power (None = auto/default)
            auto_exposure: Enable/disable auto exposure (None = use viewer default)
            use_viewer_defaults: Use RealSense Viewer quality settings
            trigger_calibration: Trigger device calibration after camera start
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.debug = debug
        self.depth_averaging_frames = depth_averaging_frames
        self.use_viewer_defaults = use_viewer_defaults
        self.trigger_calibration = trigger_calibration
        
        # Configuration file management
        self.config_file = config_file or "realsense_config.json"
        self.loaded_config = None
        
        # Initial parameter settings - None means use defaults
        self.initial_exposure = exposure
        self.initial_gain = gain
        self.initial_laser_power = laser_power
        self.initial_auto_exposure = auto_exposure
        
        # Core RealSense components
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.point_cloud = rs.pointcloud()
        
        # Configure streams - use native 640x480
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # Threading components
        self._frame_queue = queue.Queue(maxsize=5)
        self._running = False
        self._thread = None
        
        # Simple depth frame buffer
        self._depth_buffer = collections.deque(maxlen=depth_averaging_frames)
        self._buffer_lock = threading.Lock()

        # Camera parameters
        self.depth_scale = None
        self.intrinsics = None
        self.depth_sensor = None
        
        # Parameter tuning variables
        self.current_exposure = None
        self.current_gain = None
        self.current_laser_power = None
        self.auto_exposure_enabled = True
        
        # Post-processing filters - full pipeline for viewer-quality output
        self.decimation_filter = None
        self.temporal_filter = None
        self.spatial_filter = None
        self.hole_filling_filter = None
        self.threshold_filter = None
        
        # Visualization settings
        self.display_mode = 0  # 0: background removal, 1: normalized depth, 2: raw depth
        self.colormap = cv2.COLORMAP_JET
        self.colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA]
        self.colormap_index = 0
        self.clipping_distance_m = 3.0
        
        # Camera orientation handling
        self.camera_orientation = 0  # 0: normal, 90: rotated 90° clockwise, 180: upside down, 270: rotated 90° counterclockwise
        self.save_orientation_results = True  # Enable saving orientation correction results
        self.orientation_save_dir = "orientation_correction_results"
        self.frame_count_for_saving = 0
        
        # Mouse callback variables
        self.mouse_x = 0
        self.mouse_y = 0
        self.current_color_image = None
        self.current_depth_image = None

    def _init_viewer_quality_filters(self):
        """Initialize full post-processing pipeline matching RealSense Viewer quality"""
        try:
            if self.debug:
                print("Initializing RealSense Viewer quality post-processing pipeline...")
            
            # Decimation filter (optional - reduces resolution but improves performance)
            # Comment out if you want to keep full resolution
            # self.decimation_filter = rs.decimation_filter()
            # self.decimation_filter.set_option(rs.option.filter_magnitude, 2)  # 2x decimation
            
            # Threshold filter - removes values too close or too far
            self.threshold_filter = rs.threshold_filter()
            self.threshold_filter.set_option(rs.option.min_distance, 0.1)  # 10cm minimum
            self.threshold_filter.set_option(rs.option.max_distance, 4.0)   # 4m maximum
            
            # Temporal filter - reduces noise over time
            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
            self.temporal_filter.set_option(rs.option.holes_fill, 3)  # Fill temporal holes
            
            # Spatial filter - reduces noise within frame
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)      # Moderate filtering
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5) # Balance between smoothing and details
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)  # Edge-preserving
            self.spatial_filter.set_option(rs.option.holes_fill, 0)           # Don't fill holes aggressively here
            
            # Hole filling filter - fills remaining holes
            self.hole_filling_filter = rs.hole_filling_filter()
            # Mode 1 = fill from left, Mode 2 = farthest from around
            self.hole_filling_filter.set_option(rs.option.holes_fill, 1)
            
            if self.debug:
                print("Full post-processing pipeline initialized:")
                print("  - Threshold filter: 0.1m - 4.0m")
                print("  - Temporal filter: alpha=0.4, delta=20")
                print("  - Spatial filter: magnitude=2, alpha=0.5, delta=20")
                print("  - Hole filling: mode=1 (fill from left)")
                
        except Exception as e:
            if self.debug:
                print(f"Failed to initialize filters: {e}")
            # Fallback to minimal filters
            self._init_minimal_filters()

    def _init_minimal_filters(self):
        """Initialize only essential filters with conservative settings (original behavior)"""
        try:
            # Only hole filling - very conservative
            self.hole_filling_filter = rs.hole_filling_filter()
            
            # Very light spatial filter - minimal settings
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 1)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.25)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 10)
            self.spatial_filter.set_option(rs.option.holes_fill, 0)
            
            if self.debug:
                print("Minimal filters initialized (conservative settings)")
                
        except Exception as e:
            if self.debug:
                print(f"Failed to initialize filters: {e}")
            self.hole_filling_filter = None
            self.spatial_filter = None

    def apply_viewer_defaults(self):
        """Apply RealSense Viewer default settings for best quality"""
        if not self.depth_sensor:
            if self.debug:
                print("No depth sensor available for setting defaults")
            return False
        
        try:
            if self.debug:
                print("Applying RealSense Viewer default settings...")
            
            # Set visual preset to "Default" (which is what viewer uses)
            if self.depth_sensor.supports(rs.option.visual_preset):
                self.depth_sensor.set_option(rs.option.visual_preset, 0)  # Default preset
                if self.debug:
                    print("  - Visual preset: Default")
            
            # Enable auto exposure (viewer default)
            if self.depth_sensor.supports(rs.option.enable_auto_exposure):
                if self.initial_auto_exposure is None:  # Only if not explicitly set
                    self.depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
                    self.auto_exposure_enabled = True
                    if self.debug:
                        print("  - Auto exposure: Enabled")
            
            # Only set manual values if explicitly provided
            if self.initial_exposure is not None:
                if self.depth_sensor.supports(rs.option.enable_auto_exposure):
                    self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    self.auto_exposure_enabled = False
                if self.depth_sensor.supports(rs.option.exposure):
                    self.depth_sensor.set_option(rs.option.exposure, self.initial_exposure)
                    if self.debug:
                        print(f"  - Manual exposure: {self.initial_exposure}μs")
            
            if self.initial_gain is not None:
                if self.depth_sensor.supports(rs.option.gain):
                    self.depth_sensor.set_option(rs.option.gain, self.initial_gain)
                    if self.debug:
                        print(f"  - Manual gain: {self.initial_gain}")
            
            if self.initial_laser_power is not None:
                if self.depth_sensor.supports(rs.option.laser_power):
                    self.depth_sensor.set_option(rs.option.laser_power, self.initial_laser_power)
                    if self.debug:
                        print(f"  - Manual laser power: {self.initial_laser_power}")
            
            # Additional viewer-like settings
            if self.depth_sensor.supports(rs.option.confidence_threshold):
                self.depth_sensor.set_option(rs.option.confidence_threshold, 3)  # Default confidence
                if self.debug:
                    print("  - Confidence threshold: 3")
            
            # Print final applied settings
            final_params = self.get_current_parameters()
            if self.debug:
                print("Applied settings:")
                for param, value in final_params.items():
                    print(f"    {param}: {value}")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Failed to apply viewer defaults: {e}")
            return False

    def trigger_device_calibration(self):
        """
        Trigger device calibration after the camera is started.
        This can help improve depth accuracy.
        """
        if not self.depth_sensor:
            if self.debug:
                print("No depth sensor available for calibration")
            return False
        
        try:
            # Get the device from the sensor
            device = self.depth_sensor.get_device()
            
            # Check if the device supports on-chip calibration
            if device.supports(rs.camera_info.product_id):
                product_id = device.get_info(rs.camera_info.product_id)
                if self.debug:
                    print(f"Device product ID: {product_id}")
            
            # Trigger calibration if supported
            if hasattr(device, 'trigger_device_calibration'):
                device.trigger_device_calibration()
                if self.debug:
                    print("Device calibration triggered successfully")
                return True
            else:
                if self.debug:
                    print("Device calibration not supported on this device")
                return False
                
        except Exception as e:
            if self.debug:
                print(f"Failed to trigger device calibration: {e}")
            return False

    def get_parameter_ranges(self):
        """Get the valid ranges for tunable parameters"""
        if not self.depth_sensor:
            return None
        
        ranges = {}
        try:
            if self.depth_sensor.supports(rs.option.exposure):
                exp_range = self.depth_sensor.get_option_range(rs.option.exposure)
                ranges['exposure'] = (int(exp_range.min), int(exp_range.max), int(exp_range.step))
                
            if self.depth_sensor.supports(rs.option.gain):
                gain_range = self.depth_sensor.get_option_range(rs.option.gain)
                ranges['gain'] = (int(gain_range.min), int(gain_range.max), int(gain_range.step))
                
            if self.depth_sensor.supports(rs.option.laser_power):
                laser_range = self.depth_sensor.get_option_range(rs.option.laser_power)
                ranges['laser_power'] = (int(laser_range.min), int(laser_range.max), int(laser_range.step))
                
        except Exception as e:
            if self.debug:
                print(f"Error getting parameter ranges: {e}")
        
        return ranges

    def set_exposure(self, exposure_us):
        """Set exposure time in microseconds"""
        if not self.depth_sensor or not self.depth_sensor.supports(rs.option.exposure):
            return False
        
        try:
            # Disable auto exposure first
            if self.depth_sensor.supports(rs.option.enable_auto_exposure):
                self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
                self.auto_exposure_enabled = False
            
            self.depth_sensor.set_option(rs.option.exposure, exposure_us)
            self.current_exposure = exposure_us
            if self.debug:
                print(f"Exposure set to {exposure_us} μs")
            return True
        except Exception as e:
            if self.debug:
                print(f"Failed to set exposure: {e}")
            return False

    def set_gain(self, gain):
        """Set gain value"""
        if not self.depth_sensor or not self.depth_sensor.supports(rs.option.gain):
            return False
        
        try:
            self.depth_sensor.set_option(rs.option.gain, gain)
            self.current_gain = gain
            if self.debug:
                print(f"Gain set to {gain}")
            return True
        except Exception as e:
            if self.debug:
                print(f"Failed to set gain: {e}")
            return False

    def set_laser_power(self, power):
        """Set laser power (0-360)"""
        if not self.depth_sensor or not self.depth_sensor.supports(rs.option.laser_power):
            return False
        
        try:
            self.depth_sensor.set_option(rs.option.laser_power, power)
            self.current_laser_power = power
            if self.debug:
                print(f"Laser power set to {power}")
            return True
        except Exception as e:
            if self.debug:
                print(f"Failed to set laser power: {e}")
            return False

    def toggle_auto_exposure(self):
        """Toggle auto exposure on/off"""
        if not self.depth_sensor or not self.depth_sensor.supports(rs.option.enable_auto_exposure):
            return False
        
        try:
            current_state = self.depth_sensor.get_option(rs.option.enable_auto_exposure)
            new_state = 1 - current_state
            self.depth_sensor.set_option(rs.option.enable_auto_exposure, new_state)
            self.auto_exposure_enabled = bool(new_state)
            
            if self.debug:
                print(f"Auto exposure {'enabled' if self.auto_exposure_enabled else 'disabled'}")
            return True
        except Exception as e:
            if self.debug:
                print(f"Failed to toggle auto exposure: {e}")
            return False

    def get_camera_matrix(self):
        """
        Get the camera matrix for 3D projection of aligned depth data.
        This uses color intrinsics since depth is aligned to color.
        Adjusts the principal point based on camera orientation.
        
        Returns:
            3x3 numpy array representing the camera matrix
        """
        if not hasattr(self, 'intrinsics') or self.intrinsics is None:
            return None
        
        # Get the original intrinsics
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        ppx, ppy = self.intrinsics.ppx, self.intrinsics.ppy
        
        # Adjust principal point based on rotation
        if self.camera_orientation == 90:
            # 90 degree clockwise rotation: swap and adjust coordinates
            new_fx, new_fy = fy, fx
            new_ppx = self.height - 1 - ppy
            new_ppy = ppx
        elif self.camera_orientation == 180:
            # 180 degree rotation: invert coordinates
            new_fx, new_fy = fx, fy
            new_ppx = self.width - 1 - ppx
            new_ppy = self.height - 1 - ppy
        elif self.camera_orientation == 270:
            # 270 degree clockwise rotation: swap and adjust coordinates
            new_fx, new_fy = fy, fx
            new_ppx = ppy
            new_ppy = self.width - 1 - ppx
        else:
            # No rotation
            new_fx, new_fy = fx, fy
            new_ppx, new_ppy = ppx, ppy
            
        return np.array([
            [new_fx, 0, new_ppx],
            [0, new_fy, new_ppy],
            [0, 0, 1]
        ])
    
    def validate_3d_projection(self, test_pixel_x=320, test_pixel_y=240, test_depth=1.0):
        """
        Validate 3D projection accuracy by testing round-trip conversion.
        This helps debug alignment and intrinsics issues.
        
        Args:
            test_pixel_x: Test pixel X coordinate
            test_pixel_y: Test pixel Y coordinate  
            test_depth: Test depth value in meters
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'test_input': {'pixel_x': test_pixel_x, 'pixel_y': test_pixel_y, 'depth': test_depth},
            'using_original_depth_intrinsics': {},
            'using_color_intrinsics': {},
            'projection_errors': {}
        }
        
        # Test with original depth intrinsics
        if hasattr(self, 'depth_intrinsics_original'):
            x_3d_orig = (test_pixel_x - self.depth_intrinsics_original.ppx) * test_depth / self.depth_intrinsics_original.fx
            y_3d_orig = (test_pixel_y - self.depth_intrinsics_original.ppy) * test_depth / self.depth_intrinsics_original.fy
            
            # Back-project to pixels
            px_back_orig = (x_3d_orig * self.depth_intrinsics_original.fx / test_depth) + self.depth_intrinsics_original.ppx
            py_back_orig = (y_3d_orig * self.depth_intrinsics_original.fy / test_depth) + self.depth_intrinsics_original.ppy
            
            results['using_original_depth_intrinsics'] = {
                '3d_point': [x_3d_orig, y_3d_orig, test_depth],
                'back_projected_pixel': [px_back_orig, py_back_orig],
                'round_trip_error': [abs(px_back_orig - test_pixel_x), abs(py_back_orig - test_pixel_y)]
            }
        
        # Test with color intrinsics (what we should use for aligned depth)
        x_3d_color = (test_pixel_x - self.color_intrinsics.ppx) * test_depth / self.color_intrinsics.fx
        y_3d_color = (test_pixel_y - self.color_intrinsics.ppy) * test_depth / self.color_intrinsics.fy
        
        # Back-project to pixels
        px_back_color = (x_3d_color * self.color_intrinsics.fx / test_depth) + self.color_intrinsics.ppx
        py_back_color = (y_3d_color * self.color_intrinsics.fy / test_depth) + self.color_intrinsics.ppy
        
        results['using_color_intrinsics'] = {
            '3d_point': [x_3d_color, y_3d_color, test_depth],
            'back_projected_pixel': [px_back_color, py_back_color],
            'round_trip_error': [abs(px_back_color - test_pixel_x), abs(py_back_color - test_pixel_y)]
        }
        
        # Calculate differences between the two methods
        if hasattr(self, 'depth_intrinsics_original'):
            results['projection_errors'] = {
                '3d_position_difference': [
                    abs(x_3d_orig - x_3d_color),
                    abs(y_3d_orig - y_3d_color),
                    0.0
                ],
                'max_3d_error_mm': max(abs(x_3d_orig - x_3d_color), abs(y_3d_orig - y_3d_color)) * 1000
            }
        
        return results
    
    def get_current_parameters(self):
        """Get current parameter values"""
        if not self.depth_sensor:
            return {}
        
        params = {}
        try:
            if self.depth_sensor.supports(rs.option.exposure):
                params['exposure'] = int(self.depth_sensor.get_option(rs.option.exposure))
            if self.depth_sensor.supports(rs.option.gain):
                params['gain'] = int(self.depth_sensor.get_option(rs.option.gain))
            if self.depth_sensor.supports(rs.option.laser_power):
                params['laser_power'] = int(self.depth_sensor.get_option(rs.option.laser_power))
            if self.depth_sensor.supports(rs.option.enable_auto_exposure):
                params['auto_exposure'] = bool(self.depth_sensor.get_option(rs.option.enable_auto_exposure))
            if self.depth_sensor.supports(rs.option.visual_preset):
                params['visual_preset'] = int(self.depth_sensor.get_option(rs.option.visual_preset))
            if self.depth_sensor.supports(rs.option.confidence_threshold):
                params['confidence_threshold'] = int(self.depth_sensor.get_option(rs.option.confidence_threshold))
        except Exception as e:
            if self.debug:
                print(f"Error getting current parameters: {e}")
        
        return params

    def set_camera_orientation(self, orientation):
        """
        Set the camera orientation for image correction
        
        Args:
            orientation: 0 (normal), 90 (rotated 90° clockwise), 180 (upside down), 270 (rotated 90° counterclockwise)
        """
        if orientation not in [0, 90, 180, 270]:
            if self.debug:
                print(f"Invalid orientation: {orientation}. Must be 0, 90, 180, or 270 degrees")
            return False
            
        self.camera_orientation = orientation
        if self.debug:
            print(f"Camera orientation set to: {orientation} degrees")
        return True
    
    def get_camera_orientation(self):
        """Get the current camera orientation"""
        return self.camera_orientation
    
    def set_save_orientation_results(self, enable=True, save_dir=None):
        """
        Enable or disable saving of orientation correction results
        
        Args:
            enable: Whether to save results
            save_dir: Custom directory for saving results (optional)
        """
        self.save_orientation_results = enable
        if save_dir:
            self.orientation_save_dir = save_dir
        
        if enable and self.debug:
            print(f"Orientation result saving enabled. Results will be saved to: {self.orientation_save_dir}")
        elif self.debug:
            print("Orientation result saving disabled")
    
    def save_current_orientation_results(self):
        """
        Manually trigger saving of current frame's orientation correction results
        Useful for debugging or testing specific frames
        """
        try:
            frames = self.get_frames(use_averaging=False)
            if frames is None:
                if self.debug:
                    print("No frames available to save orientation results")
                return False
            
            # Get the uncorrected frames by calling the original processing
            aligned_frames = self._frame_queue.get(timeout=1.0)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                if self.debug:
                    print("Could not get raw frames for orientation comparison")
                return False
            
            original_color = np.asanyarray(color_frame.get_data())
            processed_depth_frame = self._apply_post_processing(depth_frame)
            original_depth = np.asanyarray(processed_depth_frame.get_data())
            
            # Apply rotation to get corrected images
            corrected_color = self._rotate_image(original_color)
            corrected_depth = self._rotate_image(original_depth)
            
            # Save the comparison
            self._save_orientation_results(original_color, corrected_color, 
                                         original_depth, corrected_depth)
            
            if self.debug:
                print("Manual orientation results saved successfully")
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Error manually saving orientation results: {e}")
            return False
    
    def _rotate_image(self, image):
        """
        Rotate image based on camera orientation to ensure it's right-side up
        
        Args:
            image: Input image (color or depth)
            
        Returns:
            Rotated image
        """
        if self.camera_orientation == 0:
            return image
        elif self.camera_orientation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.camera_orientation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif self.camera_orientation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
    
    def _save_orientation_results(self, original_color, corrected_color, original_depth, corrected_depth):
        """
        Save before/after images showing the results of orientation correction
        
        Args:
            original_color: Original color image before rotation
            corrected_color: Color image after rotation
            original_depth: Original depth image before rotation  
            corrected_depth: Depth image after rotation
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.orientation_save_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Create side-by-side comparison images
            
            # Color image comparison (convert BGR to RGB for consistent saving)
            original_color_rgb = cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB)
            corrected_color_rgb = cv2.cvtColor(corrected_color, cv2.COLOR_BGR2RGB)
            
            # Resize images to same height if rotation changed dimensions
            max_height = max(original_color_rgb.shape[0], corrected_color_rgb.shape[0])
            if original_color_rgb.shape[0] != max_height:
                scale = max_height / original_color_rgb.shape[0]
                new_width = int(original_color_rgb.shape[1] * scale)
                original_color_rgb = cv2.resize(original_color_rgb, (new_width, max_height))
            if corrected_color_rgb.shape[0] != max_height:
                scale = max_height / corrected_color_rgb.shape[0]
                new_width = int(corrected_color_rgb.shape[1] * scale)
                corrected_color_rgb = cv2.resize(corrected_color_rgb, (new_width, max_height))
            
            # Create side-by-side color comparison
            color_comparison = np.hstack([original_color_rgb, corrected_color_rgb])
            
            # Add labels
            cv2.putText(color_comparison, f"ORIGINAL (0°)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(color_comparison, f"CORRECTED ({self.camera_orientation}°)", 
                       (original_color_rgb.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save color comparison (convert back to BGR for OpenCV saving)
            color_filename = f"{self.orientation_save_dir}/color_orientation_{self.camera_orientation}deg_{timestamp}.png"
            cv2.imwrite(color_filename, cv2.cvtColor(color_comparison, cv2.COLOR_RGB2BGR))
            
            # Depth image comparison (create colorized versions for visualization)
            original_depth_colored = self.create_depth_colormap(original_depth)
            corrected_depth_colored = self.create_depth_colormap(corrected_depth)
            
            # Convert depth colormaps to RGB for consistency
            original_depth_rgb = cv2.cvtColor(original_depth_colored, cv2.COLOR_BGR2RGB)
            corrected_depth_rgb = cv2.cvtColor(corrected_depth_colored, cv2.COLOR_BGR2RGB)
            
            # Resize depth images to same height if needed
            max_height = max(original_depth_rgb.shape[0], corrected_depth_rgb.shape[0])
            if original_depth_rgb.shape[0] != max_height:
                scale = max_height / original_depth_rgb.shape[0]
                new_width = int(original_depth_rgb.shape[1] * scale)
                original_depth_rgb = cv2.resize(original_depth_rgb, (new_width, max_height))
            if corrected_depth_rgb.shape[0] != max_height:
                scale = max_height / corrected_depth_rgb.shape[0]
                new_width = int(corrected_depth_rgb.shape[1] * scale)
                corrected_depth_rgb = cv2.resize(corrected_depth_rgb, (new_width, max_height))
            
            # Create side-by-side depth comparison
            depth_comparison = np.hstack([original_depth_rgb, corrected_depth_rgb])
            
            # Add labels to depth comparison
            cv2.putText(depth_comparison, f"ORIGINAL (0°)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(depth_comparison, f"CORRECTED ({self.camera_orientation}°)", 
                       (original_depth_rgb.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save depth comparison
            depth_filename = f"{self.orientation_save_dir}/depth_orientation_{self.camera_orientation}deg_{timestamp}.png"
            cv2.imwrite(depth_filename, cv2.cvtColor(depth_comparison, cv2.COLOR_RGB2BGR))
            
            if self.debug:
                print(f"Saved orientation correction results:")
                print(f"  Color: {color_filename}")
                print(f"  Depth: {depth_filename}")
                print(f"  Original shape: {original_color.shape}")
                print(f"  Corrected shape: {corrected_color.shape}")
                
        except Exception as e:
            if self.debug:
                print(f"Error saving orientation results: {e}")

    def start(self) -> bool:
        """Start camera with verified resolution"""
        if self._running:
            return False

        try:
            profile = self.pipeline.start(self.config)
            
            # Get intrinsics and verify everything
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            
            # Store original intrinsics
            self.depth_intrinsics_original = depth_profile.get_intrinsics()
            self.color_intrinsics = color_profile.get_intrinsics()
            
            # After alignment, depth uses color intrinsics for 3D projection
            # This is the key fix - aligned depth should use color intrinsics
            self.intrinsics = self.color_intrinsics  # For 3D projection of aligned depth
            self.depth_intrinsics = self.color_intrinsics  # Aligned depth uses color intrinsics
            
            self.depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            
            # Apply defaults first if requested
            if self.use_viewer_defaults:
                self.apply_viewer_defaults()
            
            # Trigger calibration if requested
            if self.trigger_calibration:
                if self.debug:
                    print("Triggering device calibration...")
                self.trigger_device_calibration()
            
            # Get current parameter values after defaults applied
            current_params = self.get_current_parameters()
            self.current_exposure = current_params.get('exposure')
            self.current_gain = current_params.get('gain')
            self.current_laser_power = current_params.get('laser_power')
            self.auto_exposure_enabled = current_params.get('auto_exposure', True)
            
            # Verify resolutions match expectations
            actual_depth_res = (depth_profile.width(), depth_profile.height())
            actual_color_res = (color_profile.width(), color_profile.height())
            
            if self.debug:
                print(f"Camera started successfully:")
                print(f"  Depth resolution: {actual_depth_res}")
                print(f"  Color resolution: {actual_color_res}")
                print(f"  Expected: ({self.width}, {self.height})")
                print(f"  Depth scale: {self.depth_scale}")
                print(f"  Using viewer defaults: {self.use_viewer_defaults}")
                print(f"")
                print(f"  INTRINSICS COMPARISON:")
                print(f"  Original Depth intrinsics:")
                print(f"    fx: {self.depth_intrinsics_original.fx:.2f}, fy: {self.depth_intrinsics_original.fy:.2f}")
                print(f"    ppx: {self.depth_intrinsics_original.ppx:.2f}, ppy: {self.depth_intrinsics_original.ppy:.2f}")
                print(f"  Color intrinsics (used for aligned depth):")
                print(f"    fx: {self.color_intrinsics.fx:.2f}, fy: {self.color_intrinsics.fy:.2f}")
                print(f"    ppx: {self.color_intrinsics.ppx:.2f}, ppy: {self.color_intrinsics.ppy:.2f}")
                print(f"  Intrinsics difference:")
                print(f"    fx diff: {abs(self.depth_intrinsics_original.fx - self.color_intrinsics.fx):.2f}")
                print(f"    fy diff: {abs(self.depth_intrinsics_original.fy - self.color_intrinsics.fy):.2f}")
                print(f"    ppx diff: {abs(self.depth_intrinsics_original.ppx - self.color_intrinsics.ppx):.2f}")
                print(f"    ppy diff: {abs(self.depth_intrinsics_original.ppy - self.color_intrinsics.ppy):.2f}")
                print(f"")
                
                # Print parameter ranges
                ranges = self.get_parameter_ranges()
                if ranges:
                    print(f"  Parameter ranges:")
                    for param, (min_val, max_val, step) in ranges.items():
                        print(f"    {param}: {min_val}-{max_val} (step: {step})")
                
                # Print current parameters
                print(f"  Current parameters:")
                for param, value in current_params.items():
                    print(f"    {param}: {value}")
            
            # Verify we got what we asked for
            if actual_depth_res != (self.width, self.height):
                print(f"WARNING: Got depth {actual_depth_res}, expected ({self.width}, {self.height})")
            if actual_color_res != (self.width, self.height):
                print(f"WARNING: Got color {actual_color_res}, expected ({self.width}, {self.height})")
            
            # Initialize post-processing filters
            if self.use_viewer_defaults:
                self._init_viewer_quality_filters()
            else:
                self._init_minimal_filters()
            
            # Start processing thread
            self._running = True
            self._thread = threading.Thread(target=self._process_frames, daemon=True)
            self._thread.start()
            
            return True

        except RuntimeError as e:
            print(f"Failed to start camera: {e}")
            return False

    def _apply_post_processing(self, depth_frame):
        """Apply post-processing pipeline to depth frame"""
        if depth_frame is None:
            return None
            
        try:
            processed_frame = depth_frame
            
            if self.use_viewer_defaults:
                # Full RealSense Viewer quality pipeline
                
                # Apply decimation if enabled (reduces resolution but improves performance)
                if self.decimation_filter:
                    processed_frame = self.decimation_filter.process(processed_frame)
                
                # Apply threshold filter (remove too close/far values)
                if self.threshold_filter:
                    processed_frame = self.threshold_filter.process(processed_frame)
                
                # Apply temporal filter (reduces noise over time)
                if self.temporal_filter:
                    processed_frame = self.temporal_filter.process(processed_frame)
                
                # Apply spatial filter (reduces noise within frame)
                if self.spatial_filter:
                    processed_frame = self.spatial_filter.process(processed_frame)
                
                # Apply hole filling filter (fills remaining holes)
                if self.hole_filling_filter:
                    processed_frame = self.hole_filling_filter.process(processed_frame)
                    
            else:
                # Minimal processing (original behavior)
                if self.spatial_filter:
                    processed_frame = self.spatial_filter.process(processed_frame)
                
                if self.hole_filling_filter:
                    processed_frame = self.hole_filling_filter.process(processed_frame)
            
            return processed_frame
            
        except Exception as e:
            if self.debug:
                print(f"Error in post-processing: {e}")
            return depth_frame  # Return original on error

    def _process_frames(self):
        """Frame processing with full post-processing pipeline"""
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)
                
                # Get depth frame with post-processing
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    # Apply full post-processing pipeline
                    processed_frame = self._apply_post_processing(depth_frame)
                    
                    # Convert to numpy
                    depth_image = np.asanyarray(processed_frame.get_data())
                    
                    # Add to buffer
                    with self._buffer_lock:
                        self._depth_buffer.append(depth_image)

                if not self._frame_queue.full():
                    self._frame_queue.put(aligned_frames)
                else:
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put(aligned_frames)
                    except queue.Empty:
                        pass
                        
                if not hasattr(self, 'frame_count'):
                    self.frame_count = 0
                self.frame_count += 1

            except RuntimeError as e:
                if self.debug:
                    print(f"Frame acquisition error: {e}")
                time.sleep(0.1)

    def get_frames(self, use_averaging=True):
        """Get frames with guaranteed correct resolution and apply orientation correction"""
        try:
            aligned_frames = self._frame_queue.get(timeout=1.0)

            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                return None
            
            color_image = np.asanyarray(color_frame.get_data())
            
            if use_averaging:
                depth_image = self._get_averaged_depth()
                if depth_image is None:
                    depth_frame = aligned_frames.get_depth_frame()
                    if depth_frame:
                        processed_frame = self._apply_post_processing(depth_frame)
                        depth_image = np.asanyarray(processed_frame.get_data())
                    else:
                        return None
            else:
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    processed_frame = self._apply_post_processing(depth_frame)
                    depth_image = np.asanyarray(processed_frame.get_data())
                else:
                    return None

            # Apply orientation correction to both color and depth images
            color_image_corrected = self._rotate_image(color_image)
            depth_image_corrected = self._rotate_image(depth_image)
            
            # Save orientation correction results if enabled and rotation was applied
            if (self.save_orientation_results and self.camera_orientation != 0 and 
                self.frame_count_for_saving % 10 == 0):  # Save every 10th frame to avoid too many files
                self._save_orientation_results(color_image, color_image_corrected, 
                                             depth_image, depth_image_corrected)
            
            self.frame_count_for_saving += 1
            return color_image_corrected, depth_image_corrected

        except queue.Empty:
            return None, None

    def visualize(self, use_averaging=False):
        """Advanced visualization with multiple modes and interactive features"""
        frames = self.get_frames(use_averaging=use_averaging)
        if not frames or not self.depth_scale:
            return
        
        color_image, depth_image = frames
        
        # Convert BGR to RGB for consistent color representation with ZED camera
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.current_color_image = color_image.copy()  # Keep original BGR for saving
        self.current_depth_image = depth_image.copy()
        
        try:
            if self.display_mode == 0:
                # Background removal mode (original behavior)
                clipping_distance = self.clipping_distance_m / self.depth_scale
                depth_image_3d = np.dstack((depth_image,) * 3)
                viz_left = np.where(
                    (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                    153,  # Grey color
                    color_image_rgb
                )
                
                # Simple depth colorization
                depth_normalized = np.clip(depth_image.astype(np.float32) * self.depth_scale * 50, 0, 255).astype(np.uint8)
                viz_right = cv2.applyColorMap(depth_normalized, self.colormap)
                # Convert depth colormap from BGR to RGB to match
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
                # Convert depth colormap from BGR to RGB to match
                viz_right = cv2.cvtColor(viz_right, cv2.COLOR_BGR2RGB)
                
            else:  # display_mode == 2
                # Raw depth values mode
                viz_left = color_image_rgb.copy()
                
                # Use raw depth values with clipping
                depth_clipped = np.clip(depth_image.astype(np.float32) * self.depth_scale * 1000, 0, 5000)  # mm
                depth_norm = (depth_clipped / 5000 * 255).astype(np.uint8)
                viz_right = cv2.applyColorMap(depth_norm, self.colormap)
                # Convert depth colormap from BGR to RGB to match
                viz_right = cv2.cvtColor(viz_right, cv2.COLOR_BGR2RGB)
            
            # Mark invalid regions in depth image
            invalid_mask = depth_image == 0
            viz_right[invalid_mask] = [0, 0, 0]
            
            # Create side-by-side display
            combined = np.hstack([viz_left, viz_right])
            
            # Add info overlay
            combined = self.display_info_overlay(combined)
            
            # Convert back to BGR for OpenCV display (OpenCV expects BGR)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            
            # Display the image
            cv2.imshow('RealSense Parameter Tuning', combined_bgr)
            
        except Exception as e:
            if self.debug:
                print(f"Visualization error: {e}")
            # Fallback to simple display
            simple_combined = np.hstack([color_image, cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)])
            cv2.imshow('RealSense Parameter Tuning', simple_combined)

    def save_data(self):
        """Save current depth map and RGB image."""
        if self.current_color_image is None or self.current_depth_image is None:
            print("No current frames to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        output_dir = "realsense_captures"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save RGB image (convert BGR to RGB for proper saving)
        rgb_filename = f"{output_dir}/rgb_{timestamp}.png"
        rgb_image = cv2.cvtColor(self.current_color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        # Save raw depth data as numpy array
        depth_raw_filename = f"{output_dir}/depth_raw_{timestamp}.npy"
        np.save(depth_raw_filename, self.current_depth_image)
        
        # Save colorized depth image (ensure consistent RGB format)
        depth_colored = self.create_depth_colormap(self.current_depth_image)
        # Convert depth colormap to RGB format for consistency with ZED
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_colored_filename = f"{output_dir}/depth_colored_{timestamp}.png"
        cv2.imwrite(depth_colored_filename, cv2.cvtColor(depth_colored_rgb, cv2.COLOR_RGB2BGR))
        
        # Save depth in millimeters as 16-bit PNG
        if self.depth_scale:
            depth_mm = (self.current_depth_image * self.depth_scale * 1000).astype(np.uint16)
            depth_mm_filename = f"{output_dir}/depth_mm_{timestamp}.png"
            cv2.imwrite(depth_mm_filename, depth_mm)
        
        # Save current parameters
        params = self.get_current_parameters()
        params_filename = f"{output_dir}/params_{timestamp}.txt"
        with open(params_filename, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Camera settings:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
        
        print(f"Data saved to {output_dir}/ with timestamp {timestamp}")
        print(f"  RGB image saved in proper color order (matching ZED format)")
        print(f"  Camera parameters saved to {params_filename}")

    def run_advanced_visualization(self, use_averaging=False):
        """Run interactive visualization with keyboard controls"""
        if not self._running:
            print("Camera not started!")
            return
        
        print("\nStarting advanced visualization...")
        print("Controls:")
        print("  'q' or ESC: Quit")
        print("  's': Save current data")
        print("  'd': Toggle display mode (BG removal -> Normalized -> Raw)")
        print("  'c': Cycle through colormaps")
        print("  '+'/'-': Adjust clipping distance")
        print("  Click on image: Print depth value")
        print()
        
        # Set up OpenCV window
        cv2.namedWindow("RealSense Advanced Visualization", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("RealSense Advanced Visualization", self.mouse_callback)
        
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
                    self.clipping_distance_m = min(5.0, self.clipping_distance_m + 0.1)
                    print(f"Clipping distance: {self.clipping_distance_m:.1f}m")
                elif key == ord('-'):
                    self.clipping_distance_m = max(0.1, self.clipping_distance_m - 0.1)
                    print(f"Clipping distance: {self.clipping_distance_m:.1f}m")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()
            print("Advanced visualization stopped")

    def get_closest_blob(self, use_averaging=False):
        """Simple closest blob detection"""
        frames = self.get_frames(use_averaging=use_averaging)
        if frames is None or self.depth_scale is None:
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

    def get_point_cloud(self, use_averaging=False, manual_calculation=False):
        """
        Get point cloud from depth data
        
        Args:
            use_averaging: Whether to use averaged depth frames
            manual_calculation: If True, calculate manually using intrinsics.
                               If False, use RealSense built-in point cloud.
        
        Returns:
            numpy array of 3D points (N, 3) where each row is [x, y, z]
        """
        try:
            if manual_calculation:
                return self._get_point_cloud_manual(use_averaging)
            else:
                return self._get_point_cloud_realsense(use_averaging)
        except Exception as e:
            if self.debug:
                print(f"Error getting point cloud: {e}")
            return np.array([])

    def _get_point_cloud_realsense(self, use_averaging=False):
        """Get point cloud using RealSense built-in calculation"""
        try:
            aligned_frames = self._frame_queue.get(timeout=1.0)
            depth_frame = aligned_frames.get_depth_frame()
            
            if not depth_frame:
                return np.array([])
            
            # If using averaging, we need to create a synthetic frame
            # For simplicity with RealSense point cloud, use the current frame
            if use_averaging:
                # Apply post-processing to current frame
                processed_frame = self._apply_post_processing(depth_frame)
            else:
                processed_frame = self._apply_post_processing(depth_frame)
            
            # Calculate point cloud using RealSense
            points = self.point_cloud.calculate(processed_frame)
            pts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # Filter out zero/invalid points
            valid_mask = ~np.all(pts == 0, axis=1)
            pts_filtered = pts[valid_mask]
            
            return pts_filtered
            
        except Exception as e:
            if self.debug:
                print(f"RealSense point cloud failed: {e}, falling back to manual calculation")
            return self._get_point_cloud_manual(use_averaging)

    def _get_point_cloud_manual(self, use_averaging=False):
        """Manually calculate point cloud using camera intrinsics and depth data"""
        try:
            frames = self.get_frames(use_averaging=use_averaging)
            if not frames or not self.intrinsics or not self.depth_scale:
                return np.array([])
            
            color_image, depth_image = frames
            h, w = depth_image.shape
            
            # Create coordinate grids
            i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
            
            # Convert depth to meters
            depth_m = depth_image.astype(np.float32) * self.depth_scale
            
            # Filter valid depths (> 0 and < reasonable max distance)
            valid_mask = (depth_m > 0.1) & (depth_m < 5.0)  # 10cm to 5m range
            
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

    def get_point_cloud_with_colors(self, use_averaging=False):
        """
        Get point cloud with corresponding RGB colors
        
        Returns:
            Tuple of (points, colors) where:
            - points: (N, 3) array of [x, y, z] coordinates
            - colors: (N, 3) array of [r, g, b] values (0-255) - consistent RGB format
        """
        try:
            frames = self.get_frames(use_averaging=use_averaging)
            if not frames or not self.intrinsics or not self.depth_scale:
                return np.array([]), np.array([])
            
            color_image, depth_image = frames
            h, w = depth_image.shape
            
            # Create coordinate grids
            i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
            
            # Convert depth to meters
            depth_m = depth_image.astype(np.float32) * self.depth_scale
            
            # Filter valid depths
            valid_mask = (depth_m > 0.1) & (depth_m < 5.0)
            
            if np.sum(valid_mask) == 0:
                return np.array([]), np.array([])
            
            # Get valid coordinates, depths, and colors
            i_valid = i[valid_mask]
            j_valid = j[valid_mask]
            depth_valid = depth_m[valid_mask]
            
            # Get corresponding colors - RealSense outputs BGR, convert to RGB for consistency with ZED
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
                if depth_value_raw > 0 and self.depth_scale:
                    depth_value_m = depth_value_raw * self.depth_scale
                    print(f"Depth at ({actual_x}, {y}): {depth_value_raw} units ({depth_value_m*1000:.1f} mm, {depth_value_m:.3f} m)")
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
        
        # Get current parameters
        current_params = self.get_current_parameters()
        
        processing_mode = "Viewer Quality" if self.use_viewer_defaults else "Minimal"
        
        info_text = [
            f"RealSense: {self.width}x{self.height} @ {self.fps}fps",
            f"Processing: {processing_mode}",
            f"Mode: {mode_name}",
            f"Colormap: {colormap_name}",
            f"Clip: {self.clipping_distance_m:.1f}m",
            f"Mouse: ({self.mouse_x}, {self.mouse_y})",
            f"Frame: {getattr(self, 'frame_count', 0)}",
            "",  # Separator
            "Camera Parameters:",
            f"Exposure: {current_params.get('exposure', 'N/A')} μs",
            f"Gain: {current_params.get('gain', 'N/A')}",
            f"Laser Power: {current_params.get('laser_power', 'N/A')}",
            f"Auto Exposure: {current_params.get('auto_exposure', 'N/A')}",
            f"Visual Preset: {current_params.get('visual_preset', 'N/A')}",
            "",  # Separator
            "Controls:",
            "q=quit, s=save, d=mode, c=colormap, +/-=clip",
            "e/r=exposure ±1000μs, g/t=gain ±16",
            "l/y=laser ±30, a=auto exposure toggle",
            "v=toggle processing mode"
        ]
        
        # Add text overlay with background for better readability
        for i, text in enumerate(info_text):
            if text == "":  # Skip empty lines but maintain spacing
                continue
            y_pos = 30 + i * 22
            # Add background rectangle
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(image, (5, y_pos - 16), (15 + text_size[0], y_pos + 4), (0, 0, 0), -1)
            # Add text
            cv2.putText(image, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        return image

    def toggle_processing_mode(self):
        """Toggle between viewer quality and minimal processing"""
        self.use_viewer_defaults = not self.use_viewer_defaults
        
        if self.use_viewer_defaults:
            self._init_viewer_quality_filters()
            self.apply_viewer_defaults()
            if self.debug:
                print("Switched to Viewer Quality processing")
        else:
            self._init_minimal_filters()
            if self.debug:
                print("Switched to Minimal processing")

    def handle_parameter_adjustment(self, key):
        """Handle keyboard input for parameter adjustment"""
        current_params = self.get_current_parameters()
        ranges = self.get_parameter_ranges()
        
        # Processing mode toggle
        if key == ord('v'):  # Toggle processing mode
            self.toggle_processing_mode()
        
        # Exposure controls: e/r (decrease/increase)
        elif key == ord('e'):  # Decrease exposure
            if 'exposure' in current_params and ranges and 'exposure' in ranges:
                new_exposure = max(ranges['exposure'][0], current_params['exposure'] - 1000)
                self.set_exposure(new_exposure)
        elif key == ord('r'):  # Increase exposure  
            if 'exposure' in current_params and ranges and 'exposure' in ranges:
                new_exposure = min(ranges['exposure'][1], current_params['exposure'] + 1000)
                self.set_exposure(new_exposure)
        
        # Gain controls: g/t (decrease/increase)
        elif key == ord('g'):  # Decrease gain
            if 'gain' in current_params and ranges and 'gain' in ranges:
                new_gain = max(ranges['gain'][0], current_params['gain'] - 16)
                self.set_gain(new_gain)
        elif key == ord('t'):  # Increase gain
            if 'gain' in current_params and ranges and 'gain' in ranges:
                new_gain = min(ranges['gain'][1], current_params['gain'] + 16)
                self.set_gain(new_gain)
        
        # Laser power controls: l/y (decrease/increase)
        elif key == ord('l'):  # Decrease laser power
            if 'laser_power' in current_params and ranges and 'laser_power' in ranges:
                new_laser = max(ranges['laser_power'][0], current_params['laser_power'] - 30)
                self.set_laser_power(new_laser)
        elif key == ord('y'):  # Increase laser power
            if 'laser_power' in current_params and ranges and 'laser_power' in ranges:
                new_laser = min(ranges['laser_power'][1], current_params['laser_power'] + 30)
                self.set_laser_power(new_laser)
        
        # Auto exposure toggle
        elif key == ord('a'):  # Toggle auto exposure
            self.toggle_auto_exposure()

    # ... [rest of the methods remain the same - visualize, run_parameter_tuning, etc.]

    def run_parameter_tuning(self, use_averaging=False):
        """Run interactive parameter tuning mode"""
        if not self._running:
            print("Camera not started!")
            return
        
        print("\nStarting parameter tuning mode...")
        print("Controls:")
        print("  'q' or ESC: Quit")
        print("  's': Save current data and parameters")
        print("  'd': Toggle display mode (BG removal -> Normalized -> Raw)")
        print("  'c': Cycle through colormaps")
        print("  '+'/'-': Adjust clipping distance")
        print("  'e'/'r': Decrease/Increase exposure by 1000μs")
        print("  'g'/'t': Decrease/Increase gain by 16")
        print("  'l'/'y': Decrease/Increase laser power by 30")
        print("  'a': Toggle auto exposure")
        print("  'v': Toggle processing mode (Viewer Quality <-> Minimal)")
        print("  Click on image: Print depth value")
        print()
        
        # Print initial parameter values and ranges
        current_params = self.get_current_parameters()
        ranges = self.get_parameter_ranges()
        
        print("Current parameters:")
        for param, value in current_params.items():
            print(f"  {param}: {value}")
        
        if ranges:
            print("\nValid ranges:")
            for param, (min_val, max_val, step) in ranges.items():
                print(f"  {param}: {min_val}-{max_val} (step: {step})")
        print()
        
        processing_mode = "Viewer Quality" if self.use_viewer_defaults else "Minimal"
        print(f"Current processing mode: {processing_mode}")
        print()
        
        # Set up OpenCV window
        cv2.namedWindow("RealSense Parameter Tuning", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("RealSense Parameter Tuning", self.mouse_callback)
        
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
                    self.clipping_distance_m = min(5.0, self.clipping_distance_m + 0.1)
                    print(f"Clipping distance: {self.clipping_distance_m:.1f}m")
                elif key == ord('-'):
                    self.clipping_distance_m = max(0.1, self.clipping_distance_m - 0.1)
                    print(f"Clipping distance: {self.clipping_distance_m:.1f}m")
                else:
                    # Handle parameter adjustments
                    self.handle_parameter_adjustment(key)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()
            print("Parameter tuning stopped")

    # ... [include all other methods like visualize, save_data, point cloud methods, etc.]

    def stop(self):
        """Stop camera"""
        self._running = False
        if self._thread:
            self._thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()
        
        if self.debug:
            print("Camera stopped")


def test_viewer_quality():
    """Test function with RealSense Viewer quality defaults"""
    print("=" * 60)
    print("RealSense Viewer Quality Test")
    print("=" * 60)
    print()
    
    # Create camera with viewer defaults enabled (using actual viewer resolution)
    camera = Camera(
        width=640,  # RealSense Viewer default resolution
        height=480, 
        fps=30, 
        depth_averaging_frames=3,
        debug=True,
        use_viewer_defaults=True  # This is the key setting!
    )
    
    if not camera.start():
        print("Failed to start camera!")
        return
    
    print("\nCamera started with RealSense Viewer quality settings!")
    print("Resolution: 848x480 (matching RealSense Viewer)")
    
    # Wait for camera to stabilize
    print("Stabilizing camera...")
    time.sleep(3)  # Give auto-exposure time to adjust
    
    # Test a few frames
    print("\nTesting frame acquisition...")
    for i in range(3):
        frames = camera.get_frames()
        if frames:
            color, depth = frames
            print(f"Frame {i}: Color {color.shape}, Depth {depth.shape}")
            valid_pixels = np.sum(depth > 0)
            total_pixels = depth.size
            print(f"  Valid depth pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("VIEWER QUALITY VISUALIZATION")
    print("=" * 60)
    print()
    print("You should now see much better depth quality!")
    print("Controls:")
    print("  'v': Toggle between Viewer Quality and Minimal processing")
    print("  's': Save current data")
    print("  'q': Quit")
    print()
    
    try:
        camera.run_parameter_tuning(use_averaging=False)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera.stop()
        print("\nViewer quality test completed")


def test_both_modes():
    """Test function comparing minimal vs viewer quality"""
    print("=" * 60)
    print("RealSense Processing Mode Comparison")
    print("=" * 60)
    print()
    
    # Start with minimal processing (using viewer resolution)
    camera = Camera(
        width=640,  # RealSense Viewer default resolution
        height=480, 
        fps=30, 
        debug=True,
        use_viewer_defaults=True  # Start with minimal
    )
    
    if not camera.start():
        print("Failed to start camera!")
        return
    
    print("\nCamera started in MINIMAL processing mode")
    print("Resolution: 848x480 (matching RealSense Viewer)")
    print("Press 'v' to toggle to Viewer Quality mode and see the difference!")
    print()
    
    try:
        camera.run_parameter_tuning(use_averaging=True)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera.stop()
        print("\nComparison test completed")


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Viewer Quality (recommended)")
    print("2. Comparison Mode (toggle between minimal and viewer quality)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_both_modes()
    else:
        test_viewer_quality()