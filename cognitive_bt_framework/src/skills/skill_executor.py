#!/usr/bin/env python3
import cv2
import numpy as np
import asyncio
from typing import Optional, Dict, Any, Tuple, List
import math
import threading
import time
import sys
import logging
import torch
from datetime import datetime
import functools
import uuid
import concurrent.futures
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
from matplotlib.gridspec import GridSpec
import traceback
import json
from scipy.spatial.transform import Rotation
import math

# Import the Camera classes from cognitive_bt_framework
from cognitive_bt_framework.src.vision.zed_camera import Camera as ZedCamera
from cognitive_bt_framework.src.vision.realsense import Camera as RealSenseCamera
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig, get_alpha_id, ObjectInfo
from cognitive_bt_framework.src.skills import SkillGenerator, SkillHandler, ExecutableAction
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI
from cognitive_bt_framework.src.llm_interface.llm_interface_claude import LLMInterfaceClaude

# Import the CuRobo motion planner
from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface import CuRoboMotionPlanner 

# Import the updated stereo ArUco detector


DETECTION_RETRIES = 10

class DebugVisualizer:
    def __init__(self):
        # Create figure and axes
        self.fig = plt.figure(figsize=(20, 5))  # Made wider to accommodate more plots
        gs = GridSpec(1, 4, figure=self.fig)  # Added one more subplot
        
        # Create four subplots
        self.ax_raw = self.fig.add_subplot(gs[0, 0])
        self.ax_sam = self.fig.add_subplot(gs[0, 1])
        self.ax_poi = self.fig.add_subplot(gs[0, 2])
        self.ax_transform = self.fig.add_subplot(gs[0, 3])  # New subplot for transform visualization
        
        # Initialize image display objects
        self.im_raw = None
        self.im_sam = None
        self.im_poi = None
        self.im_transform = None
        
        # Set titles
        self.ax_raw.set_title('Camera Input')
        self.ax_sam.set_title('SAM Segmentation')
        self.ax_poi.set_title('Points of Interest')
        self.ax_transform.set_title('Transform Calibration')
        
        # Turn off axes
        for ax in [self.ax_raw, self.ax_sam, self.ax_poi, self.ax_transform]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.draw()
        self.fig.canvas.flush_events()
        
    def update(self, raw_img=None, sam_img=None, poi_img=None, transform_img=None):
        """Update visualization with new images"""
        updated = False
        
        if raw_img is not None:
            if self.im_raw is None:
                self.im_raw = self.ax_raw.imshow(raw_img)
            else:
                self.im_raw.set_data(raw_img)
            updated = True
                
        if sam_img is not None:
            if self.im_sam is None:
                self.im_sam = self.ax_sam.imshow(sam_img)
            else:
                self.im_sam.set_data(sam_img)
            updated = True
                
        if poi_img is not None:
            if self.im_poi is None:
                self.im_poi = self.ax_poi.imshow(poi_img)
            else:
                self.im_poi.set_data(poi_img)
            updated = True
                
        if transform_img is not None:
            if self.im_transform is None:
                self.im_transform = self.ax_transform.imshow(transform_img)
            else:
                self.im_transform.set_data(transform_img)
            updated = True
         
        if updated:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


class DirectSkillExecutor:
    def __init__(self, robot_ip="192.168.1.224", camera_params=None, show_debug_windows=True,
                 calibrate_transform=True, use_zed_camera=True, fast_mode=True,
                 depth_capture_delay=0.0, llm_query_logger=None):
        """
        Initialize the direct skill executor with configurable camera selection

        Args:
            robot_ip: IP address of the xArm robot
            camera_params: Camera configuration parameters
            show_debug_windows: Whether to show debug visualization windows
            calibrate_transform: Whether to perform camera-to-robot transform calibration on startup
            use_zed_camera: If True, use ZED camera as primary. If False, use RealSense camera as primary
            fast_mode: If True, skip extensive verification and use minimal debugging
            depth_capture_delay: Delay in seconds before capturing depth data for skill execution
            llm_query_logger: Optional LLMQueryLogger instance for logging LLM queries and responses
        """
        self.llm_query_logger = llm_query_logger
        self.use_zed_camera = use_zed_camera
        camera_type = "ZED" if use_zed_camera else "RealSense"
        print(f'Starting direct skill execution system initialization with {camera_type} camera')
        
        # Initialize logging
        self.setup_logging()
        
        # Set configuration parameters
        self.camera_params = camera_params or {
            'width': 640,
            'height': 480,
            'fps': 30
        }
        
        # Set flags
        self.show_debug_windows = show_debug_windows and not fast_mode  # Disable debug in fast mode
        self.calibrate_transform = calibrate_transform and not fast_mode  # Skip calibration in fast mode
        self.fast_mode = fast_mode
        self.depth_capture_delay = depth_capture_delay
        self.action_timeout = 240
        
        # Transform calibration (only relevant when using ZED camera)
        self.zed_to_robot_transform = None
        self.transform_confidence = 0.0
        self.stereo_detector = None
        
        # Initialize valid skills
        self.valid_skills = {
            'open': 1, 'close': 1, 'pickup': 1,
            'place': 2, 'switchon': 1, 'switchoff': 1,
            'twist': 1, 'detect_object': 1, "drop": 1  # Add detect_object and drop to valid skills
        }
        
        # Initialize image handling
        self.latest_color_image = None
        self.latest_depth_image = None
        self.execution_start_time = None
        self.perception_system = None
        self.current_object_info = None  # Store current object detection info for pivot calculations
        self.current_interaction_point_pixel = None  # Store pixel coordinates of interaction point
        self.current_interaction_point_world = None  # Store 3D world coordinates of interaction point
        
        # Camera references (will be set based on use_zed_camera flag)
        
        # Data recording integration
        self.data_recorder = None
        self.main_camera = None
        self.secondary_camera = None
        
        # Initialize debug visualizer (optional for faster startup)
        if self.show_debug_windows:
            try:
                self.visualizer = DebugVisualizer()
                print("Debug visualizer initialized")
            except Exception as e:
                print(f"Warning: Debug visualizer failed to initialize: {e}")
                self.show_debug_windows = False
        
        # Initialize CuRobo motion planner first (for robot camera access)
        print("Initializing CuRobo motion planner...")
        self.motion_planner = CuRoboMotionPlanner(robot_ip=robot_ip)
        print("CuRobo motion planner initialized")
        
        # Setup cameras
        self.setup_cameras()
        static_transform = None
        # Perform transform calibration if requested (only for ZED camera)
        if self.calibrate_transform and self.use_zed_camera:
            # Try to load existing calibration first
            if self.load_existing_calibration():
                print("✅ Loaded existing calibration - skipping calibration")
                static_transform = self.zed_to_robot_transform
            else:
                print("⚠️  No existing calibration found - performing new calibration")
                self.calibrate_zed_to_robot_transform()
                static_transform = self.zed_to_robot_transform
        elif not self.use_zed_camera:
            print("Using RealSense camera - transform calibration not needed (camera is robot-mounted)")
        # Initialize CuRobo motion planner first (for robot camera access)
        
        print("Initializing CuRobo motion planner...")
        # self.motion_planner = CuRoboMotionPlanner(robot_ip=robot_ip, static_camera_tf=static_transform)
        print("CuRobo motion planner initialized")
        # Initialize perception system with selected camera
        self.initialize_perception_system()
        camera_type = "ZED" if self.use_zed_camera else "RealSense"
        print(f'Perception system initialized with {camera_type} camera')
        
        # Skip verification in fast mode
        if not self.fast_mode:
            if not self.verify_camera_setup():
                raise RuntimeError("Camera setup verification failed")
        else:
            print("⚡ Fast mode: Skipping camera verification")
        
        print('Direct skill executor initialization completed')
        self.motion_planner.open_gripper()
    
    def setup_logging(self):
        """Configure logging to console"""
        camera_type = "zed" if self.use_zed_camera else "realsense"
        self.logger = logging.getLogger(f'direct_skill_execution_{camera_type}')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(console_handler)
            
    def verify_camera_setup(self):
        """Quick camera setup verification"""
        try:
            print("=== Quick Camera Setup Check ===")
            
            # Test frame acquisition
            color_image, depth_image = self.get_latest_images(timeout=2.0)
            
            if color_image is None or depth_image is None:
                print("❌ Failed to get camera frames")
                return False
            
            # Basic dimension check
            expected_width = self.camera_params['width']
            expected_height = self.camera_params['height']
            
            if (color_image.shape[1] != expected_width or 
                color_image.shape[0] != expected_height):
                print(f"❌ Color image resolution mismatch!")
                return False
            
            print(f"✅ Camera setup verified - {expected_width}x{expected_height}")
            return True
            
        except Exception as e:
            print(f"❌ Camera verification failed: {e}")
            return False
    
    def setup_cameras(self):
        """Setup cameras based on the use_zed_camera flag"""
        print("Setting up cameras...")
        
        if self.use_zed_camera:
            # Setup ZED camera as main camera
            print("Initializing ZED camera (main camera)...")
            self.main_camera = ZedCamera(
                width=self.camera_params['width'],
                height=self.camera_params['height'],
                fps=self.camera_params['fps'],
                depth_averaging_frames=1,
                debug=False
            )
            
            # Start the ZED camera
            if not self.main_camera.start():
                raise RuntimeError("Failed to start ZED camera")
            print("ZED camera started successfully")
            
            # Setup RealSense camera as secondary (for transform calibration)
            if self.calibrate_transform and not self.fast_mode:
                print("Initializing RealSense camera (secondary for calibration)...")
                self.secondary_camera = RealSenseCamera(
                    width=self.camera_params['width'],
                    height=self.camera_params['height'],
                    fps=self.camera_params['fps'],
                    depth_averaging_frames=3,
                    debug=False
                )
                
                # Start the secondary camera
                if not self.secondary_camera.start():
                    print("Warning: Failed to start secondary RealSense camera")
                    print("Transform calibration will not be available")
                    self.secondary_camera = None
                else:
                    print("RealSense camera (secondary) started successfully")
                    # Set initial camera orientation based on current robot gripper position
                    initial_orientation = self._get_camera_orientation_from_robot()
                    self.secondary_camera.set_camera_orientation(initial_orientation)
            else:
                print("⚡ Fast mode: Skipping secondary camera setup")
                self.secondary_camera = None
                
        else:
            # Setup RealSense camera as main camera
            print("Initializing RealSense camera (main camera)...")
            self.main_camera = RealSenseCamera(
                width=self.camera_params['width'],
                height=self.camera_params['height'],
                fps=self.camera_params['fps'],
                depth_averaging_frames=3,
                debug=False
            )
            
            # Start the RealSense camera
            if not self.main_camera.start():
                raise RuntimeError("Failed to start RealSense camera")
            print("RealSense camera started successfully")
            
            # Set initial camera orientation based on current robot gripper position
            initial_orientation = self._get_camera_orientation_from_robot()
            self.main_camera.set_camera_orientation(initial_orientation)
            
            # No secondary camera needed when using RealSense as primary
            self.secondary_camera = None
        
        # Set convenient aliases for backward compatibility
        if self.use_zed_camera:
            self.camera = self.main_camera  # ZED camera
            self.robot_camera = self.secondary_camera  # RealSense camera
        else:
            self.camera = self.main_camera  # RealSense camera
            self.robot_camera = None  # No separate robot camera needed
    
    def update_camera_orientation(self, orientation=None):
        """
        Update the camera orientation based on robot gripper pitch rotation or manual input
        
        Args:
            orientation: Manual orientation override (0, 90, 180, 270). 
                        If None, reads from robot gripper pitch rotation.
        """
        if orientation is not None:
            # Manual override
            target_orientation = orientation
        else:
            # Read gripper orientation from robot interface
            target_orientation = self._get_camera_orientation_from_robot()
        
        # # Update orientation for RealSense cameras
        # if not self.use_zed_camera and hasattr(self.main_camera, 'set_camera_orientation'):
        #     self.main_camera.set_camera_orientation(target_orientation)
        #     if self.debug or True:  # Always log this important info
        #         print(f"Updated main camera orientation to {target_orientation} degrees")
        
        # if self.secondary_camera and hasattr(self.secondary_camera, 'set_camera_orientation'):
        #     self.secondary_camera.set_camera_orientation(target_orientation)
        #     if self.debug or True:  # Always log this important info
        #         print(f"Updated secondary camera orientation to {target_orientation} degrees")
    
    def _get_camera_orientation_from_robot(self):
        """
        Determine camera orientation from robot gripper pitch rotation
        
        Returns:
            int: Camera orientation in degrees (0, 90, 180, 270)
        """
        try:
            # Get current robot TCP pose (position and quaternion)
            pose_result = self.motion_planner.get_robot_tcp_pose()
            if pose_result is None:
                self.logger.warning("Could not get robot TCP pose, using default camera orientation")
                return 0
            
            position, quaternion = pose_result
            
            # Convert quaternion to scipy Rotation object
            # The quaternion format from robot might be [x,y,z,w] or [w,x,y,z]
            # Let's check the format and convert accordingly
            if len(quaternion) == 4:
                # Try both formats to see which one gives reasonable euler angles
                try:
                    # Assume quaternion is [x,y,z,w] format
                    rot = Rotation.from_quat(quaternion)
                    euler_xyz = rot.as_euler('xyz', degrees=True)
                    
                    # Extract pitch rotation (rotation about Y-axis)
                    pitch_degrees = euler_xyz[1]  # Y-axis rotation is pitch
                    
                    self.logger.info(f"Robot gripper euler angles (XYZ): {euler_xyz}")
                    self.logger.info(f"Gripper pitch: {pitch_degrees:.1f} degrees")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to convert quaternion {quaternion} to euler angles: {e}")
                    return 0
            else:
                self.logger.warning(f"Unexpected quaternion format: {quaternion}")
                return 0
            
            # Convert pitch angle to camera orientation
            # Camera orientation is the rotation needed to make image upright
            # This is opposite to the gripper rotation about pitch
            camera_orientation_degrees = self._pitch_to_camera_orientation(pitch_degrees)
            
            return camera_orientation_degrees
            
        except Exception as e:
            self.logger.error(f"Error getting camera orientation from robot: {str(e)}")
            return 0  # Default to normal orientation on error
    
    def _pitch_to_camera_orientation(self, pitch_degrees):
        """
        Convert gripper pitch rotation to camera orientation
        
        Args:
            pitch_degrees: Gripper pitch rotation in degrees
            
        Returns:
            int: Camera orientation (0, 90, 180, 270)
        """
        # Normalize pitch to -180 to 180 range
        pitch_normalized = ((pitch_degrees + 180) % 360) - 180
        
        # Map pitch ranges to camera orientations
        # These thresholds may need adjustment based on your specific setup
        if -45 <= pitch_normalized <= 45:
            camera_orientation = 0    # Normal orientation
        elif 45 < pitch_normalized <= 135:
            camera_orientation = 90   # 90 degrees clockwise
        elif pitch_normalized > 135 or pitch_normalized <= -135:
            camera_orientation = 180  # Upside down
        else:  # -135 < pitch_normalized <= -45
            camera_orientation = 270  # 90 degrees counterclockwise
        
        self.logger.info(f"Pitch {pitch_degrees:.1f}° -> Camera orientation {camera_orientation}°")
        
        return camera_orientation
    
    def calibrate_camera_orientation_mapping(self, pitch_thresholds=None):
        """
        Calibrate the mapping between gripper pitch and camera orientation
        
        Args:
            pitch_thresholds: Custom thresholds as [normal_max, clockwise_max, upside_down_max]
                            Default: [45, 135, 180] degrees
        """
        if pitch_thresholds is None:
            pitch_thresholds = [45, 135, 180]  # Default thresholds
        
        self.pitch_thresholds = pitch_thresholds
        print(f"Camera orientation mapping calibrated:")
        print(f"  Normal (0°): pitch within ±{pitch_thresholds[0]}°")
        print(f"  Clockwise 90°: pitch {pitch_thresholds[0]}° to {pitch_thresholds[1]}°")
        print(f"  Upside down (180°): pitch beyond ±{pitch_thresholds[1]}°")
        print(f"  Counterclockwise 270°: pitch -{pitch_thresholds[1]}° to -{pitch_thresholds[0]}°")
    
    def test_camera_orientation_detection(self):
        """
        Test the camera orientation detection by printing current robot state and camera orientation
        """
        try:
            pose_result = self.motion_planner.get_robot_tcp_pose()
            if pose_result is None:
                print("❌ Could not get robot TCP pose")
                return
            
            position, quaternion = pose_result
            
            # Convert to euler angles
            rot = Rotation.from_quat(quaternion)
            euler_xyz = rot.as_euler('xyz', degrees=True)
            pitch_degrees = euler_xyz[1]
            
            # Get predicted camera orientation
            camera_orientation = self._pitch_to_camera_orientation(pitch_degrees)
            
            print("=== Camera Orientation Detection Test ===")
            print(f"Robot TCP Position: {[f'{x:.3f}' for x in position]}")
            print(f"Robot TCP Quaternion: {[f'{x:.3f}' for x in quaternion]}")
            print(f"Euler angles (XYZ): {[f'{x:.1f}°' for x in euler_xyz]}")
            print(f"Gripper Pitch: {pitch_degrees:.1f}°")
            print(f"Predicted Camera Orientation: {camera_orientation}°")
            print("==========================================")
            
            return {
                'position': position,
                'quaternion': quaternion,
                'euler_xyz': euler_xyz,
                'pitch': pitch_degrees,
                'camera_orientation': camera_orientation
            }
            
        except Exception as e:
            print(f"❌ Error testing camera orientation detection: {str(e)}")
            return None
    
    def enable_orientation_image_saving(self, enable=True, save_dir=None):
        """
        Enable or disable saving of camera orientation correction images
        
        Args:
            enable: Whether to save orientation correction images
            save_dir: Custom directory for saving images (optional)
        """
        if not self.use_zed_camera and hasattr(self.main_camera, 'set_save_orientation_results'):
            self.main_camera.set_save_orientation_results(enable, save_dir)
            if enable:
                save_path = save_dir or self.main_camera.orientation_save_dir
                print(f"✅ Orientation image saving ENABLED for main camera")
                print(f"   Images will be saved to: {save_path}")
            else:
                print("❌ Orientation image saving DISABLED for main camera")
        
        if self.secondary_camera and hasattr(self.secondary_camera, 'set_save_orientation_results'):
            self.secondary_camera.set_save_orientation_results(enable, save_dir)
            if enable:
                save_path = save_dir or self.secondary_camera.orientation_save_dir
                print(f"✅ Orientation image saving ENABLED for secondary camera")
                print(f"   Images will be saved to: {save_path}")
            else:
                print("❌ Orientation image saving DISABLED for secondary camera")
    
    def save_current_orientation_images(self):
        """
        Manually save current frame's orientation correction images
        """
        success = False
        
        if not self.use_zed_camera and hasattr(self.main_camera, 'save_current_orientation_results'):
            if self.main_camera.save_current_orientation_results():
                print("✅ Main camera orientation images saved")
                success = True
            else:
                print("❌ Failed to save main camera orientation images")
        
        if self.secondary_camera and hasattr(self.secondary_camera, 'save_current_orientation_results'):
            if self.secondary_camera.save_current_orientation_results():
                print("✅ Secondary camera orientation images saved")
                success = True
            else:
                print("❌ Failed to save secondary camera orientation images")
        
        return success
    
    def calibrate_zed_to_robot_transform(self):
        """Calibrate the ZED-to-robot-base transform using simplified stereo ArUco detection"""
        if not self.use_zed_camera:
            print("Transform calibration only available when using ZED camera")
            return False
            
        if self.secondary_camera is None:
            print("Secondary camera (RealSense) not available - skipping transform calibration")
            return False
            
        print("=== Starting ZED-to-Robot Transform Calibration (Simplified Interface) ===")
        print("This will use ArUco markers to compute the transform from ZED camera to robot base frame")
        print("Make sure ArUco markers are visible to both cameras")
        
        try:
            # Get robot base to camera link transform
            robot_camera_position, robot_camera_rotation = self.motion_planner.get_camera_transform()
            T_robot_base_to_camera_link = self.create_transform_matrix(robot_camera_position, robot_camera_rotation)
            
            # Initialize simplified stereo ArUco detector
            # self.stereo_detector = StereoArucoDetector(
            #     robot_camera=self.secondary_camera,  # RealSense camera
            #     zed_camera=self.main_camera,         # ZED camera
            #     T_robot_base_to_camera_link=T_robot_base_to_camera_link,
            #     marker_size=0.05  # 5cm markers
            # )
            
            print("Simplified stereo ArUco detector initialized!")
            print("Looking for ArUco markers to calibrate transform...")
            print("Automatic calibration will accept transforms with >70% confidence")
            
            # Calibration loop (reduced attempts for faster startup)
            calibration_attempts = 0
            max_attempts = 20  # Reduced from 100 for faster startup
            
            while calibration_attempts < max_attempts:
                try:
                    # Get detection data and compute transform
                    robot_detection, zed_detection = self.stereo_detector.get_detection_data()
                    transform_matrix = self.stereo_detector.compute_transformation()
                    
                    # Create visualization if we have camera data
                    if robot_detection is not None and zed_detection is not None:
                        # Get current frames for visualization
                        robot_frames = self.secondary_camera.get_frames()
                        zed_frames = self.main_camera.get_frames()
                        
                        if robot_frames is not None and zed_frames is not None:
                            robot_color, _ = robot_frames
                            zed_color, _ = zed_frames
                            
                            # Get camera matrices for drawing
                            robot_K, robot_dist, zed_K, zed_dist = self.stereo_detector._get_camera_matrices()
                            
                            # Draw markers on images
                            robot_output = self.stereo_detector.draw_markers_on_image(
                                robot_color, robot_detection, robot_K, robot_dist, "Robot"
                            )
                            zed_output = self.stereo_detector.draw_markers_on_image(
                                zed_color, zed_detection, zed_K, zed_dist, "ZED"
                            )
                            
                            if self.show_debug_windows:
                                # Create side-by-side visualization
                                vis_height = max(robot_output.shape[0], zed_output.shape[0])
                                vis_width = robot_output.shape[1] + zed_output.shape[1]
                                transform_vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
                                
                                # Place robot camera image on left
                                transform_vis[:robot_output.shape[0], :robot_output.shape[1]] = robot_output
                                
                                # Place ZED image on right
                                transform_vis[:zed_output.shape[0], robot_output.shape[1]:] = zed_output
                                
                                # Add status text
                                status_text = f"Calibration - Attempt {calibration_attempts + 1}/{max_attempts}"
                                
                                # Get detection summary
                                summary = self.stereo_detector.get_detection_summary()
                                robot_markers = summary['robot_markers']
                                zed_markers = summary['zed_markers']
                                common_markers = summary['common_markers']
                                transform_available = summary['transform_available']
                                
                                if transform_available and transform_matrix is not None:
                                    # Estimate confidence based on detection quality
                                    confidence = 0.1# if len(common_markers) > 0 else 0.0
                                    status_text += f" | Transform: Available (conf: {confidence:.3f})"
                                    
                                    # If confidence is high enough, auto-accept
                                    if confidence > 0.8:
                                        print(f"High confidence transform found (conf: {confidence:.3f})")
                                        print("Auto-accepting calibration...")
                                        self.accept_calibration_simplified(transform_matrix, confidence)
                                        break
                                else:
                                    status_text += f" | R:{robot_markers} Z:{zed_markers} C:{len(common_markers)}"
                                
                                cv2.putText(transform_vis, status_text, (10, 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                
                                # Convert to RGB for matplotlib
                                transform_vis_rgb = cv2.cvtColor(transform_vis, cv2.COLOR_BGR2RGB)
                                self.visualizer.update(transform_img=transform_vis_rgb)
                    
                    # Check for successful transform
                    if transform_matrix is not None:
                        # Get detection summary for confidence estimation
                        summary = self.stereo_detector.get_detection_summary()
                        common_markers = summary['common_markers']
                        
                        if len(common_markers) > 0:
                            confidence = 0.8  # Simplified confidence estimation
                            print(f"Transform found with {len(common_markers)} common marker(s) (conf: {confidence:.3f})")
                            self.accept_calibration_simplified(transform_matrix, confidence)
                            break
                    
                    calibration_attempts += 1
                    time.sleep(0.1)  # Small delay
                    
                except KeyboardInterrupt:
                    print("Calibration interrupted by user")
                    break
                except Exception as e:
                    print(f"Error during calibration: {e}")
                    calibration_attempts += 1
            
            if calibration_attempts >= max_attempts:
                print("Maximum calibration attempts reached without finding suitable transform")
                return False
                
        except Exception as e:
            print(f"Transform calibration failed: {e}")
            import traceback
            print(traceback.format_exc())
            return False
        
        return self.zed_to_robot_transform is not None
    
    def accept_calibration_simplified(self, transform_matrix, confidence):
        """Accept the current calibration transform from simplified interface"""
        try:
            self.zed_to_robot_transform = transform_matrix.copy()
            self.transform_confidence = confidence
            
            print(f"Transform calibration accepted!")
            print(f"Confidence: {self.transform_confidence:.3f}")
            print(f"Transform matrix:\n{self.zed_to_robot_transform}")
            
            # Save calibration to file
            calibration_data = {
                'T_zed_to_robot_base': self.zed_to_robot_transform.tolist(),
                'transform_confidence': self.transform_confidence,
                'timestamp': time.time(),
                'calibration_method': 'simplified_stereo_aruco'
            }
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"zed_to_robot_calibration_simplified_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"Calibration saved to {filename}")
            
            # Configure motion planner with static camera transform
            self.configure_motion_planner_with_zed_transform()
            
            return True
                
        except Exception as e:
            print(f"Error accepting calibration: {e}")
            return False
    
    def configure_motion_planner_with_zed_transform(self):
        """Configure the motion planner to use the ZED camera transform"""
        if not self.use_zed_camera:
            print("Not using ZED camera - skipping ZED transform configuration")
            return
            
        if self.zed_to_robot_transform is None:
            print("No ZED transform available - motion planner will use default robot camera")
            return
            
        try:
            # Extract position and rotation from transform matrix
            position = self.zed_to_robot_transform[:3, 3]
            rotation_matrix = self.zed_to_robot_transform[:3, :3]
            rotation = Rotation.from_matrix(rotation_matrix)
            
            print("Configuring motion planner to use ZED camera transform...")
            print(f"ZED position in robot base frame: {position}")
            print(f"ZED orientation (quaternion): {rotation.as_quat()}")
            
            # Create a new motion planner instance with the static camera transform
            # The static_camera_tf parameter should be the transform from ZED to robot base
            self.motion_planner = CuRoboMotionPlanner(
                robot_ip=self.motion_planner.config.robot_ip,
                static_camera_tf=self.zed_to_robot_transform
            )
            
            print("Motion planner reconfigured with ZED camera transform")
            
        except Exception as e:
            print(f"Error configuring motion planner with ZED transform: {e}")
    
    def create_transform_matrix(self, position, rotation):
        """Create 4x4 transformation matrix from position and rotation"""
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = position
        return transform
    
    def load_existing_calibration(self, filename=None):
        """Load existing calibration from file (only relevant for ZED camera)"""
        if not self.use_zed_camera:
            print("Calibration loading only available when using ZED camera")
            return False
            
        if filename is None:
            # Look for the most recent calibration file
            import glob
            calibration_files = glob.glob("zed_to_robot_calibration_*.json")
            if not calibration_files:
                print("No existing calibration files found")
                return False
            filename = max(calibration_files)  # Get most recent
            
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.zed_to_robot_transform = np.array(calibration_data['T_zed_to_robot_base'])
            self.transform_confidence = calibration_data.get('transform_confidence', 0.0)
            
            print(f"Loaded calibration from {filename}")
            print(f"Confidence: {self.transform_confidence:.3f}")
            
            # Configure motion planner
            self.configure_motion_planner_with_zed_transform()
            
            return True
            
        except Exception as e:
            print(f"Error loading calibration from {filename}: {e}")
            return False
    
    def initialize_perception_system(self):
        """Initialize perception system with the selected camera matrix"""
        try:
            # Get camera matrix from the main camera
            camera_matrix = np.array([
                [self.main_camera.intrinsics.fx, 0, self.main_camera.intrinsics.ppx],
                [0, self.main_camera.intrinsics.fy, self.main_camera.intrinsics.ppy],
                [0, 0, 1]
            ])
            camera_type = "ZED" if self.use_zed_camera else "RealSense"
            print(f"Using camera matrix from {camera_type} camera")
            
            fs_config = FastSAMConfig(
                max_image_size=max(self.camera_params['height'], self.camera_params['width']),
                model_type="FastSAM-x",
                device="cuda" if torch.cuda.is_available() else "cpu",
                conf_threshold=0.9,
                iou_threshold=0.8,
                retina_masks=True,
                remove_small_regions=False,
                merge_overlapping=False,
                overlap_threshold=0.5,
                min_area=25.0,
                draw_borders=True
            )
            
            # Get depth scale from the main camera
            if self.use_zed_camera:
                depth_scale = getattr(self.main_camera, 'depth_scale', 1000.0)  # ZED depth scale
            else:
                depth_scale = getattr(self.main_camera, 'depth_scale', 1000.0)  # RealSense depth scale
            
            # Initialize PerceptionSystem with selected camera parameters
            self.perception_system = PerceptionSystem(
                fast_sam_config=fs_config,
                camera_matrix=camera_matrix,
                depth_scale=depth_scale,
                debug=True
            )
            
            # Initialize LLM and skill handler components
            llm_interface = LLMInterfaceOpenAI(query_logger=self.llm_query_logger)
            skill_generator = SkillGenerator(llm_interface=llm_interface)
            self.skill_handler = SkillHandler(skill_generator, self.perception_system)
            
            camera_type = "ZED" if self.use_zed_camera else "RealSense"
            self.logger.info(f'Perception system initialized with {camera_type} camera')
        
        except Exception as e:
            self.logger.error(f'Failed to initialize perception system: {str(e)}')
            raise
    
    def log_with_timestamp(self, level: str, message: str):
        """Log a message with timestamp and execution time if available"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        elapsed = f"[{self.get_elapsed_time():.2f}s]" if self.execution_start_time else ""
        full_message = f"{timestamp} {elapsed} - {message}"
        
        if level == 'error':
            self.logger.error(full_message)
        elif level == 'warn' or level == 'warning':
            self.logger.warning(full_message)
        else:
            self.logger.info(full_message)
    
    def get_elapsed_time(self):
        """Get elapsed time since execution start in seconds"""
        if self.execution_start_time is None:
            return 0.0
        return time.time() - self.execution_start_time
    
    def get_latest_images(self, timeout: float = 2.0):
        """Get latest images directly from the main camera"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            color_image, depth_image = self.main_camera.get_frames()
            
            if color_image is not None and depth_image is not None:
                self.latest_color_image = color_image
                self.latest_depth_image = depth_image
                
                if self.show_debug_windows:
                    self.update_visualizations(color_image=color_image)
                    
                return color_image, depth_image
                
            time.sleep(0.05)
        
        camera_type = "ZED" if self.use_zed_camera else "RealSense"
        self.logger.warning(f'Timeout waiting for {camera_type} camera images')
        return None, None
    
    def update_visualizations(self, color_image=None, sam_masks=None, poi_image=None, transform_image=None):
        """Update visualizations directly"""
        try:
            if hasattr(self, 'visualizer') and self.show_debug_windows:
                # Convert BGR to RGB for matplotlib if needed
                if color_image is not None:
                    raw_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    self.visualizer.update(raw_img=raw_img)
                    
                if sam_masks is not None:
                    if len(sam_masks.shape) == 3 and sam_masks.shape[2] == 3:
                        sam_img = cv2.cvtColor(sam_masks, cv2.COLOR_BGR2RGB)
                    else:
                        sam_img = sam_masks
                        if len(sam_img.shape) == 2:
                            sam_img = cv2.cvtColor(sam_img, cv2.COLOR_GRAY2RGB)
                    self.visualizer.update(sam_img=sam_img)
                    
                if poi_image is not None:
                    if len(poi_image.shape) == 3 and poi_image.shape[2] == 3:
                        poi_img = cv2.cvtColor(poi_image, cv2.COLOR_BGR2RGB)
                    else:
                        poi_img = poi_image
                        if len(poi_img.shape) == 2:
                            poi_img = cv2.cvtColor(poi_img, cv2.COLOR_GRAY2RGB)
                    self.visualizer.update(poi_img=poi_img)
                    
                if transform_image is not None:
                    self.visualizer.update(transform_img=transform_image)
                    
        except Exception as e:
            self.logger.error(f"Error updating visualizations: {str(e)}")
    
    def create_points_of_interest_visualization(self, color_image, object_info=None, points=None, target_points=[]):
        """Create visualization of points of interest on the image"""
        if color_image is None:
            return None
            
        vis_img = color_image.copy()
        
        # Draw object information if available
        if object_info is not None:
            # Draw bounding box
            if hasattr(object_info, 'bbox') and object_info.bbox is not None:
                x, y, w, h = object_info.bbox
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw mask if available
            if hasattr(object_info, 'mask') and object_info.mask is not None:
                mask_overlay = np.zeros_like(vis_img, dtype=np.uint8)
                mask_overlay[object_info.mask] = [0, 100, 0]
                vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.3, 0)
            
            # Draw center point
            if hasattr(object_info, 'pixel_pose') and object_info.pixel_pose is not None:
                center = object_info.pixel_pose
            elif hasattr(object_info, 'obj_center') and object_info.obj_center is not None:
                center = tuple(map(int, object_info.obj_center[:2]))
            else:
                x, y, w, h = object_info.bbox
                center = (x + w//2, y + h//2)
                
            cv2.drawMarker(vis_img, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            # Add label with name
            if hasattr(object_info, 'name'):
                alpha_id = object_info.alpha_id if hasattr(object_info, 'alpha_id') else ""
                label = f"{alpha_id}: {object_info.name}"
                cv2.putText(vis_img, label, (center[0] + 10, center[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw 2D points if provided
        if points is not None and len(points) > 0:
            num_points = len(points)
            colors = []
            for i in range(num_points):
                hue = int(180.0 * i / max(1, num_points - 1))
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                colors.append((int(color[0]), int(color[1]), int(color[2])))
            
            for i, pt in enumerate(points):
                if not isinstance(pt, (np.ndarray, list, tuple)) or len(pt) < 2:
                    continue
                    
                x, y = int(pt[0]), int(pt[1])
                
                if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                    color = colors[i % len(colors)]
                    cv2.circle(vis_img, (x, y), 5, (0,0,255), 1)
                    point_id = get_alpha_id(i+1)
                    cv2.putText(vis_img, point_id, (x + 5, y + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add title and camera type
        camera_type = "ZED" if self.use_zed_camera else "RealSense"
        cv2.putText(vis_img, f"Points of Interest ({camera_type} Camera)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add transform status (only relevant for ZED camera)
        if self.use_zed_camera:
            if self.zed_to_robot_transform is not None:
                transform_status = f"Transform: Calibrated (conf: {self.transform_confidence:.3f})"
                cv2.putText(vis_img, transform_status, (10, vis_img.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                transform_status = "Transform: Not Calibrated"
                cv2.putText(vis_img, transform_status, (10, vis_img.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            transform_status = "Transform: Robot-mounted (no calibration needed)"
            cv2.putText(vis_img, transform_status, (10, vis_img.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_img
    
    def validate_skill_command(self, skill: str, parameters: str) -> Tuple[bool, str, Optional[list]]:
        """Validate the skill command and its parameters"""
        if skill not in self.valid_skills:
            return False, f"Invalid skill: {skill}. Must be one of: {', '.join(self.valid_skills.keys())}", None
            
        param_list = parameters.strip().split(',')
        required_params = self.valid_skills[skill]
        
        if len(param_list) < required_params:
            return False, f"Skill '{skill}' requires {required_params} parameter(s), but got {len(param_list)}", None
            
        if skill == 'place' and len(param_list) != 2:
            return False, "Place skill requires exactly 2 parameters: object and destination", None
            
        if skill == 'twist' and len(param_list) != 1:
            return False, "Twist skill requires exactly 1 parameter: direction (clockwise or counterclockwise)", None
            
        return True, "", param_list
    
    def adjust_tcp_pose_for_depth_centering(self, target_position, search_radius_m=0.05, 
                                            depth_threshold_ratio=0.1, cluster_min_size=10):
        """
        Adjust TCP pose position using depth clustering and peak depth optimization to center 
        the grasp on the main object cluster within a configurable radius around the grasp point.
        
        Args:
            target_position: Original target position [x, y, z] in camera/world coordinates
            search_radius_m: Physical search radius in meters (default 0.05 = 5cm for side grasps)
            depth_threshold_ratio: Ratio of average depth to use as clustering threshold (default 0.05 = 5%)
            cluster_min_size: Minimum cluster size for consideration (default 10 pixels)
            
        Returns:
            adjusted_position: Position adjusted to center on the peak depth region within the search radius
        """
        try:
            if self.latest_depth_image is None or self.latest_color_image is None:
                self.logger.warning("No depth/color image available for centering adjustment")
                return target_position
                
            # Get camera intrinsics
            fx = self.main_camera.intrinsics.fx
            fy = self.main_camera.intrinsics.fy
            ppx = self.main_camera.intrinsics.ppx
            ppy = self.main_camera.intrinsics.ppy
            
            # Get depth scale
            depth_scale = getattr(self.main_camera, 'depth_scale', 1000.0)
            
            # Debug: Print camera intrinsics and depth scale
            self.logger.info(f"DEBUG - Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, ppx={ppx:.2f}, ppy={ppy:.2f}")
            self.logger.info(f"DEBUG - Depth scale: {depth_scale}")
            self.logger.info(f"DEBUG - Depth image shape: {self.latest_depth_image.shape}")
            self.logger.info(f"DEBUG - Depth image dtype: {self.latest_depth_image.dtype}")
            
            # Convert target position to pixel coordinates (assuming it's in camera frame)
            if isinstance(target_position, np.ndarray):
                pos = target_position.copy()
            else:
                pos = np.array(target_position)
                
            self.logger.info(f"DEBUG - Target position: {pos}")
            
            # Project 3D point to 2D pixel coordinates
            if len(pos) >= 3 and pos[2] > 0:  # Valid depth
                pixel_x = int(fx * pos[0] / pos[2] + ppx)
                pixel_y = int(fy * pos[1] / pos[2] + ppy)
            else:
                self.logger.warning("Invalid target position for depth centering")
                return target_position
                
            self.logger.info(f"DEBUG - Projected pixel coordinates: ({pixel_x}, {pixel_y})")
            
            # Ensure pixel coordinates are within image bounds
            h, w = self.latest_depth_image.shape
            pixel_x = max(0, min(w-1, pixel_x))
            pixel_y = max(0, min(h-1, pixel_y))
            
            self.logger.info(f"DEBUG - Clamped pixel coordinates: ({pixel_x}, {pixel_y})")
            
            # Define search region around target pixel based on physical radius
            # Convert physical radius to pixels at the target depth
            target_depth_m = pos[2] if pos[2] > 0 else 1.0  # Use 1m default if invalid
            radius_pixels_x = int(search_radius_m * fx / target_depth_m)
            radius_pixels_y = int(search_radius_m * fy / target_depth_m)
            
            x_min = max(0, pixel_x - radius_pixels_x)
            x_max = min(w, pixel_x + radius_pixels_x)
            y_min = max(0, pixel_y - radius_pixels_y) 
            y_max = min(h, pixel_y + radius_pixels_y)
            
            self.logger.info(f"DEBUG - Search radius: {search_radius_m}m ({radius_pixels_x}x{radius_pixels_y} pixels)")
            self.logger.info(f"DEBUG - Search region: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]")
            
            # Extract depth region
            depth_region = self.latest_depth_image[y_min:y_max, x_min:x_max]
            
            # Get sample depth values around the target pixel for debugging
            target_depth_raw = self.latest_depth_image[pixel_y, pixel_x]
            self.logger.info(f"DEBUG - Target pixel depth (raw): {target_depth_raw}")
            self.logger.info(f"DEBUG - Depth region shape: {depth_region.shape}")
            self.logger.info(f"DEBUG - Depth region min/max: {depth_region.min()}/{depth_region.max()}")
            
            # Filter out invalid depth values (0 or very large values)
            # For RealSense, depth is typically in mm, valid range is usually 0.1m to 10m
            valid_mask = (depth_region > 100) & (depth_region < 10000)  # 100mm to 10m in mm units
            
            if not np.any(valid_mask):
                self.logger.warning("No valid depth values in gripper region")
                return target_position
                
            # Get valid depth values and their coordinates
            valid_depths = depth_region[valid_mask]
            y_coords, x_coords = np.where(valid_mask)
            
            self.logger.info(f"DEBUG - Valid depth points: {len(valid_depths)}")
            self.logger.info(f"DEBUG - Valid depth range: {valid_depths.min():.1f} to {valid_depths.max():.1f} mm")
            
            # Calculate dynamic depth threshold based on average depth of the masked region
            average_depth = np.mean(valid_depths)
            depth_threshold = average_depth * depth_threshold_ratio
            
            self.logger.info(f"DEBUG - Average depth in region: {average_depth:.1f} mm")
            self.logger.info(f"DEBUG - Dynamic depth threshold: {depth_threshold:.1f} mm ({depth_threshold_ratio*100:.1f}% of average)")
            
            # DEPTH CLUSTERING: Group points by similar depth values
            # This helps identify the main object vs background/other objects
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist
            
            if len(valid_depths) >= cluster_min_size:
                try:
                    # Use hierarchical clustering on depth values
                    depth_points = valid_depths.reshape(-1, 1)  # Shape for clustering
                    
                    # Calculate linkage matrix
                    linkage_matrix = linkage(depth_points, method='ward')
                    
                    # Form clusters with depth threshold
                    cluster_labels = fcluster(linkage_matrix, depth_threshold, criterion='distance')
                    
                    # Find the largest cluster (most likely the main object)
                    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
                    
                    # Filter clusters by minimum size
                    valid_clusters = unique_labels[label_counts >= cluster_min_size]
                    
                    if len(valid_clusters) > 0:
                        # Choose the cluster closest to the original target depth
                        target_depth_mm = target_depth_raw
                        best_cluster = None
                        best_depth_diff = float('inf')
                        
                        for cluster_id in valid_clusters:
                            cluster_mask = cluster_labels == cluster_id
                            cluster_depths = valid_depths[cluster_mask]
                            cluster_mean_depth = np.mean(cluster_depths)
                            
                            depth_diff = abs(cluster_mean_depth - target_depth_mm)
                            
                            self.logger.info(f"DEBUG - Cluster {cluster_id}: size={np.sum(cluster_mask)}, "
                                           f"mean_depth={cluster_mean_depth:.1f}mm, diff_from_target={depth_diff:.1f}mm")
                            
                            if depth_diff < best_depth_diff:
                                best_depth_diff = depth_diff
                                best_cluster = cluster_id
                        
                        if best_cluster is not None:
                            # Use only the best cluster for centering
                            cluster_mask = cluster_labels == best_cluster
                            clustered_depths = valid_depths[cluster_mask]
                            clustered_x_coords = x_coords[cluster_mask]
                            clustered_y_coords = y_coords[cluster_mask]
                            
                            self.logger.info(f"DEBUG - Selected cluster {best_cluster}: {len(clustered_depths)} points, "
                                           f"depth_range={clustered_depths.min():.1f}-{clustered_depths.max():.1f}mm")
                            
                            # PEAK DEPTH OPTIMIZATION: Find the peak (closest) depth within the cluster
                            # This helps center on the most prominent/graspable part of the object
                            peak_depth = np.min(clustered_depths)  # Closest depth = peak
                            peak_threshold = peak_depth + 10  # Allow 10mm tolerance from peak
                            
                            # Select points near the peak depth
                            peak_mask = clustered_depths <= peak_threshold
                            peak_depths = clustered_depths[peak_mask]
                            peak_x_coords = clustered_x_coords[peak_mask]
                            peak_y_coords = clustered_y_coords[peak_mask]
                            
                            if len(peak_depths) >= 3:  # Need minimum points for reliable centering
                                self.logger.info(f"DEBUG - Peak depth optimization: peak={peak_depth:.1f}mm, "
                                               f"threshold={peak_threshold:.1f}mm, peak_points={len(peak_depths)}")
                                
                                # Use peak region for final centering
                                final_depths = peak_depths
                                final_x_coords = peak_x_coords + x_min  # Convert to absolute coordinates
                                final_y_coords = peak_y_coords + y_min
                            else:
                                # Fall back to full cluster if not enough peak points
                                self.logger.info("DEBUG - Not enough peak points, using full cluster")
                                final_depths = clustered_depths
                                final_x_coords = clustered_x_coords + x_min
                                final_y_coords = clustered_y_coords + y_min
                        else:
                            # No suitable cluster found, use all valid points
                            self.logger.info("DEBUG - No suitable cluster found, using all valid points")
                            final_depths = valid_depths
                            final_x_coords = x_coords + x_min
                            final_y_coords = y_coords + y_min
                    else:
                        # No clusters meet minimum size, use all points
                        self.logger.info("DEBUG - No clusters meet minimum size, using all valid points")
                        final_depths = valid_depths
                        final_x_coords = x_coords + x_min
                        final_y_coords = y_coords + y_min
                        
                except Exception as clustering_error:
                    # Clustering failed, fall back to all valid points
                    self.logger.warning(f"Clustering failed: {clustering_error}, using all valid points")
                    final_depths = valid_depths
                    final_x_coords = x_coords + x_min
                    final_y_coords = y_coords + y_min
            else:
                # Not enough points for clustering
                self.logger.info("DEBUG - Not enough points for clustering, using all valid points")
                final_depths = valid_depths
                final_x_coords = x_coords + x_min
                final_y_coords = y_coords + y_min
            
            # Calculate weighted centroid of the selected points
            # Weight by inverse depth (closer points get higher weight) and cluster density
            weights = 1.0 / (final_depths + 1)  # Inverse depth weighting
            centroid_x = np.average(final_x_coords, weights=weights)
            centroid_y = np.average(final_y_coords, weights=weights)
            centroid_depth = np.average(final_depths, weights=weights)
            
            self.logger.info(f"DEBUG - Final centroid pixel: ({centroid_x:.1f}, {centroid_y:.1f})")
            self.logger.info(f"DEBUG - Final centroid depth: {centroid_depth:.1f} mm")
            self.logger.info(f"DEBUG - Used {len(final_depths)} points for centering")
            
            # Convert depth using depth scale (RealSense depth_scale converts raw to meters)
            centroid_depth_m = centroid_depth * depth_scale
            
            self.logger.info(f"DEBUG - Centroid depth in meters: {centroid_depth_m:.4f} m")
            
            # Convert centroid pixel back to 3D camera coordinates
            centroid_x_3d = (centroid_x - ppx) * centroid_depth_m / fx
            centroid_y_3d = (centroid_y - ppy) * centroid_depth_m / fy
            
            self.logger.info(f"DEBUG - Centroid 3D coordinates: ({centroid_x_3d:.4f}, {centroid_y_3d:.4f}, {centroid_depth_m:.4f})")
            
            # Adjust both X and Y coordinates to center on the clustered region,
            # but maintain original depth for contact (unless peak optimization suggests otherwise)
            original_pos = np.array(target_position[:3])
            
            # Calculate X and Y adjustments for centering
            x_adjustment = centroid_x_3d - original_pos[0]
            y_adjustment = centroid_y_3d - original_pos[1]
            
            # Limit adjustments to the search radius to ensure centering within the specified region
            x_adjustment = np.clip(x_adjustment, -search_radius_m, search_radius_m)
            y_adjustment = np.clip(y_adjustment, -search_radius_m, search_radius_m)
            
            # For Z coordinate, use peak depth if it's significantly different from original
            # This helps grasp the most prominent part of the object
            z_adjustment = 0.0
            original_depth_mm = target_depth_raw
            depth_difference = abs(centroid_depth - original_depth_mm)
            
            if depth_difference > depth_threshold and centroid_depth < original_depth_mm:
                # Peak is significantly closer, adjust Z to grasp the peak
                centroid_z_3d = centroid_depth_m
                z_adjustment = centroid_z_3d - original_pos[2]
                max_z_adjustment = 0.02  # Maximum 2cm adjustment in Z direction
                z_adjustment = np.clip(z_adjustment, -max_z_adjustment, max_z_adjustment)
                self.logger.info(f"DEBUG - Using peak depth for Z adjustment: {z_adjustment:.4f} m")
            else:
                self.logger.info(f"DEBUG - Maintaining original Z coordinate")
            
            # Create adjusted position
            adjusted_position = [
                original_pos[0] + x_adjustment,  # Adjusted X for centering
                original_pos[1] + y_adjustment,  # Adjusted Y for centering
                original_pos[2] + z_adjustment   # Adjusted Z for peak optimization or original
            ]
            
            self.logger.info(f"DEBUG - X adjustment for centering: {x_adjustment:.4f} m")
            self.logger.info(f"DEBUG - Y adjustment for centering: {y_adjustment:.4f} m")
            self.logger.info(f"DEBUG - Z adjustment for peak optimization: {z_adjustment:.4f} m")
            
            # Calculate final adjustment offset for logging
            final_offset = np.array(adjusted_position) - np.array(target_position[:3])
            final_offset_magnitude = np.linalg.norm(final_offset)
            
            self.logger.info(f"Depth clustering and peak optimization adjustment: offset={final_offset}, magnitude={final_offset_magnitude:.4f}m")
            self.logger.info(f"Original position: {target_position[:3]}")
            self.logger.info(f"Adjusted position: {adjusted_position}")
            
            return adjusted_position
            
        except Exception as e:
            self.logger.error(f"Error in depth centering adjustment: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return target_position

    def estimate_pixel_to_meter_ratio(self, world_position):
        """
        Estimate the pixel-to-meter conversion ratio based on distance from camera.
        This is a rough approximation - ideally should use proper camera intrinsics.
        
        Args:
            world_position: Current position in world coordinates [x, y, z]
            
        Returns:
            float: Estimated meters per pixel
        """
        try:
            # Estimate distance from camera (assume camera is roughly at origin)
            distance_to_camera = np.linalg.norm(world_position)
            
            # Rule of thumb: at 1m distance, ~1mm per pixel for typical camera FOV
            # Scale linearly with distance
            pixels_per_meter_at_1m = 1000  # approximately 1000 pixels per meter at 1m
            pixels_per_meter = pixels_per_meter_at_1m / distance_to_camera
            meters_per_pixel = 1.0 / pixels_per_meter
            
            # Clamp to reasonable bounds
            meters_per_pixel = max(0.0005, min(0.005, meters_per_pixel))  # 0.5mm to 5mm per pixel
            
            self.logger.debug(f"Estimated pixel-to-meter ratio: {meters_per_pixel:.6f} m/pixel at distance {distance_to_camera:.3f}m")
            return meters_per_pixel
            
        except Exception as e:
            self.logger.warning(f"Error estimating pixel-to-meter ratio: {e}, using default")
            return 0.001  # Default 1mm per pixel

    def execute_action(self, action: ExecutableAction, timeout=120, initial_cam_tf=None, is_place=False) -> bool:
        """Execute a skill action using the CuRobo motion planner with camera poses"""
        try:
            start_time = time.time()
            def is_timeout_approaching():
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining < 0.5:
                    self.logger.warning(f"Action timeout approaching: {elapsed:.2f}s elapsed, {remaining:.2f}s remaining")
                    return True
                return False
            print(f"################## ARM STATE: {self.motion_planner.arm.get_state()}")
            while self.motion_planner.arm.get_state()[1] in [1, 4, 5]:
                print(f"################## ARM STATE: {self.motion_planner.arm.get_state()}")
                if self.motion_planner.arm.get_state()[1] == 5:
                    self.motion_planner.arm.set_state(0)
                time.sleep(0.5)
            # Handle gripper actions directly
            if action.action_type == 'close_gripper':
                if self.data_recorder:
                    self.data_recorder.start_robot_motion(f"close_gripper")
                force = action.parameters.get('force', 10.0)
                result = self.motion_planner.close_gripper(wait=True, timeout=100.0)
                if self.data_recorder:
                    self.data_recorder.end_robot_motion(f"close_gripper")
                return result
                    
            elif action.action_type == 'open_gripper':
                if self.data_recorder:
                    self.data_recorder.start_robot_motion(f"open_gripper")
                self.motion_planner.open_gripper(wait=True, timeout=100)
                time.sleep(1)
                result = self.motion_planner.open_gripper(wait=True, timeout=100)
                if self.data_recorder:
                    self.data_recorder.end_robot_motion(f"open_gripper")
                return result

            elif action.action_type == 'drop':
                # Drop action is simply opening the gripper to release the held object
                if self.data_recorder:
                    self.data_recorder.start_robot_motion(f"drop")
                self.logger.info("Executing drop action - opening gripper to release object")
                result = self.motion_planner.open_gripper(wait=True, timeout=100)
                if self.data_recorder:
                    self.data_recorder.end_robot_motion(f"drop")
                return result

            elif action.action_type == 'retract_gripper':
                # Move gripper back by specified distance
                distance = 0.1
                speed = action.parameters.get('speed', 0.3)
                
                # Get current pose
                pose = self.motion_planner.get_robot_tcp_pose()
                if self.data_recorder:
                    self.data_recorder.start_robot_motion(f"retract_gripper")
                speed_factor = 2.0 if self.fast_mode else 1.5  # Increased retract speeds
                success = self.motion_planner.retract_gripper(speed_factor=speed_factor)
                if self.data_recorder:
                    self.data_recorder.end_robot_motion(f"retract_gripper")
                return success
                if pose is None:
                    self.logger.error("Could not get current TCP pose for retraction")
                    return False
                    
                current_position, current_orientation = pose
                
                # Move back along Z-axis
                target_position = np.array(current_position) + np.array([0, 0, distance]).tolist()
                print(f"Start pose retract: {current_position}")
                print(f"Target retract: {target_position}")
                success, _, _ = self.motion_planner.move_to_pose(
                    target_position=target_position,
                    target_orientation=current_orientation,
                    execute=True,
                    speed_factor=speed,
                    planning_timeout=min(timeout - (time.time() - start_time), 10.0)
                )
                return success
            
            # Handle twist action
            elif action.action_type == 'twist':
                direction = action.parameters.get('direction', 'clockwise')
                angular_velocity = 2#action.parameters.get('angular_velocity', 1.5)
                rotation_angle = (2 * np.pi) - 1.0#action.parameters.get('rotation_angle', 2 * np.pi)  # Default 90 degrees
                
                self.logger.info(f"Executing twist {direction} with angle {rotation_angle:.2f} rad")
                
                # Calculate remaining time for twist
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0.5:
                    self.logger.error("Insufficient time remaining for twist")
                    return False
                
                # Execute the twist motion
                if self.data_recorder:
                    self.data_recorder.start_robot_motion(f"twist_{direction}")
                success = self.motion_planner.execute_wrist_twist(
                    direction=direction,
                    rotation_angle=rotation_angle,
                    speed_factor=angular_velocity,
                    timeout=min(remaining_time, 30.0)
                )
                if self.data_recorder:
                    self.data_recorder.end_robot_motion(f"twist_{direction}")
                
                return success
            
            # Set orientation based on grasp type
            if action.is_top_down_grasp:
                action.orientation = self.motion_planner.create_top_down_orientation()
                self.logger.info("Using top-down orientation for grasp")
                
            elif action.is_side_grasp:
                # For side grasp, we might want to use a custom orientation
                action.orientation = (0, 0.7071, 0, 0.7071)
                # action.orientation = (-0.5, -0.5, -0.5, 0.5)
                self.logger.info("Using side orientation for grasp")
            
            # Check timeout after setup
            if is_timeout_approaching():
                self.logger.error("Timeout approaching after action setup")
                return False
                
            # Execute based on action type
            if action.action_type == 'move_gripper_to_pose':
                # Store interaction point coordinates for later pivot calculations
                if action.pixel_position is not None:
                    self.current_interaction_point_pixel = action.pixel_position
                    self.logger.info(f"Stored interaction point pixel coordinates: {self.current_interaction_point_pixel}")
                
                # Store 3D world coordinates of interaction point for pivot calculations
                if action.position is not None:
                    self.current_interaction_point_world = action.position.tolist() if isinstance(action.position, np.ndarray) else action.position
                    self.logger.info(f"Stored interaction point world coordinates: {self.current_interaction_point_world}")
                
                # Position handling depends on camera type
                target_position = action.position.tolist() if isinstance(action.position, np.ndarray) else action.position
                target_orientation = action.orientation
                
                # Apply depth-based centering adjustment before sending to robot interface
                # Apply position adjustment if needed (for non-top-down grasps)
                # Top-down grasps will be adjusted in the robot interface for better accuracy
                if not action.is_top_down_grasp:
                    # Use configurable radius for side grasps (default 5cm)
                    search_radius = 0.05 if action.is_side_grasp else 0.03
                    self.logger.info(f"Applying depth-based centering adjustment to TCP pose (radius: {search_radius}m)")
                    adjusted_position = self.adjust_tcp_pose_for_depth_centering(target_position, search_radius_m=search_radius)
                else:
                    # Top-down grasps will be adjusted in robot interface using robot-frame coordinates
                    # Set search radius for top-down grasps (default 3cm for precision)
                    search_radius = 0.05  # 3cm radius for top-down grasp centering
                    self.logger.info(f"Top-down grasp will be adjusted in robot interface with search radius: {search_radius}m")
                    adjusted_position = target_position
                
                # Calculate remaining time for move
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0.5:
                    self.logger.error("Insufficient time remaining for move_gripper_to_pose")
                    return False
                                
                # Use the motion planner to move to pose with adjusted position
                # Check if this is a top-down grasp that needs surface adjustment
                is_top_down_param = action.parameters.get('is_top_down_grasp', False)
                is_top_down_orientation = target_orientation is not None and abs(target_orientation[1]) > 0.9
                is_top_down = is_top_down_param or is_top_down_orientation
                
                self.logger.info(f"Surface adjustment check: is_top_down_param={is_top_down_param}, "
                               f"is_top_down_orientation={is_top_down_orientation}, "
                               f"final_is_top_down={is_top_down}, "
                               f"target_orientation={target_orientation}")
                
                # Get fresh camera images for surface adjustment if needed
                if is_top_down:
                    self.logger.info("Getting fresh camera images for surface adjustment")
                    color_image, depth_image = self.get_latest_images(timeout=1.0)
                    if depth_image is not None:
                        self.latest_depth_image = depth_image
                        self.logger.info(f"Updated depth image for surface adjustment, shape: {depth_image.shape}")
                    else:
                        self.logger.warning("Failed to get fresh depth image for surface adjustment")
                
                # Extract object mask if available
                object_mask = None
                self.logger.info(f"Object info check: action.object_info={action.object_info is not None}")
                if action.object_info is not None:
                    self.logger.info(f"Object info has mask: {hasattr(action.object_info, 'mask')}")
                    if hasattr(action.object_info, 'mask'):
                        object_mask = action.object_info.mask
                        mask_shape = object_mask.shape if object_mask is not None else None
                        self.logger.info(f"Using object mask for surface-based TCP adjustment, shape: {mask_shape}")
                else:
                    self.logger.info("No object_info available for surface adjustment")
                
                # Start motion tracking
                if self.data_recorder:
                    # Get object name from ObjectInfo.name
                    object_name = 'unknown_object'
                    if hasattr(action, 'object_info') and action.object_info and hasattr(action.object_info, 'name'):
                        object_name = action.object_info.name

                    self.data_recorder.start_robot_motion(f"{action.action_type}_{object_name}")

                # Determine orientation based on grasp type
                # Use fixed orientations: top-down (180, 0, 0) or side grasp (160, -87, 20)
                is_side_grasp = action.is_side_grasp if hasattr(action, 'is_side_grasp') else False
                if is_top_down:
                    grasp_orientation = None  # Will use force_top_down flag
                elif is_side_grasp:
                    grasp_orientation = 'side_grasp'  # Signal to use fixed side grasp orientation
                else:
                    grasp_orientation = 'side_grasp'  # Default to side grasp if not top-down

                # Use direct xArm interface instead of CuRobo trajectory planning
                # CuRobo is only used for frame transformation (convert_cam_pose_to_base)
                success = self.motion_planner.move_to_pose_direct(
                    target_position=adjusted_position,
                    target_orientation=grasp_orientation,
                    force_top_down=is_top_down,
                    speed=100,  # mm/s
                    acc=1000,   # mm/s²
                    is_camera_frame=True,
                    is_place=is_place,
                    wait=True
                )

                # End motion tracking
                if self.data_recorder:
                    # Get object name from ObjectInfo.name
                    object_name = 'unknown_object'
                    if hasattr(action, 'object_info') and action.object_info and hasattr(action.object_info, 'name'):
                        object_name = action.object_info.name

                    self.data_recorder.end_robot_motion(f"{action.action_type}_{object_name}")
                
                return success
                
            elif action.action_type in ['push', 'pull']:
                # Handle push/pull actions
                distance = action.parameters.get('distance', 0.1)

                if action.parameters.get('is_button', False):
                    distance = 0.01
                    self.motion_planner.close_gripper()
                    
                speed = action.parameters.get('speed', 0.3)
                is_parallel = action.parameters.get('is_parallel', False)
                surface_normal = action.parameters.get('surface_normal', None)
                pivot_position = action.parameters.get('pivot_position', None)
                hinge_location = action.parameters.get('hinge_location', None)
                
                # Surface normal conversion only needed for ZED camera
                if surface_normal is not None and self.use_zed_camera and self.zed_to_robot_transform is not None:
                    # Convert surface normal from ZED frame to robot base frame
                    surface_normal_homogeneous = np.array([surface_normal[0], surface_normal[1], surface_normal[2], 0])
                    surface_normal_robot = self.zed_to_robot_transform @ surface_normal_homogeneous
                    surface_normal = surface_normal_robot[:3]
                    print(f"Surface normal converted from ZED to robot frame: {surface_normal}")
                
                # Calculate pivot point based on hinge_location or legacy pivot_position
                pivot_point = None
                
                # Check for new hinge_location parameter first
                if hinge_location and hinge_location in ['top', 'bottom', 'left', 'right']:
                    self.logger.info(f"Using hinge location: {hinge_location}")
                    # For hinge_location, we'll calculate the pivot point based on the target position and hinge edge
                    # This is a simplified implementation - in practice, this would need object geometry information
                    
                    # Get current gripper position
                    current_pose = self.motion_planner.get_robot_tcp_pose()
                    if current_pose is not None:
                        current_position, _ = current_pose
                        
                        # Flatten the position array if it's nested
                        if hasattr(current_position, 'flatten'):
                            current_position = current_position.flatten()
                        elif isinstance(current_position, (list, tuple)) and len(current_position) > 0:
                            if isinstance(current_position[0], (list, tuple, np.ndarray)):
                                current_position = current_position[0]
                        
                        # Calculate pivot point based on hinge location using actual object bounding box
                        # COORDINATE SYSTEM: Y+ = right, Y- = left, X- = forward/top, X+ = back/bottom
                        
                        # Use actual object bounding box if available
                        if hasattr(self, 'current_object_info') and self.current_object_info:
                            bbox = None
                            if self.current_object_info.bbox:
                                bbox = self.current_object_info.bbox  # [x, y, w, h] format in pixels
                            elif hasattr(self.current_object_info, 'mask') and self.current_object_info.mask is not None:
                                # Fallback: calculate bounding box from segmentation mask
                                mask = self.current_object_info.mask
                                # Find contours in the mask
                                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    # Get bounding box from largest contour
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    x, y, w, h = cv2.boundingRect(largest_contour)
                                    bbox = [x, y, w, h]
                                    self.logger.info(f"Calculated bounding box from segmentation mask: {bbox}")
                            
                            if bbox:
                                bbox_x, bbox_y, bbox_w, bbox_h = bbox
                                
                                # Convert pixel coordinates to world coordinates
                                pixel_to_meter = self.estimate_pixel_to_meter_ratio(current_position)
                                
                                # Calculate pivot point using 3D world coordinates instead of pixel conversion
                                if hasattr(self, 'current_interaction_point_world') and self.current_interaction_point_world:
                                    # Use 3D world coordinates of interaction point for accurate pivot calculation
                                    interaction_world_x, interaction_world_y, interaction_world_z = self.current_interaction_point_world
                                    
                                    # Calculate 3D offset from interaction point to hinge edge using bounding box dimensions
                                    # Convert bounding box dimensions to world units
                                    bbox_width_world = bbox_w * pixel_to_meter
                                    bbox_height_world = bbox_h * pixel_to_meter
                                    
                                    # Get interaction point pixel coordinates for offset calculation
                                    if hasattr(self, 'current_interaction_point_pixel') and self.current_interaction_point_pixel:
                                        interaction_px, interaction_py = self.current_interaction_point_pixel
                                        
                                        # CAMERA FRAME: X+ = right, Y+ = down
                                        # ROBOT FRAME: Y+ = right, Y- = left, X- = forward/top, X+ = back/bottom
                                        
                                        # Calculate world position of the specific bounding box edge
                                        if hinge_location == 'left':
                                            # Left edge: offset from interaction point to left edge of bbox
                                            pixel_offset_x = bbox_x - interaction_px
                                            world_offset_y = pixel_offset_x * pixel_to_meter  # Camera X -> Robot Y
                                            pivot_point = [interaction_world_x, interaction_world_y + world_offset_y, interaction_world_z]
                                        elif hinge_location == 'right':
                                            # Right edge: offset from interaction point to right edge of bbox
                                            pixel_offset_x = (bbox_x + bbox_w) - interaction_px
                                            world_offset_y = pixel_offset_x * pixel_to_meter  # Camera X -> Robot Y
                                            pivot_point = [interaction_world_x, interaction_world_y + world_offset_y, interaction_world_z]
                                        elif hinge_location == 'top':
                                            # Top edge: offset from interaction point to top edge of bbox
                                            pixel_offset_y = bbox_y - interaction_py
                                            world_offset_x = pixel_offset_y * pixel_to_meter  # Camera Y -> Robot X (inverted)
                                            pivot_point = [interaction_world_x - world_offset_x, interaction_world_y, interaction_world_z]
                                        elif hinge_location == 'bottom':
                                            # Bottom edge: offset from interaction point to bottom edge of bbox
                                            pixel_offset_y = (bbox_y + bbox_h) - interaction_py
                                            world_offset_x = pixel_offset_y * pixel_to_meter  # Camera Y -> Robot X (inverted)
                                            pivot_point = [interaction_world_x - world_offset_x, interaction_world_y, interaction_world_z]
                                            
                                        self.logger.info(f"Calculated pivot using 3D world coordinates: interaction_point={self.current_interaction_point_world}, hinge={hinge_location}, pivot={pivot_point}, offset={world_offset_y if 'world_offset_y' in locals() else world_offset_x:.3f}m")
                                    else:
                                        self.logger.warning("No interaction point pixel coordinates available for offset calculation")
                                        pivot_point = None
                            else:
                                # Fallback: use middle of the specified edge of bounding box
                                self.logger.info("No interaction point stored, using middle of bounding box edge")
                                if hinge_location == 'left':
                                    # Distance from center to left edge
                                    offset_distance = bbox_w * 0.5 * pixel_to_meter
                                    pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                                elif hinge_location == 'right':
                                    # Distance from center to right edge
                                    offset_distance = bbox_w * 0.5 * pixel_to_meter
                                    pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                                elif hinge_location == 'top':
                                    # Distance from center to top edge
                                    offset_distance = bbox_h * 0.5 * pixel_to_meter
                                    pivot_point = [current_position[0] - offset_distance, current_position[1], current_position[2]]
                                elif hinge_location == 'bottom':
                                    # Distance from center to bottom edge
                                    offset_distance = bbox_h * 0.5 * pixel_to_meter
                                    pivot_point = [current_position[0] + offset_distance, current_position[1], current_position[2]]
                                
                            self.logger.info(f"Using object bounding box for pivot calculation: bbox={bbox}, offset={offset_distance:.3f}m")
                        else:
                            # Fallback to reasonable estimates if no bounding box available
                            self.logger.warning("No object bounding box available, using estimated distances")
                            if hinge_location == 'left':
                                offset_distance = 0.33  # 25cm fallback  
                                # For left hinge: pivot to the left (negative Y direction) 
                                pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                            elif hinge_location == 'right':
                                offset_distance = 0.33   # 25cm fallback (reduced from 0.4m)
                                # For right hinge: pivot should be at the far right edge of object (negative Y direction)
                                pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                            elif hinge_location == 'top':
                                offset_distance = 0.3   # 30cm fallback
                                pivot_point = [current_position[0] - offset_distance, current_position[1], current_position[2]]
                            elif hinge_location == 'bottom':
                                offset_distance = 0.3   # 30cm fallback
                                pivot_point = [current_position[0] + offset_distance, current_position[1], current_position[2]]
                        
                        self.logger.info(f"Hinge pivot point calculated: gripper={current_position}, hinge={hinge_location}, pivot={pivot_point}")
                    else:
                        self.logger.warning("Could not get current gripper position for hinge pivot calculation")
                
                # Fall back to legacy pivot_position for backward compatibility
                elif pivot_position is not None:
                    self.logger.info("Using legacy pivot_position parameter")
                    # Get current gripper position
                    current_pose = self.motion_planner.get_robot_tcp_pose()
                    if current_pose is not None:
                        current_position, _ = current_pose
                        
                        # Flatten the position array if it's nested (handle both [x,y,z] and [[x,y,z]] formats)
                        if hasattr(current_position, 'flatten'):
                            current_position = current_position.flatten()
                        elif isinstance(current_position, (list, tuple)) and len(current_position) > 0:
                            if isinstance(current_position[0], (list, tuple, np.ndarray)):
                                current_position = current_position[0]
                        
                        # Extract x offset from pivot_position parameter (only use x component)
                        if isinstance(pivot_position, (list, tuple, np.ndarray)):
                            x_offset = pivot_position[0]
                        else:
                            x_offset = pivot_position  # Assume it's a scalar x offset
                        
                        # Construct pivot point: x offset from gripper, same y and z as gripper
                        pivot_point = [
                            current_position[0],  # Gripper x + offset
                            current_position[1] + x_offset,             # Same y as gripper
                            current_position[2]              # Same z as gripper
                        ]
                        print(f"pivot provided: {pivot_position}")
                        print(f"Pivot point calculated: gripper={current_position}, x_offset={x_offset}, pivot={pivot_point}")
                    else:
                        self.logger.warning("Could not get current gripper position for pivot point calculation")
                
                # Check if we have enough time for the action
                if is_timeout_approaching():
                    self.logger.error("Timeout approaching, cannot execute push/pull action")
                    return False
                    
                # Calculate remaining time
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0.5:
                    self.logger.error("Insufficient time remaining for push/pull")
                    return False
                
                # Execute the push/pull motion
                is_push = (action.action_type == 'push')
                
                # Use direct pivoted pull if pivot position is provided and it's a pull action
                if not is_push and pivot_point is not None:
                    self.logger.info(f"Using direct pivoted pull with pivot point: {pivot_point}")
                    
                    # Get current robot pose for pivot pull execution (use API method like in test)
                    current_pose = self.motion_planner.get_tcp_pose_api()
                    if current_pose is not None:
                        current_position, current_orientation = current_pose
                        
                        # Flatten position if needed (like in test method)
                        if hasattr(current_position, 'flatten'):
                            current_position = current_position.flatten()
                        
                        # Convert pivot_point to numpy array
                        pivot_point_array = np.array(pivot_point)
                        
                        # Calculate radius based on hinge location
                        if hinge_location and hinge_location in ['top', 'bottom', 'left', 'right']:
                            # For hinge location, the radius should be the distance from interaction point to hinge edge
                            # The current_position is the interaction point, pivot_point is calculated hinge location
                            radius = np.linalg.norm(np.array(current_position) - pivot_point_array)
                            self.logger.info(f"Using hinge-based radius calculation: hinge_location={hinge_location}, radius={radius:.3f}")
                        else:
                            # Legacy radius calculation (distance from current position to pivot)
                            radius = np.linalg.norm(np.array(current_position) - pivot_point_array)
                            self.logger.info(f"Using legacy pivot radius calculation: radius={radius:.3f}")
                        
                        self.logger.info(f"Executing pivot pull: current_pos={current_position}, pivot={pivot_point_array}, radius={radius:.3f}")
                        
                        # Use conservative settings like in the test method
                        success = self.motion_planner.execute_pivot_pull_direct_xarm(
                            pivot_point=pivot_point_array,
                            current_position=current_position,
                            current_orientation=current_orientation,
                            radius=radius,
                            arc_angle_degrees=75.0,  # Use conservative angle like in test
                            segments=5,
                            speed_factor=0.05,  # Use ultra conservative speed like in test
                            is_quat=False,  # Match test method setting
                            hinge_location=hinge_location,  # Pass hinge location for logging
                            is_push=is_push  # Pass the actual push/pull action type
                        )
                    else:
                        self.logger.error("Could not get current robot pose for pivot pull")
                        success = False
                    
                    if not success:
                        self.logger.warning("Direct pivoted pull failed, falling back to standard plan_push_pull")
                        # Fall back to standard method
                        success, _, _ = self.motion_planner.plan_push_pull(
                            distance=distance,
                            is_push=is_push,
                            custom_normal=surface_normal,
                            move_parallel=is_parallel,
                            pivot_point=pivot_point,
                            planning_timeout=min(remaining_time, 10.0),
                            execute=True,
                            speed_factor=speed
                        )
                else:
                    # Use the motion planner's standard push/pull functionality
                    success, _, _ = self.motion_planner.plan_push_pull(
                        distance=distance,
                        is_push=is_push,
                        custom_normal=surface_normal,
                        move_parallel=is_parallel,
                        pivot_point=pivot_point,
                        planning_timeout=min(remaining_time, 10.0),
                        execute=True,
                        speed_factor=speed
                    )
                
                return success
                    
            self.logger.warning(f'Unknown action type: {action.action_type}')
            return False
                
        except Exception as e:
            self.logger.error(f'Action execution failed: {str(e)}')
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            # Update camera orientation after any action execution
            # This ensures the camera orientation is refreshed in case the robot moved
            # try:
            pass
            #     self.update_camera_orientation()
            # except Exception as e:
            #     self.logger.warning(f'Failed to update camera orientation after action: {str(e)}')
    
    def execute_skill(self, skill_name: str, parameters: str) -> Tuple[bool, str]:
        """Execute a skill directly using the selected camera"""
        try:
            start_time = time.time()
            self.execution_start_time = start_time
            camera_type = "ZED" if self.use_zed_camera else "RealSense"
            self.logger.info(f"Starting skill execution with {camera_type} camera: {skill_name} {parameters}")
            
            # Check if transform is available/needed
            if self.use_zed_camera and self.zed_to_robot_transform is None:
                self.logger.warning("ZED-to-robot transform not calibrated - poses may be inaccurate")
            elif not self.use_zed_camera:
                self.logger.info("Using RealSense camera - no transform calibration needed")
            
            # Handle detect_object skill specially - it doesn't go through validation
            if skill_name.lower() == 'detect_object':
                target_object = parameters.strip()
                if not target_object:
                    error_msg = "detect_object requires an object name parameter"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # Move to home position before object detection
                self.logger.info("Moving to home position before object detection")
                try:
                    # self.motion_planner.arm.move_gohome()
                    self.logger.info("Successfully moved to home position")
                except Exception as e:
                    self.logger.warning(f"Failed to move to home position: {str(e)}")
                
                # Get camera images for object detection
                color_image, depth_image = self.camera.get_frames()
                if color_image is None or depth_image is None:
                    error_msg = f"Failed to get {camera_type} camera images for object detection"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # Detect the target object
                self.logger.info(f"Detecting target object: {target_object}")
                object_info = None
                for attempt in range(DETECTION_RETRIES):
                    try:
                        object_info = self.perception_system.detect_object(
                            target_object=target_object,
                            image=color_image,
                            depth_image=depth_image
                        )
                        if object_info is not None:
                            self.logger.info(f"Successfully detected {target_object}")
                            break
                        else:
                            self.logger.warning(f"Could not detect {target_object} (attempt {attempt + 1}/{DETECTION_RETRIES})")
                    except Exception as e:
                        self.logger.warning(f"Error detecting {target_object} (attempt {attempt + 1}): {str(e)}")
                
                if object_info is None:
                    error_msg = f"Failed to detect object: {target_object}"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # Log success and timing
                elapsed_time = time.time() - start_time
                success_msg = f"Successfully detected {target_object} using {camera_type} camera in {elapsed_time:.2f}s"
                self.logger.info(success_msg)
                return True, success_msg
            
            # Validate skill command for non-detect_object skills
            is_valid, error_msg, params = self.validate_skill_command(skill_name.lower(), parameters)
            
            if not is_valid:
                self.logger.error(f"Validation failed: {error_msg}")
                return False, error_msg
            
            # Handle twist skill specially - it doesn't need object detection
            if skill_name.lower() == 'twist':
                direction = params[0].strip().lower()
                if direction not in ['clockwise', 'counterclockwise']:
                    error_msg = f"Invalid twist direction: {direction}. Must be 'clockwise' or 'counterclockwise'"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # Execute twist directly
                self.logger.info(f"Executing twist {direction}")
                success = self.motion_planner.execute_wrist_twist(
                    direction=direction,
                    rotation_angle=np.pi/2,  # 90 degrees
                    speed_factor=0.5,
                    timeout=30.0
                )
                
                if success:
                    elapsed_time = time.time() - start_time
                    success_msg = f"Successfully executed twist {direction} in {elapsed_time:.2f}s"
                    self.logger.info(success_msg)
                    return True, success_msg
                else:
                    error_msg = f"Failed to execute twist {direction}"
                    self.logger.error(error_msg)
                    return False, error_msg
            
            # Get images from the main camera
            self.logger.info(f"Acquiring {camera_type} camera images...")
            color_image, depth_image = self.get_latest_images(timeout=2.0)
            
            if color_image is None or depth_image is None:
                error_msg = f"Failed to get {camera_type} camera images within timeout"
                self.logger.error(error_msg)
                return False, error_msg
            
            # Initial visualization with raw image
            self.update_visualizations(color_image=color_image)
            
            # Generate skill command
            target_object = params[0]
            skill_command = (f"place {params[0]} on {params[1]}" 
                        if skill_name == 'place' 
                        else f"{skill_name} {params[0]}")
            
            self.logger.info(f"Generating skill for command: {skill_command}")
            
            # Detect object in the scene first
            object_info = None
            for _ in range(DETECTION_RETRIES):
                try:
                    self.logger.info(f"Detecting target object: {target_object}")
                    object_info = self.perception_system.detect_object(
                        target_object=target_object,
                        image=color_image,
                        depth_image=depth_image
                    )
                    
                    if object_info is not None:
                        self.logger.info(f"Detected {target_object} in {camera_type} camera scene")
                        # Store object_info for use in pivot point calculations
                        self.current_object_info = object_info
                        break
                    else:
                        self.logger.warning(f"Could not detect {target_object} in {camera_type} camera scene")
                        
                except Exception as e:
                    self.logger.warning(f"Error during object detection: {str(e)}")
            
            # Generate skill using synchronous interface
            skill = None
            try:
                self.logger.info(f"Generating skill synchronously for: {skill_command}")
                start_skill_time = time.time()
                
                skill = self.skill_handler.instantiate_skill(
                    skill_command,
                    target_object,
                    color_image,
                    depth_image,
                    object_info,
                    # task_context=self.task_context
                )
                
                skill_time = time.time() - start_skill_time
                self.logger.info(f"Skill generation completed in {skill_time:.2f}s")
                
                if skill is None:
                    error_msg = f"Failed to generate skill for: {skill_command}"
                    self.logger.error(error_msg)
                    return False, error_msg
                    
            except Exception as e:
                error_msg = f"Skill generation failed: {traceback.format_exc()}"
                self.logger.error(error_msg)
                return False, error_msg

            # Save manipulation specification file
            self.save_manipulation_spec(skill_command, target_object, skill)

            # Generate and display SAM mask visualization
            sam_vis = None
            if hasattr(self.perception_system, 'segmenter') and self.perception_system.segmenter is not None:
                self.logger.info("Generating SAM segmentation visualization")
                
                if hasattr(self.perception_system.segmenter, 'last_masks') and \
                   self.perception_system.segmenter.last_masks is not None:
                    
                    masks = self.perception_system.segmenter.last_masks
                    metadata = getattr(self.perception_system.segmenter, 'last_metadata', None)
                    
                    try:
                        sam_vis = self.perception_system.segmenter.visualize_masks(
                            color_image, masks, metadata, alpha=0.5
                        )
                        self.logger.info("Generated SAM visualization")
                    except Exception as e:
                        self.logger.warning(f"Error generating SAM visualization: {str(e)}")
                
                if sam_vis is not None:
                    self.update_visualizations(sam_masks=sam_vis)
            
            # Create Points of Interest visualization
            poi_image = None
            try:
                if object_info is not None:
                    points = getattr(object_info, 'points', {}).get('pixel_coords', [])
                    target_points = []
                    for action in skill.action_sequence:
                        if hasattr(action, 'position') and action.position is not None and \
                           not np.all(np.isclose(action.position, 0.0, atol=1e-6)):
                            target_points.append(action.position)
                    
                    self.logger.info(f"Creating points of interest visualization with {len(points)} points")
                    poi_image = self.create_points_of_interest_visualization(
                        color_image=color_image,
                        object_info=object_info,
                        points=points,
                        target_points=target_points
                    )
                    
                    if poi_image is not None:
                        self.update_visualizations(poi_image=poi_image)
                        
            except Exception as e:
                self.logger.error(f"Error creating points of interest visualization: {str(e)}")
            
            # Get point cloud from the main camera
            # pcd = self.main_camera.get_point_cloud()
            # if pcd is not None:
            #     self.logger.info(f"!!!!!!!!!!!!!!!!!! Retrieved point cloud from {camera_type} camera")
            #     # Note: Point cloud coordinates depend on camera type
            #     self.motion_planner.update_dynamic_collision_objects(pcd)
            # else:
            #     print(f"!!!!!!!!!!!!!!!!!! NO PCD FOR COLLISION OBJECTS")
            
            # Get camera transform (depends on camera type)
            if self.use_zed_camera and self.zed_to_robot_transform is not None:
                initial_cam_tf = Rotation.from_matrix(self.zed_to_robot_transform[:3, :3])
            else:
                _, initial_cam_tf = self.motion_planner.get_camera_transform()
            
            # Execute actions
            self.logger.info(f"Executing {len(skill.action_sequence)} actions with {camera_type} camera poses...")
            for i, action in enumerate(skill.action_sequence, 1):
                self.logger.info(f"Executing action {i}/{len(skill.action_sequence)}: {action.action_type}")
                
                success = self.execute_action(action, timeout=self.action_timeout, initial_cam_tf=initial_cam_tf)
                
                if not success:
                    error_msg = f"Failed to execute action {i}: {action.action_type}"
                    self.logger.error(error_msg)
                    return False, error_msg
                    
                self.logger.info(f"Action {i} completed successfully")
            
            # Record executed skill data including ObjectInfo
            if self.data_recorder:
                self.data_recorder.record_executed_skill(skill_command, target_object, skill)
            
            # Log success and timing
            elapsed_time = time.time() - start_time
            success_msg = f"Successfully executed {skill_command} using {camera_type} camera in {elapsed_time:.2f}s"
            self.logger.info(success_msg)
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Skill execution failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
            
        finally:
            self.execution_start_time = None
    
    def execute_skill_sequence(self, skill_sequence: List[Tuple[str, str]], task_context: str = None) -> Tuple[bool, str]:
        """
        Execute a sequence of skills with their parameters.
        Maintains a dictionary of detected objects and executes skills in order.
        
        Args:
            skill_sequence: List of tuples containing (skill_name, parameters)
            task_context: Optional full task description for context
            
        Returns:
            Tuple[bool, str]: Success status and result message
        """
        try:
            start_time = time.time()
            self.execution_start_time = start_time
            camera_type = "ZED" if self.use_zed_camera else "RealSense"
            self.logger.info(f"Starting skill sequence execution with {camera_type} camera: {len(skill_sequence)} skills")
            
            # Dictionary to track detected objects
            detected_objects = {}
            
            # List to track executed skills for context
            executed_skills = []
            
            # Reset skill generation file tracking for new sequence
            if hasattr(self, 'skill_handler') and self.skill_handler:
                if hasattr(self.skill_handler, 'skill_generator') and self.skill_handler.skill_generator:
                    self.skill_handler.skill_generator.reset_generation_tracking()
                    self.logger.info("Reset skill generation file tracking for new task sequence")
            
            # Execute skills in sequence
            for i, (skill_name, parameters) in enumerate(skill_sequence):
                self.logger.info(f"Executing skill {i+1}/{len(skill_sequence)}: {skill_name} {parameters}")
                time.sleep(1)
                # Handle detect_object skill - perform detection and store result
                if skill_name.lower() == 'detect_object':
                    target_object = parameters.strip()
                    if not target_object:
                        error_msg = f"detect_object requires an object name parameter for skill {i+1}"
                        self.logger.error(error_msg)
                        return False, error_msg
                    
                    # Move to home position before object detection
                    self.logger.info("Moving to home position before object detection")
                    try:
                        # self.motion_planner.arm.move_gohome()
                        self.logger.info("Successfully moved to home position")
                    except Exception as e:
                        self.logger.warning(f"Failed to move to home position: {str(e)}")
                    
                    # Get camera images for object detection
                    color_image, depth_image = self.camera.get_frames()
                    if color_image is None or depth_image is None:
                        error_msg = f"Failed to get {camera_type} camera images for detect_object in skill {i+1}"
                        self.logger.error(error_msg)
                        return False, error_msg
                    
                    # Detect the target object
                    self.logger.info(f"Detecting target object: {target_object}")
                    object_info = None
                    for attempt in range(DETECTION_RETRIES):
                        try:
                            object_info = self.perception_system.detect_object(
                                target_object=target_object,
                                image=color_image,
                                depth_image=depth_image
                            )
                            if object_info is not None:
                                self.logger.info(f"Successfully detected {target_object}")
                                # Store the detection in our dictionary
                                detected_objects[target_object] = object_info
                                break
                            else:
                                self.logger.warning(f"Could not detect {target_object} (attempt {attempt + 1}/{DETECTION_RETRIES})")
                        except Exception as e:
                            self.logger.warning(f"Error detecting {target_object} (attempt {attempt + 1}): {str(e)}")
                    
                    if object_info is None:
                        error_msg = f"Failed to detect object {target_object} in skill {i+1}"
                        self.logger.error(error_msg)
                        return False, error_msg
                    
                    self.logger.info(f"Successfully executed detect_object for {target_object}")
                    
                    # Add successfully executed detect_object to context for future skills
                    executed_skills.append({
                        'skill_command': f"detect_object {target_object}",
                        'skill_name': 'detect_object',
                        'parameters': target_object,
                        'target_object': target_object
                    })
                    continue
                
                # Handle manipulation skills - check if required objects have been detected
                # Validate skill command
                is_valid, error_msg, params = self.validate_skill_command(skill_name.lower(), parameters)
                if not is_valid:
                    error_msg = f"Validation failed for skill {i+1} ({skill_name}): {error_msg}"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # Handle place skill specially - only needs target location detected, uses target location for skill generation
                if skill_name.lower() == 'place':
                    source_object = params[0]  # Object being placed (assumed to be in gripper)
                    target_location = params[1]  # Location where object is placed
                    target_location = target_location.strip()
                    # Only check if target location has been detected (source object is assumed to be held by robot)
                    if target_location not in detected_objects:
                        error_msg = f"Target location '{target_location}' has not been detected for place skill {i+1}. Use detect_object {target_location} first."
                        self.logger.error(error_msg)
                        return False, error_msg
                    
                    # Use target location object info for skill generation (important for placement)
                    target_object = target_location
                    object_info = detected_objects[target_location]
                    self.logger.info(f"Using detected object info for target location: {target_location} (source object '{source_object}' assumed to be held by robot)")
                else:
                    # For other manipulation skills, check if target object has been detected
                    target_object = params[0]
                    if target_object not in detected_objects:
                        error_msg = f"Target object '{target_object}' has not been detected for skill {i+1} ({skill_name}). Use detect_object {target_object} first."
                        self.logger.error(error_msg)
                        return False, error_msg
                    
                    # Get the detected object info
                    object_info = detected_objects[target_object]
                    self.logger.info(f"Using detected object info for {target_object}")
                
                # Get camera images for skill generation
                color_image, depth_image = self.camera.get_frames()
                if color_image is None or depth_image is None:
                    error_msg = f"Failed to get {camera_type} camera images for skill {i+1}"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # Generate skill command
                skill_command = (f"place {params[0]} on {params[1]}" 
                            if skill_name == 'place' 
                            else f"{skill_name} {params[0]}")
                
                # Generate skill using synchronous interface
                try:
                    self.logger.info(f"Generating skill for: {skill_command}")
                    skill = self.skill_handler.instantiate_skill(
                        skill_command,
                        target_object,
                        color_image,
                        depth_image,
                        object_info,
                        executed_skills,
                        task_context=task_context
                    )
                    
                    if skill is None:
                        error_msg = f"Failed to generate skill {i+1} for: {skill_command}"
                        self.logger.error(error_msg)
                        return False, error_msg
                    
                    self.logger.info(f"Successfully generated skill {i+1}")

                    # Save manipulation specification file
                    self.save_manipulation_spec(skill_command, target_object, skill, task_context=task_context)

                    # Execute the skill immediately
                    # Get camera transform for execution
                    if self.use_zed_camera and self.zed_to_robot_transform is not None:
                        initial_cam_tf = Rotation.from_matrix(self.zed_to_robot_transform[:3, :3])
                    else:
                        _, initial_cam_tf = self.motion_planner.get_camera_transform()
                    # Get point cloud from the main camera
                    # pcd = self.main_camera.get_point_cloud()
                    # if pcd is not None:
                    #     self.logger.info(f"!!!!!!!!!!!!!!!!!! Retrieved point cloud from {camera_type} camera")
                    #     # Note: Point cloud coordinates depend on camera type
                    #     # self.motion_planner.update_dynamic_collision_objects(pcd)
                    # else:
                    #     print(f"!!!!!!!!!!!!!!!!!! NO PCD FOR COLLISION OBJECTS")
                    # Execute skill actions
                    for j, action in enumerate(skill.action_sequence, 1):
                        self.logger.info(f"Executing action {j}/{len(skill.action_sequence)} of skill {i+1}: {action.action_type}")
                        
                        success = self.execute_action(action, timeout=self.action_timeout, initial_cam_tf=initial_cam_tf, is_place="place" in skill_command.lower())
                        
                        if not success:
                            error_msg = f"Failed to execute action {j} of skill {i+1}: {action.action_type}"
                            self.logger.error(error_msg)
                            return False, error_msg
                            
                        self.logger.info(f"Action {j} of skill {i+1} completed successfully")
                    
                    self.logger.info(f"Skill {i+1} completed successfully")
                    
                    # Add successfully executed skill to context for future skills
                    executed_skills.append({
                        'skill_command': skill_command,
                        'skill_name': skill_name,
                        'parameters': parameters,
                        'target_object': target_object
                    })
                    
                except Exception as e:
                    error_msg = f"Skill execution failed for skill {i+1}: {str(e)}"
                    self.logger.error(error_msg)
                    return False, error_msg
            
            # Log success and timing
            elapsed_time = time.time() - start_time
            success_msg = f"Successfully executed skill sequence of {len(skill_sequence)} skills using {camera_type} camera in {elapsed_time:.2f}s"
            self.logger.info(success_msg)
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Skill sequence execution failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
            
        finally:
            self.execution_start_time = None
    
    def save_manipulation_spec(self, skill_command: str, target_object: str, skill, task_context: str = None):
        """
        Save a manipulation specification file containing the primitive sequence
        and instantiated parameters converted to the world (robot base) frame.

        Args:
            skill_command: The skill command string (e.g., "pickup cup")
            target_object: Name of the target object
            skill: InstantiatedSkill instance with action_sequence
            task_context: Optional task context string
        """
        try:
            from pathlib import Path
            import os

            # Create output directory
            spec_dir = Path("manipulation_specs")
            spec_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = spec_dir / f"manip_spec_{timestamp}_{skill_command.replace(' ', '_')}.json"

            # Build primitive sequence list
            primitive_sequence = []
            for i, action in enumerate(skill.action_sequence):
                primitive_sequence.append({
                    "index": i,
                    "action_type": action.action_type,
                    "raw_parameters": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in action.parameters.items()
                        if k != 'surface_normal' and k != 'pivot_position' and k != 'object_bbox' and k != 'object_info'
                    }
                })

            # Build instantiated parameters for each action in world frame
            instantiated_actions = []
            for i, action in enumerate(skill.action_sequence):
                action_spec = {
                    "index": i,
                    "action_type": action.action_type,
                }

                # Continuous parameters (positions, orientations, forces)
                continuous_params = {}

                # Convert position from camera frame to world frame
                if action.position is not None:
                    cam_position = action.position.tolist() if isinstance(action.position, np.ndarray) else action.position
                    continuous_params["position_camera_frame"] = cam_position

                    try:
                        world_pos, world_ori = self.motion_planner.convert_cam_pose_to_base(
                            action.position,
                            [0, 0, 0, 1],  # identity orientation for position-only conversion
                            do_translation=True,
                            debug=False
                        )
                        continuous_params["position_world_frame"] = (
                            world_pos.tolist() if isinstance(world_pos, np.ndarray) else world_pos
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not convert position to world frame for action {i}: {e}")
                        continuous_params["position_world_frame"] = None

                # Convert orientation
                if action.orientation is not None:
                    ori = action.orientation.tolist() if isinstance(action.orientation, np.ndarray) else action.orientation
                    continuous_params["orientation_camera_frame"] = ori

                    # Convert orientation to world frame
                    if action.position is not None:
                        try:
                            _, world_ori = self.motion_planner.convert_cam_pose_to_base(
                                action.position,
                                action.orientation,
                                do_translation=True,
                                debug=False
                            )
                            continuous_params["orientation_world_frame"] = (
                                world_ori.tolist() if isinstance(world_ori, np.ndarray) else world_ori
                            )
                        except Exception as e:
                            self.logger.warning(f"Could not convert orientation to world frame for action {i}: {e}")
                            continuous_params["orientation_world_frame"] = None

                # Surface normal in world frame
                surface_normal = action.parameters.get('surface_normal', None)
                if surface_normal is not None:
                    sn = surface_normal.tolist() if isinstance(surface_normal, np.ndarray) else surface_normal
                    continuous_params["surface_normal_camera_frame"] = sn

                    try:
                        # Transform normal as a direction vector (no translation)
                        world_normal, _ = self.motion_planner.convert_cam_pose_to_base(
                            surface_normal, [0, 0, 0, 1], do_translation=False, debug=False
                        )
                        continuous_params["surface_normal_world_frame"] = (
                            world_normal.tolist() if isinstance(world_normal, np.ndarray) else world_normal
                        )
                    except Exception as e:
                        continuous_params["surface_normal_world_frame"] = None

                # Pivot position in world frame
                pivot_pos = action.parameters.get('pivot_position', None)
                if pivot_pos is not None:
                    pp = pivot_pos.tolist() if isinstance(pivot_pos, np.ndarray) else pivot_pos
                    continuous_params["pivot_position_camera_frame"] = pp

                    try:
                        world_pivot, _ = self.motion_planner.convert_cam_pose_to_base(
                            pivot_pos, [0, 0, 0, 1], do_translation=True, debug=False
                        )
                        continuous_params["pivot_position_world_frame"] = (
                            world_pivot.tolist() if isinstance(world_pivot, np.ndarray) else world_pivot
                        )
                    except Exception as e:
                        continuous_params["pivot_position_world_frame"] = None

                # Other continuous params
                if 'speed' in action.parameters:
                    continuous_params["speed"] = action.parameters['speed']
                if 'angular_velocity' in action.parameters:
                    continuous_params["angular_velocity"] = action.parameters['angular_velocity']
                if 'rotation_angle' in action.parameters:
                    continuous_params["rotation_angle"] = action.parameters['rotation_angle']

                # Discrete parameters (action types, grasp types, boolean flags, labels)
                discrete_params = {}
                if action.action_type == 'move_gripper_to_pose':
                    discrete_params["is_top_down_grasp"] = action.is_top_down_grasp
                    discrete_params["is_side_grasp"] = action.is_side_grasp

                if 'is_parallel' in action.parameters:
                    discrete_params["is_parallel"] = action.parameters['is_parallel']
                if 'is_button' in action.parameters:
                    discrete_params["is_button"] = action.parameters['is_button']
                if 'has_pivot' in action.parameters:
                    discrete_params["has_pivot"] = action.parameters['has_pivot']
                if 'surface_label' in action.parameters:
                    discrete_params["surface_label"] = action.parameters['surface_label']
                if 'surface_keywords' in action.parameters:
                    discrete_params["surface_keywords"] = action.parameters['surface_keywords']
                if 'hinge_location' in action.parameters:
                    discrete_params["hinge_location"] = action.parameters['hinge_location']
                if 'direction' in action.parameters:
                    discrete_params["direction"] = action.parameters['direction']
                if 'point_label' in action.parameters:
                    discrete_params["point_label"] = action.parameters['point_label']

                if action.pixel_position is not None:
                    discrete_params["pixel_position"] = list(action.pixel_position)

                action_spec["continuous_parameters"] = continuous_params
                action_spec["discrete_parameters"] = discrete_params
                instantiated_actions.append(action_spec)

            # Assemble full spec
            spec = {
                "timestamp": timestamp,
                "skill_command": skill_command,
                "skill_name": skill.skill_name,
                "target_object": target_object,
                "task_context": task_context,
                "object_pixel_pose": list(skill.object_pixel_pose) if skill.object_pixel_pose else None,
                "primitive_sequence": primitive_sequence,
                "instantiated_actions": instantiated_actions,
            }

            with open(filename, 'w') as f:
                json.dump(spec, f, indent=2, default=str)

            self.logger.info(f"Saved manipulation specification to {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save manipulation specification: {e}")
            traceback.print_exc()

    def set_data_recorder(self, data_recorder):
        """Set the data recorder for motion tracking"""
        self.data_recorder = data_recorder
        self.logger.info("Data recorder set for motion tracking")
    
    def shutdown(self):
        """Clean shutdown of the system"""
        self.logger.info("Shutting down direct skill execution system")
        
        # Stop the main camera
        if hasattr(self, 'main_camera'):
            self.main_camera.stop()
            camera_type = "ZED" if self.use_zed_camera else "RealSense"
            self.logger.info(f"{camera_type} camera (main) stopped")
        
        # Stop the secondary camera if it exists
        if hasattr(self, 'secondary_camera') and self.secondary_camera is not None:
            self.secondary_camera.stop()
            self.logger.info("Secondary camera stopped")
        
        # Disconnect from robot
        if hasattr(self, 'motion_planner'):
            self.motion_planner.disconnect_robot()
            self.logger.info("Robot disconnected")
    
    def get_robot_status(self):
        """Get current robot status"""
        if hasattr(self, 'motion_planner'):
            return self.motion_planner.get_robot_status()
        return None
    
    def move_to_home(self):
        """Move robot to home position"""
        if hasattr(self, 'motion_planner'):
            return self.motion_planner.move_to_home(execute=True)
        return False, None, None
    
    def get_transform_status(self):
        """Get current transform calibration status"""
        if not self.use_zed_camera:
            return {
                'transform_available': True,
                'transform_confidence': 1.0,
                'transform_matrix': None,
                'camera_type': 'RealSense',
                'note': 'No transform needed - camera is robot-mounted'
            }
        
        return {
            'transform_available': self.zed_to_robot_transform is not None,
            'transform_confidence': self.transform_confidence,
            'transform_matrix': self.zed_to_robot_transform.tolist() if self.zed_to_robot_transform is not None else None,
            'camera_type': 'ZED',
            'calibration_method': 'simplified_stereo_aruco'
        }
    
    def get_execution_data(self) -> Dict[str, Any]:
        """
        Get execution data for recording purposes
        
        Returns:
            Dictionary containing execution data including points of interest and images
        """
        execution_data = {}
        
        # Collect points of interest from current object info
        if hasattr(self, 'current_object_info') and self.current_object_info:
            points_data = {
                'pixel_coords': [],
                'ids': [],
                'scores': [],
                'detection_methods': [],
                'interaction_types': [],
                'confidences': []
            }
            
            # Extract points from object_info if available
            if hasattr(self.current_object_info, 'points') and self.current_object_info.points:
                points = self.current_object_info.points
                if 'pixel_coords' in points:
                    points_data['pixel_coords'] = points['pixel_coords']
                if 'ids' in points:
                    points_data['ids'] = points['ids']
                if 'scores' in points:
                    points_data['scores'] = points['scores']
                if 'detection_methods' in points:
                    points_data['detection_methods'] = points['detection_methods']
                elif 'detection_method' in points:
                    points_data['detection_methods'] = points['detection_method']
                if 'interaction_types' in points:
                    points_data['interaction_types'] = points['interaction_types']
                if 'confidences' in points:
                    points_data['confidences'] = points['confidences']
            
            execution_data['points_of_interest'] = points_data
        
        # Collect surface images if available
        if hasattr(self, 'current_object_info') and self.current_object_info:
            surface_images = {}
            
            # Add the main object image if available
            if hasattr(self.current_object_info, 'image') and self.current_object_info.image is not None:
                surface_images['main_object'] = self.current_object_info.image
            
            # Add surface masks as images if available
            if hasattr(self.current_object_info, 'surface_masks') and self.current_object_info.surface_masks:
                for surface_name, mask in self.current_object_info.surface_masks.items():
                    if mask is not None and hasattr(self.current_object_info, 'image'):
                        # Create masked image
                        masked_image = self.current_object_info.image.copy()
                        masked_image[~mask] = 0  # Set non-mask areas to black
                        surface_images[surface_name] = masked_image
            
            if surface_images:
                execution_data['surface_images'] = surface_images
        
        # Add timing information if available
        if hasattr(self, 'execution_start_time') and self.execution_start_time:
            execution_data['execution_start_time'] = self.execution_start_time
            execution_data['current_time'] = time.time()
            execution_data['execution_duration'] = time.time() - self.execution_start_time
        
        # Add camera type and robot info
        execution_data['camera_type'] = "ZED" if self.use_zed_camera else "RealSense"
        execution_data['robot_ip'] = getattr(self, 'robot_ip', 'unknown')
        
        # Collect skill generator file information
        if hasattr(self, 'skill_handler') and self.skill_handler:
            if hasattr(self.skill_handler, 'skill_generator') and self.skill_handler.skill_generator:
                skill_gen_files = self.skill_handler.skill_generator.get_last_generation_files()
                execution_data['skill_generation_files'] = skill_gen_files
        
        return execution_data

    def recalibrate_transform(self):
        """Recalibrate the camera-to-robot transform (only available for ZED camera)"""
        if not self.use_zed_camera:
            print("Transform calibration only available when using ZED camera")
            return False
        return self.calibrate_zed_to_robot_transform()

    def get_camera_info(self):
        """Get information about the current camera configuration"""
        camera_type = "ZED" if self.use_zed_camera else "RealSense"
        info = {
            'primary_camera': camera_type,
            'camera_params': self.camera_params,
            'main_camera_active': hasattr(self, 'main_camera') and self.main_camera is not None,
            'secondary_camera_active': hasattr(self, 'secondary_camera') and self.secondary_camera is not None,
        }
        
        if self.use_zed_camera:
            info['transform_calibration'] = self.get_transform_status()
        
        return info

    def adjust_tcp_pose_for_top_down_grasp(self, target_position, object_mask=None, search_radius_m=0.03, 
                                           depth_threshold_ratio=0.05, cluster_min_size=5, tcp_standoff_m=0.02):
        """
        Adjust TCP pose for top-down grasps by finding the topmost point within the object mask region
        and centering it on the depth surface.
        
        Args:
            target_position: Original target position [x, y, z] in camera/world coordinates
            object_mask: Binary mask of the object region (if None, will use search radius)
            search_radius_m: Physical search radius in meters (default 0.03 = 3cm for top-down grasps)
            depth_threshold_ratio: Ratio of average depth to use as clustering threshold (default 0.05 = 5%)
            cluster_min_size: Minimum cluster size for consideration (default 5 pixels)
            
        Returns:
            adjusted_position: Position adjusted to center on the topmost point within object mask
        """
        try:
            if self.latest_depth_image is None or self.latest_color_image is None:
                self.logger.warning("No depth/color image available for top-down grasp adjustment")
                return target_position
                
            # Get camera intrinsics
            fx = self.main_camera.intrinsics.fx
            fy = self.main_camera.intrinsics.fy
            ppx = self.main_camera.intrinsics.ppx
            ppy = self.main_camera.intrinsics.ppy
            
            # Get depth scale
            depth_scale = getattr(self.main_camera, 'depth_scale', 1000.0)
            
            self.logger.info(f"DEBUG - Top-down grasp adjustment: target_position={target_position}")
            
            # Convert target position to pixel coordinates
            if isinstance(target_position, np.ndarray):
                pos = target_position.copy()
            else:
                pos = np.array(target_position)
                
            # Project 3D point to 2D pixel coordinates
            if len(pos) >= 3 and pos[2] > 0:  # Valid depth
                pixel_x = int(fx * pos[0] / pos[2] + ppx)
                pixel_y = int(fy * pos[1] / pos[2] + ppy)
            else:
                self.logger.warning("Invalid target position for top-down grasp adjustment")
                return target_position
                
            # Ensure pixel coordinates are within image bounds
            h, w = self.latest_depth_image.shape
            pixel_x = max(0, min(w-1, pixel_x))
            pixel_y = max(0, min(h-1, pixel_y))
            
            # Define search region
            if object_mask is not None:
                # Use object mask if provided
                search_mask = object_mask
                self.logger.info(f"DEBUG - Using provided object mask, shape: {search_mask.shape}")
            else:
                # Create circular search region around target pixel
                target_depth_m = pos[2] if pos[2] > 0 else 1.0
                radius_pixels_x = int(search_radius_m * fx / target_depth_m)
                radius_pixels_y = int(search_radius_m * fy / target_depth_m)
                
                x_min = max(0, pixel_x - radius_pixels_x)
                x_max = min(w, pixel_x + radius_pixels_x)
                y_min = max(0, pixel_y - radius_pixels_y) 
                y_max = min(h, pixel_y + radius_pixels_y)
                
                # Create circular mask within the search region
                search_mask = np.zeros((h, w), dtype=bool)
                center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
                radius = min(radius_pixels_x, radius_pixels_y)
                
                y_coords, x_coords = np.ogrid[:h, :w]
                mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                search_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
                
                self.logger.info(f"DEBUG - Created circular search mask, radius: {radius} pixels")
            
            # Extract depth region within the search mask
            depth_region = self.latest_depth_image.copy()
            valid_mask = (depth_region > 100) & (depth_region < 10000) & search_mask  # Valid depth + object mask
            
            if not np.any(valid_mask):
                self.logger.warning("No valid depth values in object mask region")
                return target_position
                
            # Get valid depth values and their coordinates within the object mask
            valid_depths = depth_region[valid_mask]
            y_coords, x_coords = np.where(valid_mask)
            
            self.logger.info(f"DEBUG - Valid depth points in object mask: {len(valid_depths)}")
            self.logger.info(f"DEBUG - Valid depth range: {valid_depths.min():.1f} to {valid_depths.max():.1f} mm")
            
            # For top-down grasps, we want to find the TOPMOST point (highest Y coordinate in camera frame)
            # This corresponds to the highest point on the object surface
            if len(valid_depths) >= cluster_min_size:
                # Calculate dynamic depth threshold
                average_depth = np.mean(valid_depths)
                depth_threshold = average_depth * depth_threshold_ratio
                
                self.logger.info(f"DEBUG - Average depth in object region: {average_depth:.1f} mm")
                self.logger.info(f"DEBUG - Depth threshold: {depth_threshold:.1f} mm")
                
                # Find points near the surface (within depth threshold of the closest point)
                min_depth = np.min(valid_depths)
                surface_threshold = min_depth + depth_threshold
                
                # Select points on the surface
                surface_mask = valid_depths <= surface_threshold
                surface_depths = valid_depths[surface_mask]
                surface_x_coords = x_coords[surface_mask]
                surface_y_coords = y_coords[surface_mask]
                
                if len(surface_depths) >= 3:
                    self.logger.info(f"DEBUG - Surface points: {len(surface_depths)}")
                    
                    # For top-down grasps, find the CENTER of the surface region (centroid)
                    # This gives better grasp positioning than edge points
                    
                    # Calculate centroid of surface points in pixel coordinates
                    centroid_x = np.mean(surface_x_coords).astype(int)
                    centroid_y = np.mean(surface_y_coords).astype(int)
                    
                    # Use median depth for robustness against outliers
                    centroid_depth = np.median(surface_depths)
                    
                    self.logger.info(f"DEBUG - Surface centroid: ({centroid_x}, {centroid_y}), depth: {centroid_depth:.1f} mm")
                    
                    # Alternative: If we want the point closest to the actual surface center
                    # Find the surface point closest to the computed centroid
                    distances_to_centroid = np.sqrt((surface_x_coords - centroid_x)**2 + 
                                                   (surface_y_coords - centroid_y)**2)
                    closest_to_centroid_idx = np.argmin(distances_to_centroid)
                    
                    # Use the actual surface point closest to centroid for better depth accuracy
                    center_x = surface_x_coords[closest_to_centroid_idx]
                    center_y = surface_y_coords[closest_to_centroid_idx] 
                    center_depth = surface_depths[closest_to_centroid_idx]
                    
                    self.logger.info(f"DEBUG - Actual center point: ({center_x}, {center_y}), depth: {center_depth:.1f} mm")
                    
                    # Convert depth from mm to meters
                    center_depth_m = center_depth / 1000.0  # Convert mm to meters
                    
                    # Convert center pixel back to 3D camera coordinates
                    center_x_3d = (center_x - ppx) * center_depth_m / fx
                    center_y_3d = (center_y - ppy) * center_depth_m / fy
                    
                    self.logger.info(f"DEBUG - Center 3D coordinates: ({center_x_3d:.4f}, {center_y_3d:.4f}, {center_depth_m:.4f})")
                    
                    # Create adjusted position centered on the surface center
                    adjusted_position = [
                        center_x_3d,  # X coordinate of surface center
                        center_y_3d,  # Y coordinate of surface center
                        center_depth_m  # Z coordinate (depth) of surface center
                    ]
                    
                    # Calculate adjustment offset for logging
                    original_pos = np.array(target_position[:3])
                    final_offset = np.array(adjusted_position) - original_pos
                    final_offset_magnitude = np.linalg.norm(final_offset)
                    
                    self.logger.info(f"Top-down grasp adjustment: offset={final_offset}, magnitude={final_offset_magnitude:.4f}m")
                    self.logger.info(f"Original position: {target_position[:3]}")
                    self.logger.info(f"Adjusted position: {adjusted_position}")
                    
                    return adjusted_position
                else:
                    self.logger.warning("Not enough surface points for top-down grasp adjustment")
                    return target_position
            else:
                self.logger.warning("Not enough valid depth points for top-down grasp adjustment")
                return target_position
                
        except Exception as e:
            self.logger.error(f"Error in top-down grasp adjustment: {e}")
            return target_position

    def set_data_recorder(self, data_recorder):
        """
        Set the data recorder for motion tracking during skill execution
        
        Args:
            data_recorder: DataRecorder instance for tracking motion data
        """
        self.data_recorder = data_recorder
        self.logger.info("Data recorder set for motion tracking")

    def execute_stored_skill(self, skill_json_path: str) -> Tuple[bool, Optional[str]]:
        """
        Execute a stored skill from a JSON file
        
        Args:
            skill_json_path: Path to the skill JSON file
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            import json
            import cv2
            import os
            
            # Load the skill JSON file
            with open(skill_json_path, 'r') as f:
                skill_data = json.load(f)
            
            # Set current skill name for placement detection
            skill_filename = os.path.basename(skill_json_path)
            self.current_skill_name = skill_data.get('name', skill_filename.replace('.json', ''))
            self.logger.info(f"Executing stored skill: {self.current_skill_name}")
            
            # Extract the primitive sequence from the skill data
            if 'primitive_sequence' not in skill_data:
                return False, f"No primitive_sequence found in skill file: {skill_json_path}"
            
            primitive_sequence = skill_data['primitive_sequence']
            points_of_interest = skill_data.get('points_of_interest', {})
            surface_info = skill_data.get('surface_info', {})
            object_bbox = skill_data.get('object_bbox', None)
            image_id = skill_data.get('image_id', '')
            
            self.logger.info(f"Executing stored skill: {skill_data.get('name', 'unknown')}")
            self.logger.info(f"Primitive sequence: {primitive_sequence}")
            self.logger.info(f"Image ID: {image_id}")
            if surface_info:
                self.logger.info(f"Surface info available: {list(surface_info.keys())}")
            if object_bbox:
                self.logger.info(f"Object bounding box available: {object_bbox}")
            
            # Set up current_object_info with stored bounding box for pivot calculation
            if object_bbox is not None:
                # Create a mock ObjectInfo with the stored bounding box
                from types import SimpleNamespace
                self.current_object_info = SimpleNamespace()
                self.current_object_info.bbox = object_bbox
                self.current_object_info.mask = None  # Could be populated if needed
                self.logger.info(f"Set current_object_info with stored bounding box: {object_bbox}")
            else:
                self.current_object_info = None
                self.logger.info("No object bounding box available from stored skill")
            
            # For now, use stored pixel coordinates with live camera depth
            # This is a compromise solution until we have proper 3D coordinates stored
            self.logger.info("Using stored skill coordinates with live camera depth")
            
            # Always capture fresh camera images for each skill execution
            # This ensures depth data reflects current environment state after previous skill changes
            if self.depth_capture_delay > 0:
                self.logger.info(f"Waiting {self.depth_capture_delay} seconds before capturing depth data...")
                time.sleep(self.depth_capture_delay)
            
            self.logger.info("Capturing fresh camera frames for current environment state...")
            try:
                if hasattr(self.camera, 'get_frames'):
                    frames = self.camera.get_frames()
                    if frames:
                        self.latest_color_image, self.latest_depth_image = frames
                        self.logger.info("Successfully captured fresh camera frames for current skill")
                    else:
                        self.logger.error("Failed to capture camera frames")
                        return False, "Could not capture camera frames"
                else:
                    self.logger.error("Camera does not have get_frames method")
                    return False, "Camera get_frames method not available"
            except Exception as e:
                self.logger.error(f"Error capturing camera frames: {e}")
                return False, f"Error capturing frames: {str(e)}"
            
            # Fallback to original behavior if no stored images
            for primitive in primitive_sequence:
                self.logger.info(f"Executing primitive: {primitive}")
                success = self.execute_primitive_command(primitive, points_of_interest, surface_info)
                if not success:
                    return False, f"Failed to execute primitive: {primitive}"
            
            return True, None
                
        except Exception as e:
            return False, f"Failed to execute stored skill {skill_json_path}: {str(e)}"
        finally:
            # Clear current skill name and object info
            self.current_skill_name = None
            self.current_object_info = None

    def _load_stored_skill_images(self, skill_json_path: str, image_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load stored images associated with a skill
        
        Args:
            skill_json_path: Path to the skill JSON file
            image_id: Image ID from the skill data
            
        Returns:
            Dictionary with 'rgb' and 'depth' images, or None if failed
        """
        try:
            import cv2
            import os
            
            # Determine images directory path
            skill_dir = os.path.dirname(skill_json_path)
            images_dir = os.path.join(skill_dir, 'images')
            
            # Construct image file paths
            rgb_path = os.path.join(images_dir, f"{image_id}.png")
            depth_path = os.path.join(images_dir, f"depth_image_{image_id}.png")
            
            self.logger.info(f"Looking for RGB image: {rgb_path}")
            self.logger.info(f"Looking for depth image: {depth_path}")
            
            # Load RGB image
            if not os.path.exists(rgb_path):
                self.logger.error(f"RGB image not found: {rgb_path}")
                return None
                
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                self.logger.error(f"Failed to load RGB image: {rgb_path}")
                return None
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Load depth image
            if not os.path.exists(depth_path):
                self.logger.error(f"Depth image not found: {depth_path}")
                return None
                
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                self.logger.error(f"Failed to load depth image: {depth_path}")
                return None
            
            # Handle depth image format - stored images are often saved as RGB for visualization
            if len(depth_image.shape) == 3:
                if depth_image.shape[2] == 3:
                    # For visualized depth images, convert to grayscale first
                    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
                elif depth_image.shape[2] == 1:
                    depth_image = depth_image[:, :, 0]
                
            # Since these are stored depth images that were converted for visualization,
            # we need to convert them back to proper depth values
            # The depth scale is typically applied during visualization, so we may need to reverse it
            if depth_image.dtype == np.uint8:
                # Convert 8-bit grayscale back to depth values
                # This is an approximation - the original depth scaling may be lost
                depth_image = depth_image.astype(np.uint16) * 65535 // 255
            elif depth_image.dtype != np.uint16:
                depth_image = depth_image.astype(np.uint16)
            
            self.logger.info(f"Successfully loaded stored images - RGB: {rgb_image.shape}, Depth: {depth_image.shape}")
            
            return {
                'rgb': rgb_image,
                'depth': depth_image
            }
            
        except Exception as e:
            self.logger.error(f"Error loading stored skill images: {e}")
            return None

    def execute_primitive_command(self, primitive: str, points_of_interest: dict, surface_info: dict = None) -> bool:
        """
        Execute a single primitive command with points of interest and surface information
        
        Args:
            primitive: Primitive command string (e.g., "move_gripper_to_pose('tio', true, false)")
            points_of_interest: Dictionary of available points of interest
            surface_info: Dictionary of available surface information (optional)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Parse the primitive command
            import re
            
            if primitive.strip().startswith("move_gripper_to_pose("):
                # Extract parameters from move_gripper_to_pose command
                match = re.match(r"move_gripper_to_pose\('([^']+)',\s*(\w+),\s*(\w+)\)", primitive.strip())
                if match:
                    poi_name = match.group(1)
                    is_top_down = match.group(2).lower() == 'true'
                    is_side_grasp = match.group(3).lower() == 'true'
                    
                    if poi_name in points_of_interest:
                        poi = points_of_interest[poi_name]
                        pixel_coords = poi.get('pixel_coords', [])
                        
                        if len(pixel_coords) >= 2:
                            # Store pixel coordinates for pivot calculations (matches original skill execution)
                            self.current_interaction_point_pixel = pixel_coords
                            self.logger.info(f"Set interaction point pixel coordinates for pivot calculation: {self.current_interaction_point_pixel}")
                            
                            # Convert to normalized coordinates for robot interface
                            if hasattr(self, 'latest_color_image') and self.latest_color_image is not None:
                                h, w = self.latest_color_image.shape[:2]
                                norm_x = pixel_coords[0] / w
                                norm_y = pixel_coords[1] / h
                            else:
                                # Use stored position if available
                                position = poi.get('position', [])
                                if len(position) >= 2:
                                    norm_x, norm_y = position[0], position[1]
                                else:
                                    self.logger.error(f"Cannot determine position for POI: {poi_name}")
                                    return False
                            
                            # Execute the movement based on the parameters
                            # Signal motion start to data recorder
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.start_robot_motion("move_gripper_to_pose")
                            
                            # Check if this specific primitive is a placement movement
                            is_placement_movement = False
                            if hasattr(self, 'current_skill_name') and self.current_skill_name:
                                skill_has_place = 'place' in self.current_skill_name.lower()
                                # For stored skills, if the skill itself is a placement skill, treat move_gripper_to_pose as placement
                                # POI names in stored skills are often generic identifiers, so we rely on skill name
                                is_placement_movement = skill_has_place
                                self.logger.info(f"Placement movement check - Skill name: '{self.current_skill_name}', has place: {skill_has_place}, applying placement offset: {is_placement_movement}")
                            
                            result = self._execute_grasp_movement(norm_x, norm_y, is_top_down, is_side_grasp, is_placement_movement)
                            
                            # Signal motion end to data recorder
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.end_robot_motion("move_gripper_to_pose")
                            
                            return result
                        else:
                            self.logger.error(f"Invalid pixel coordinates for POI: {poi_name}")
                            return False
                    else:
                        self.logger.error(f"Point of interest not found: {poi_name}")
                        return False
                else:
                    self.logger.error(f"Cannot parse move_gripper_to_pose command: {primitive}")
                    return False
                    
            elif primitive.strip() == "close_gripper()":
                self.logger.info("Attempting to close gripper...")
                # Signal motion start to data recorder
                if hasattr(self, 'data_recorder') and self.data_recorder:
                    self.data_recorder.start_robot_motion("close_gripper")
                    
                success = self.motion_planner.close_gripper()
                
                # Signal motion end to data recorder
                if hasattr(self, 'data_recorder') and self.data_recorder:
                    self.data_recorder.end_robot_motion("close_gripper")
                    
                if success:
                    self.logger.info("Gripper closed successfully")
                else:
                    self.logger.error("Failed to close gripper")
                return success
                
            elif primitive.strip() == "open_gripper()":
                self.logger.info("Attempting to open gripper...")
                # Signal motion start to data recorder
                if hasattr(self, 'data_recorder') and self.data_recorder:
                    self.data_recorder.start_robot_motion("open_gripper")
                    
                success = self.motion_planner.open_gripper()
                
                # Signal motion end to data recorder
                if hasattr(self, 'data_recorder') and self.data_recorder:
                    self.data_recorder.end_robot_motion("open_gripper")
                    
                if success:
                    self.logger.info("Gripper opened successfully")
                else:
                    self.logger.error("Failed to open gripper")
                return success
                
            elif primitive.strip() == "retract_gripper()":
                # Retract gripper by moving up 10cm
                self.logger.info("Retracting gripper...")
                # Signal motion start to data recorder
                if hasattr(self, 'data_recorder') and self.data_recorder:
                    self.data_recorder.start_robot_motion("retract_gripper")
                    
                speed_factor = 2.0 if self.fast_mode else 1.5  # Increased retract speeds
                success = self.motion_planner.retract_gripper(distance=0.1, speed_factor=speed_factor)
                
                # Signal motion end to data recorder
                if hasattr(self, 'data_recorder') and self.data_recorder:
                    self.data_recorder.end_robot_motion("retract_gripper")
                    
                return success
                
            elif primitive.strip().startswith("twist("):
                # Extract direction from twist command
                match = re.match(r"twist\('([^']+)'\)", primitive.strip())
                if match:
                    direction = match.group(1)
                    self.logger.info(f"Executing twist in direction: {direction}")
                    
                    # Execute wrist twist motion
                    if direction.lower() in ['counterclockwise', 'counter_clockwise']:
                        twist_direction = "counterclockwise"
                    elif direction.lower() == 'clockwise':
                        twist_direction = "clockwise"
                    else:
                        self.logger.error(f"Unknown twist direction: {direction}")
                        return False
                    
                    # Signal motion start to data recorder
                    if hasattr(self, 'data_recorder') and self.data_recorder:
                        self.data_recorder.start_robot_motion(f"twist_{twist_direction}")
                    
                    # Execute wrist twist with a moderate rotation (π/4 radians = 45 degrees)
                    success = self.motion_planner.execute_wrist_twist(
                        direction=twist_direction,
                        rotation_angle=np.pi,  # 45 degrees 
                        speed_factor=1
                    )
                    
                    # Signal motion end to data recorder
                    if hasattr(self, 'data_recorder') and self.data_recorder:
                        self.data_recorder.end_robot_motion(f"twist_{twist_direction}")
                    
                    return success
                else:
                    self.logger.error(f"Cannot parse twist command: {primitive}")
                    return False
                    
            elif primitive.strip().startswith("pull("):
                # Parse pull command with pivot support
                # Format: pull('point', 'direction', is_button, has_pivot, 'hinge_location')
                match = re.match(r"pull\('([^']+)',\s*'([^']+)',\s*(\w+),\s*(\w+),\s*'([^']*)'\)", primitive.strip())
                if match:
                    poi_name = match.group(1)
                    force_direction = match.group(2)
                    is_button = match.group(3).lower() == 'true'
                    has_pivot = match.group(4).lower() == 'true' 
                    hinge_location = match.group(5)
                    
                    self.logger.info(f"Executing pull: poi={poi_name}, direction={force_direction}, button={is_button}, pivot={has_pivot}, hinge={hinge_location}")
                    
                    # Check if poi_name is a surface ID rather than a point of interest
                    if surface_info and poi_name in surface_info:
                        # Use surface information
                        surface_data = surface_info[poi_name]
                        centroid = surface_data.get('centroid', [])
                        surface_normal = surface_data.get('normal', None)
                        
                        if len(centroid) >= 2:
                            # Convert centroid pixel coordinates to normalized coordinates
                            if hasattr(self, 'latest_color_image') and self.latest_color_image is not None:
                                h, w = self.latest_color_image.shape[:2]
                                norm_x = centroid[0] / w
                                norm_y = centroid[1] / h
                            else:
                                self.logger.error(f"Cannot determine image dimensions for surface centroid")
                                return False
                            
                            self.logger.info(f"Using surface '{poi_name}' with centroid at ({centroid[0]}, {centroid[1]}) -> normalized ({norm_x:.3f}, {norm_y:.3f})")
                            if surface_normal:
                                self.logger.info(f"Surface normal: {surface_normal}")
                            
                            # Signal motion start to data recorder  
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.start_robot_motion(f"pull_{poi_name}")
                            
                            # Execute pull movement with surface normal information
                            success = self._execute_pull_action(
                                norm_x, norm_y, 
                                force_direction, 
                                is_button, 
                                has_pivot, 
                                hinge_location,
                                surface_normal=surface_normal
                            )
                            
                            # Signal motion end to data recorder
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.end_robot_motion(f"pull_{poi_name}")
                            
                            return success
                        else:
                            self.logger.error(f"Invalid surface centroid for surface: {poi_name}")
                            return False
                    
                    elif poi_name in points_of_interest:
                        poi = points_of_interest[poi_name]
                        pixel_coords = poi.get('pixel_coords', [])
                        
                        if len(pixel_coords) >= 2:
                            # Convert pixel coordinates to normalized coordinates
                            if hasattr(self, 'latest_color_image') and self.latest_color_image is not None:
                                h, w = self.latest_color_image.shape[:2]
                                norm_x = pixel_coords[0] / w
                                norm_y = pixel_coords[1] / h
                            else:
                                # Use position if available
                                position = poi.get('position', [])
                                if len(position) >= 2:
                                    norm_x, norm_y = position[0], position[1]
                                else:
                                    self.logger.error(f"Cannot determine position for POI: {poi_name}")
                                    return False
                            
                            # Signal motion start to data recorder  
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.start_robot_motion(f"pull_{poi_name}")
                            
                            # Execute pull movement using the action execution logic
                            success = self._execute_pull_action(
                                norm_x, norm_y, 
                                force_direction, 
                                is_button, 
                                has_pivot, 
                                hinge_location
                            )
                            
                            # Signal motion end to data recorder
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.end_robot_motion(f"pull_{poi_name}")
                            
                            return success
                        else:
                            self.logger.error(f"Invalid pixel coordinates for POI: {poi_name}")
                            return False
                    else:
                        self.logger.error(f"Point of interest not found: {poi_name}")
                        return False
                else:
                    self.logger.error(f"Cannot parse pull command: {primitive}")
                    return False
                    
            elif primitive.strip().startswith("push("):
                # Parse push command with pivot support
                # Format: push('point', 'direction', is_button, has_pivot, 'hinge_location')
                match = re.match(r"push\('([^']+)',\s*'([^']+)',\s*(\w+),\s*(\w+),\s*'([^']*)'\)", primitive.strip())
                if match:
                    poi_name = match.group(1)
                    force_direction = match.group(2)
                    is_button = match.group(3).lower() == 'true'
                    has_pivot = match.group(4).lower() == 'true' 
                    hinge_location = match.group(5)
                    
                    self.logger.info(f"Executing push: poi={poi_name}, direction={force_direction}, button={is_button}, pivot={has_pivot}, hinge={hinge_location}")
                    
                    # Check if poi_name is a surface ID rather than a point of interest
                    if surface_info and poi_name in surface_info:
                        # Use surface information
                        surface_data = surface_info[poi_name]
                        centroid = surface_data.get('centroid', [])
                        surface_normal = surface_data.get('normal', None)
                        
                        if len(centroid) >= 2:
                            # Convert centroid pixel coordinates to normalized coordinates
                            if hasattr(self, 'latest_color_image') and self.latest_color_image is not None:
                                h, w = self.latest_color_image.shape[:2]
                                norm_x = centroid[0] / w
                                norm_y = centroid[1] / h
                            else:
                                self.logger.error(f"Cannot determine image dimensions for surface centroid")
                                return False
                            
                            self.logger.info(f"Using surface '{poi_name}' with centroid at ({centroid[0]}, {centroid[1]}) -> normalized ({norm_x:.3f}, {norm_y:.3f})")
                            if surface_normal:
                                self.logger.info(f"Surface normal: {surface_normal}")
                            
                            # Signal motion start to data recorder  
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.start_robot_motion(f"push_{poi_name}")
                            
                            # Execute push movement with surface normal information
                            success = self._execute_push_action(
                                norm_x, norm_y, 
                                force_direction, 
                                is_button, 
                                has_pivot, 
                                hinge_location,
                                surface_normal=surface_normal
                            )
                            
                            # Signal motion end to data recorder
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.end_robot_motion(f"push_{poi_name}")
                            
                            return success
                        else:
                            self.logger.error(f"Invalid surface centroid for surface: {poi_name}")
                            return False
                    
                    elif poi_name in points_of_interest:
                        poi = points_of_interest[poi_name]
                        pixel_coords = poi.get('pixel_coords', [])
                        
                        if len(pixel_coords) >= 2:
                            # Convert pixel coordinates to normalized coordinates
                            if hasattr(self, 'latest_color_image') and self.latest_color_image is not None:
                                h, w = self.latest_color_image.shape[:2]
                                norm_x = pixel_coords[0] / w
                                norm_y = pixel_coords[1] / h
                            else:
                                # Use position if available
                                position = poi.get('position', [])
                                if len(position) >= 2:
                                    norm_x, norm_y = position[0], position[1]
                                else:
                                    self.logger.error(f"Cannot determine position for POI: {poi_name}")
                                    return False
                            
                            # Signal motion start to data recorder  
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.start_robot_motion(f"push_{poi_name}")
                            
                            # Execute push movement using the action execution logic
                            success = self._execute_push_action(
                                norm_x, norm_y, 
                                force_direction, 
                                is_button, 
                                has_pivot, 
                                hinge_location
                            )
                            
                            # Signal motion end to data recorder
                            if hasattr(self, 'data_recorder') and self.data_recorder:
                                self.data_recorder.end_robot_motion(f"push_{poi_name}")
                            
                            return success
                        else:
                            self.logger.error(f"Invalid pixel coordinates for POI: {poi_name}")
                            return False
                    else:
                        self.logger.error(f"Point of interest not found: {poi_name}")
                        return False
                else:
                    self.logger.error(f"Cannot parse push command: {primitive}")
                    return False
                
            else:
                self.logger.error(f"Unknown primitive command: {primitive}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing primitive command {primitive}: {e}")
            return False

    def _execute_pull_action(self, norm_x: float, norm_y: float, force_direction: str, 
                            is_button: bool, has_pivot: bool, hinge_location: str, surface_normal=None) -> bool:
        """
        Execute a pull action using the existing pull/push logic from action execution
        """
        try:
            # Move to the target position first
            success = self._execute_grasp_movement(norm_x, norm_y, False, True, False)
            if not success:
                self.logger.error("Failed to move to pull position")
                return False
            
            # Calculate pivot point if needed
            pivot_point = None
            if has_pivot and hinge_location and hinge_location in ['top', 'bottom', 'left', 'right']:
                self.logger.info(f"Using hinge location: {hinge_location}")
                
                # Get current gripper position
                current_pose = self.motion_planner.get_robot_tcp_pose()
                if current_pose is not None:
                    current_position, _ = current_pose
                    
                    # Flatten the position array if it's nested
                    if hasattr(current_position, 'flatten'):
                        current_position = current_position.flatten()
                    elif isinstance(current_position, (list, tuple)) and len(current_position) > 0:
                        if isinstance(current_position[0], (list, tuple, np.ndarray)):
                            current_position = current_position[0]
                    
                    # Calculate pivot point based on hinge location using actual object bounding box
                    # COORDINATE SYSTEM: Y+ = right, Y- = left, X- = forward/top, X+ = back/bottom
                    
                    # Use actual object bounding box if available (from stored skill data)
                    if hasattr(self, 'current_object_info') and self.current_object_info and self.current_object_info.bbox:
                        bbox = self.current_object_info.bbox  # [x, y, w, h] format in pixels
                        self.logger.info(f"Using stored object bounding box for pivot calculation: {bbox}")
                        
                        bbox_x, bbox_y, bbox_w, bbox_h = bbox
                        
                        # Convert pixel coordinates to world coordinates
                        pixel_to_meter = self.estimate_pixel_to_meter_ratio(current_position)
                        
                        # Calculate pivot point using 3D world coordinates instead of pixel conversion
                        if hasattr(self, 'current_interaction_point_world') and self.current_interaction_point_world:
                            # Use 3D world coordinates of interaction point for accurate pivot calculation
                            interaction_world_x, interaction_world_y, interaction_world_z = self.current_interaction_point_world
                            
                            # Calculate 3D offset from interaction point to hinge edge using bounding box dimensions
                            # Convert bounding box dimensions to world units
                            bbox_width_world = bbox_w * pixel_to_meter
                            bbox_height_world = bbox_h * pixel_to_meter
                            
                            # Get interaction point pixel coordinates for offset calculation
                            if hasattr(self, 'current_interaction_point_pixel') and self.current_interaction_point_pixel:
                                interaction_px, interaction_py = self.current_interaction_point_pixel
                                
                                # CAMERA FRAME: X+ = right, Y+ = down
                                # ROBOT FRAME: Y+ = right, Y- = left, X- = forward/top, X+ = back/bottom
                                
                                # Calculate world position of the specific bounding box edge
                                if hinge_location == 'left':
                                    # Left edge: offset from interaction point to left edge of bbox
                                    pixel_offset_x = bbox_x - interaction_px
                                    world_offset_y = pixel_offset_x * pixel_to_meter  # Camera X -> Robot Y
                                    pivot_point = [interaction_world_x, interaction_world_y - world_offset_y, interaction_world_z]  # Left = negative Y
                                elif hinge_location == 'right':
                                    # Right edge: offset from interaction point to right edge of bbox
                                    pixel_offset_x = (bbox_x + bbox_w) - interaction_px
                                    world_offset_y = pixel_offset_x * pixel_to_meter  # Camera X -> Robot Y
                                    pivot_point = [interaction_world_x, interaction_world_y - world_offset_y, interaction_world_z]  # Right = negative Y
                                elif hinge_location == 'top':
                                    # Top edge: offset from interaction point to top edge of bbox
                                    pixel_offset_y = bbox_y - interaction_py
                                    world_offset_x = pixel_offset_y * pixel_to_meter  # Camera Y -> Robot X (inverted)
                                    pivot_point = [interaction_world_x - world_offset_x, interaction_world_y, interaction_world_z]  # Top = negative X
                                elif hinge_location == 'bottom':
                                    # Bottom edge: offset from interaction point to bottom edge of bbox
                                    pixel_offset_y = (bbox_y + bbox_h) - interaction_py
                                    world_offset_x = pixel_offset_y * pixel_to_meter  # Camera Y -> Robot X (inverted)
                                    pivot_point = [interaction_world_x + world_offset_x, interaction_world_y, interaction_world_z]  # Bottom = positive X
                                    
                                self.logger.info(f"Calculated pivot using 3D world coordinates: interaction_point={self.current_interaction_point_world}, hinge={hinge_location}, pivot={pivot_point}, offset={world_offset_y if 'world_offset_y' in locals() else world_offset_x:.3f}m")
                            else:
                                self.logger.warning("No interaction point pixel coordinates available for offset calculation")
                                pivot_point = None
                        else:
                            # Fallback: use middle of the specified edge of bounding box
                            self.logger.info("No interaction point stored, using middle of bounding box edge")
                            if hinge_location == 'left':
                                # Distance from center to left edge
                                offset_distance = bbox_w * 0.5 * pixel_to_meter
                                pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                            elif hinge_location == 'right':
                                # Distance from center to right edge
                                offset_distance = bbox_w * 0.5 * pixel_to_meter
                                pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                            elif hinge_location == 'top':
                                # Distance from center to top edge
                                offset_distance = bbox_h * 0.5 * pixel_to_meter
                                pivot_point = [current_position[0] - offset_distance, current_position[1], current_position[2]]
                            elif hinge_location == 'bottom':
                                # Distance from center to bottom edge
                                offset_distance = bbox_h * 0.5 * pixel_to_meter
                                pivot_point = [current_position[0] + offset_distance, current_position[1], current_position[2]]
                            
                            self.logger.info(f"Using object bounding box for pivot calculation: bbox={bbox}, offset={offset_distance:.3f}m")
                    else:
                        # Fallback to reasonable estimates if no bounding box available
                        self.logger.warning("No object bounding box available, using estimated distances")
                        if hinge_location == 'left':
                            offset_distance = 0.33  # 33cm fallback  
                            # For left hinge: pivot to the left (negative Y direction) 
                            pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                        elif hinge_location == 'right':
                            offset_distance = 0.33   # 33cm fallback
                            # For right hinge: pivot should be at the far right edge of object (negative Y direction)
                            pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                        elif hinge_location == 'top':
                            offset_distance = 0.3   # 30cm fallback
                            pivot_point = [current_position[0] - offset_distance, current_position[1], current_position[2]]
                        elif hinge_location == 'bottom':
                            offset_distance = 0.3   # 30cm fallback
                            pivot_point = [current_position[0] + offset_distance, current_position[1], current_position[2]]
                    
                    self.logger.info(f"Hinge pivot point calculated: gripper={current_position}, hinge={hinge_location}, pivot={pivot_point}")
            
            # Execute pull using existing motion planner logic
            distance = 0.15  # Standard pull distance
            
            if pivot_point is not None:
                # Use pivoted pull
                current_pose = self.motion_planner.get_tcp_pose_api()
                if current_pose is not None:
                    current_position, current_orientation = current_pose
                    pivot_point_array = np.array(pivot_point)
                    
                    # Calculate radius based on hinge location
                    if hinge_location and hinge_location in ['top', 'bottom', 'left', 'right']:
                        # For hinge location, the radius should be the distance from interaction point to hinge edge
                        # The current_position is the interaction point, pivot_point is calculated hinge location
                        radius = np.linalg.norm(np.array(current_position) - pivot_point_array)
                        self.logger.info(f"Using hinge-based radius calculation: hinge_location={hinge_location}, radius={radius:.3f}")
                    else:
                        # Legacy radius calculation (distance from current position to pivot)
                        radius = np.linalg.norm(np.array(current_position) - pivot_point_array)
                        self.logger.info(f"Using legacy pivot radius calculation: radius={radius:.3f}")
                    
                    success = self.motion_planner.execute_pivot_pull_direct_xarm(
                        pivot_point=pivot_point_array,
                        current_position=current_position,
                        current_orientation=current_orientation,
                        radius=radius,
                        arc_angle_degrees=70.0,  # Use conservative angle like in test
                        segments=5,
                        speed_factor=0.05,  # Use ultra conservative speed like in test
                        is_quat=False,  # Match test method setting
                        hinge_location=hinge_location,  # Pass hinge location for logging
                        is_push=False  # Pass the actual push/pull action type
                    )
                else:
                    self.logger.error("Could not get current pose for pivoted pull")
                    success = False
            else:
                # Use standard pull with surface normal if available
                success, _, _ = self.motion_planner.plan_push_pull(
                    distance=distance,
                    is_push=False,
                    custom_normal=surface_normal
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing pull action: {e}")
            return False

    def _execute_push_action(self, norm_x: float, norm_y: float, force_direction: str, 
                            is_button: bool, has_pivot: bool, hinge_location: str, surface_normal=None) -> bool:
        """
        Execute a push action using the existing push/pull logic from action execution
        """
        try:
            # Move to the target position first
            success = self._execute_grasp_movement(norm_x, norm_y, False, True, False)
            if not success:
                self.logger.error("Failed to move to push position")
                return False
            
            # Calculate pivot point if needed (same logic as pull)
            pivot_point = None
            if has_pivot and hinge_location and hinge_location in ['top', 'bottom', 'left', 'right']:
                self.logger.info(f"Using hinge location: {hinge_location}")
                
                # Get current gripper position
                current_pose = self.motion_planner.get_robot_tcp_pose()
                if current_pose is not None:
                    current_position, _ = current_pose
                    
                    # Flatten the position array if it's nested
                    if hasattr(current_position, 'flatten'):
                        current_position = current_position.flatten()
                    elif isinstance(current_position, (list, tuple)) and len(current_position) > 0:
                        if isinstance(current_position[0], (list, tuple, np.ndarray)):
                            current_position = current_position[0]
                    
                    # Use actual object bounding box if available (from stored skill data)
                    if hasattr(self, 'current_object_info') and self.current_object_info and self.current_object_info.bbox:
                        bbox = self.current_object_info.bbox  # [x, y, w, h] format in pixels
                        self.logger.info(f"Using stored object bounding box for pivot calculation: {bbox}")
                        
                        bbox_x, bbox_y, bbox_w, bbox_h = bbox
                        
                        # Convert pixel coordinates to world coordinates
                        pixel_to_meter = self.estimate_pixel_to_meter_ratio(current_position)
                        
                        # Calculate pivot point using 3D world coordinates instead of pixel conversion
                        if hasattr(self, 'current_interaction_point_world') and self.current_interaction_point_world:
                            # Use 3D world coordinates of interaction point for accurate pivot calculation
                            interaction_world_x, interaction_world_y, interaction_world_z = self.current_interaction_point_world
                            
                            # Calculate 3D offset from interaction point to hinge edge using bounding box dimensions
                            # Convert bounding box dimensions to world units
                            bbox_width_world = bbox_w * pixel_to_meter
                            bbox_height_world = bbox_h * pixel_to_meter
                            
                            # Get interaction point pixel coordinates for offset calculation
                            if hasattr(self, 'current_interaction_point_pixel') and self.current_interaction_point_pixel:
                                interaction_px, interaction_py = self.current_interaction_point_pixel
                                
                                # CAMERA FRAME: X+ = right, Y+ = down
                                # ROBOT FRAME: Y+ = right, Y- = left, X- = forward/top, X+ = back/bottom
                                
                                # Calculate world position of the specific bounding box edge
                                if hinge_location == 'left':
                                    # Left edge: offset from interaction point to left edge of bbox
                                    pixel_offset_x = bbox_x - interaction_px
                                    world_offset_y = pixel_offset_x * pixel_to_meter  # Camera X -> Robot Y
                                    pivot_point = [interaction_world_x, interaction_world_y - world_offset_y, interaction_world_z]  # Left = negative Y
                                elif hinge_location == 'right':
                                    # Right edge: offset from interaction point to right edge of bbox
                                    pixel_offset_x = (bbox_x + bbox_w) - interaction_px
                                    world_offset_y = pixel_offset_x * pixel_to_meter  # Camera X -> Robot Y
                                    pivot_point = [interaction_world_x, interaction_world_y - world_offset_y, interaction_world_z]  # Right = negative Y
                                elif hinge_location == 'top':
                                    # Top edge: offset from interaction point to top edge of bbox
                                    pixel_offset_y = bbox_y - interaction_py
                                    world_offset_x = pixel_offset_y * pixel_to_meter  # Camera Y -> Robot X (inverted)
                                    pivot_point = [interaction_world_x - world_offset_x, interaction_world_y, interaction_world_z]  # Top = negative X
                                elif hinge_location == 'bottom':
                                    # Bottom edge: offset from interaction point to bottom edge of bbox
                                    pixel_offset_y = (bbox_y + bbox_h) - interaction_py
                                    world_offset_x = pixel_offset_y * pixel_to_meter  # Camera Y -> Robot X (inverted)
                                    pivot_point = [interaction_world_x + world_offset_x, interaction_world_y, interaction_world_z]  # Bottom = positive X
                                    
                                self.logger.info(f"Calculated pivot using 3D world coordinates: interaction_point={self.current_interaction_point_world}, hinge={hinge_location}, pivot={pivot_point}, offset={world_offset_y if 'world_offset_y' in locals() else world_offset_x:.3f}m")
                            else:
                                self.logger.warning("No interaction point pixel coordinates available for offset calculation")
                                pivot_point = None
                        else:
                            # Fallback: use middle of the specified edge of bounding box
                            self.logger.info("No interaction point stored, using middle of bounding box edge")
                            if hinge_location == 'left':
                                # Distance from center to left edge
                                offset_distance = bbox_w * 0.5 * pixel_to_meter
                                pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                            elif hinge_location == 'right':
                                # Distance from center to right edge
                                offset_distance = bbox_w * 0.5 * pixel_to_meter
                                pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                            elif hinge_location == 'top':
                                # Distance from center to top edge
                                offset_distance = bbox_h * 0.5 * pixel_to_meter
                                pivot_point = [current_position[0] - offset_distance, current_position[1], current_position[2]]
                            elif hinge_location == 'bottom':
                                # Distance from center to bottom edge
                                offset_distance = bbox_h * 0.5 * pixel_to_meter
                                pivot_point = [current_position[0] + offset_distance, current_position[1], current_position[2]]
                            
                            self.logger.info(f"Using object bounding box for pivot calculation: bbox={bbox}, offset={offset_distance:.3f}m")
                    else:
                        # Fallback to reasonable estimates if no bounding box available
                        self.logger.warning("No object bounding box available, using estimated distances")
                        if hinge_location == 'left':
                            offset_distance = 0.33  # 33cm fallback  
                            # For left hinge: pivot to the left (negative Y direction) 
                            pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                        elif hinge_location == 'right':
                            offset_distance = 0.33   # 33cm fallback
                            # For right hinge: pivot should be at the far right edge of object (negative Y direction)
                            pivot_point = [current_position[0], current_position[1] - offset_distance, current_position[2]]
                        elif hinge_location == 'top':
                            offset_distance = 0.3   # 30cm fallback
                            pivot_point = [current_position[0] - offset_distance, current_position[1], current_position[2]]
                        elif hinge_location == 'bottom':
                            offset_distance = 0.3   # 30cm fallback
                            pivot_point = [current_position[0] + offset_distance, current_position[1], current_position[2]]
                    
                    self.logger.info(f"Hinge pivot point calculated: gripper={current_position}, hinge={hinge_location}, pivot={pivot_point}")
                else:
                    self.logger.error("Could not get current robot pose for pivot calculation")
            
            # Execute push using existing motion planner logic
            distance = 0.15  # Standard push distance
            
            if pivot_point is not None:
                # Use pivoted push (though pivot push is less common than pivot pull)
                current_pose = self.motion_planner.get_tcp_pose_api()
                if current_pose is not None:
                    current_position, current_orientation = current_pose
                    pivot_point_array = np.array(pivot_point)
                    
                    # For pivoted push, we use the same method but with is_push=True
                    success = self.motion_planner.execute_pivot_pull_direct_xarm(
                        pivot_point=pivot_point_array,
                        current_position=current_position,
                        current_orientation=current_orientation,
                        radius=np.linalg.norm(np.array(current_position[:2]) - pivot_point_array[:2]),
                        arc_angle_degrees=70.0,  # Use conservative angle like in test
                        segments=5,
                        speed_factor=0.05,  # Use ultra conservative speed like in test
                        is_quat=False,  # Match test method setting
                        hinge_location=hinge_location,  # Pass hinge location for logging
                        is_push=True  # Pass the actual push/pull action type
                    )
                else:
                    self.logger.error("Could not get current pose for pivoted push")
                    success = False
            else:
                # Use standard push with surface normal if available
                success, _, _ = self.motion_planner.plan_push_pull(
                    distance=distance,
                    is_push=True,
                    custom_normal=surface_normal
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing push action: {e}")
            return False

    def _execute_grasp_movement(self, norm_x: float, norm_y: float, is_top_down: bool, is_side_grasp: bool, is_placement: bool = None) -> bool:
        """
        Execute a grasp movement to the specified normalized coordinates
        
        Args:
            norm_x: Normalized X coordinate (0-1)
            norm_y: Normalized Y coordinate (0-1)
            is_top_down: Whether this is a top-down grasp
            is_side_grasp: Whether this is a side grasp
            is_placement: Whether this is a placement action (auto-detected if None)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Check if we have images available (either stored or live)
            if self.latest_color_image is None or self.latest_depth_image is None:
                self.logger.error("No images available for grasp movement (neither stored nor live)")
                return False
            
            # Convert normalized coordinates to pixel coordinates
            h, w = self.latest_color_image.shape[:2]
            pixel_x = int(norm_x * w)
            pixel_y = int(norm_y * h)
            
            # Use perception system to convert pixel coordinates to 3D position (camera frame)
            target_position_camera, confidence = self.perception_system._estimate_point_pose(
                (pixel_x, pixel_y), 
                self.latest_depth_image
            )
            
            if target_position_camera is None:
                self.logger.error("Failed to convert pixel coordinates to 3D position")
                return False
            
            self.logger.info(f"Target 3D position (camera frame): {target_position_camera}, confidence: {confidence}")
            
            # Convert from camera frame to robot base frame
            target_position_robot, target_orientation_robot = self.motion_planner.convert_cam_pose_to_base(
                target_position_camera,
                [0, 0, 0, 1],  # Identity quaternion for position-only transformation
                do_translation=True
            )
            
            self.logger.info(f"Target 3D position (robot frame): {target_position_robot}")
            
            # Use robot frame coordinates for motion planning
            target_position = target_position_robot
            
            # Use the explicitly passed placement flag (no auto-detection at this level)
            if is_placement is None:
                is_placement = False
            
            # Calculate approach pose based on grasp type
            # Use fixed orientations: top-down (180, 0, 0) or side grasp (160, -87, 20)
            approach_position = target_position.copy()

            if is_top_down:
                # Top-down grasp: approach from above with vertical orientation
                if is_placement:
                    self.logger.info("Top-down placement action")
                # Use None orientation with force_top_down=True to get (180, 0, 0) RPY
                approach_orientation = None

            elif is_side_grasp:
                # Side grasp: approach from the side
                if is_placement:
                    self.logger.info("Side grasp placement action")
                # Use 'side_grasp' string to get fixed (160, -87, 20) RPY
                approach_orientation = 'side_grasp'

            else:
                # Default to side grasp orientation
                if is_placement:
                    self.logger.info("Default placement action (using side grasp orientation)")
                approach_orientation = 'side_grasp'

            # Execute the movement using direct xArm interface (no CuRobo trajectory planning)
            # Frame transformation already done above via convert_cam_pose_to_base
            self.logger.info(f"Moving to position: {approach_position}, orientation: {approach_orientation}, is_top_down: {is_top_down}")
            speed = 200 if self.fast_mode else 100  # mm/s
            success = self.motion_planner.move_to_pose_direct(
                target_position=approach_position,
                target_orientation=approach_orientation,
                force_top_down=is_top_down,
                speed=speed,
                acc=1000,
                is_camera_frame=False,  # Already in robot frame
                is_place=is_placement,
                wait=True
            )

            if success:
                self.logger.info("Grasp movement executed successfully")
                return True
            else:
                self.logger.error("Failed to execute grasp movement")
                return False
            
        except Exception as e:
            self.logger.error(f"Error executing grasp movement: {e}")
            return False


def main():
    """Example usage of the direct skill executor with simplified ArUco interface"""
    try:
        print("=== Direct Skill Executor with Simplified ArUco Interface ===")
        
        # Choose camera type
        use_zed = 'n' #input("Use ZED camera? (y/n, default=y): ").lower()
        use_zed_camera = use_zed != 'n'
        
        camera_type = "ZED" if use_zed_camera else "RealSense"
        print(f"Using {camera_type} camera with simplified ArUco interface")
        
        # Initialize the direct skill executor with selected camera
        executor = DirectSkillExecutor(
            robot_ip="192.168.1.224",  # Replace with your robot's IP
            show_debug_windows=True,
            calibrate_transform=use_zed_camera,  # Only calibrate for ZED camera
            use_zed_camera=use_zed_camera,
            fast_mode=True  # Enable fast mode for quicker startup
        )
        
        # Display camera info
        camera_info = executor.get_camera_info()
        print(f"Camera configuration: {camera_info}")
        
        # Check transform status
        transform_status = executor.get_transform_status()
        print(f"Transform status: {transform_status}")
        
        # Check robot status
        status = executor.get_robot_status()
        if status:
            print(f"Robot status: Mode={status['mode']}, State={status['state']}")
        
        # Move to home position
        print("Moving to home position...")
        # executor.move_to_home()
        
        # Execute a skill using the selected camera
        print(f"Executing skill with {camera_type} camera...")
        success, message = executor.execute_skill("pickup", "bottle")
        
        if success:
            print(f"Skill execution successful: {message}")
        else:
            print(f"Skill execution failed: {message}")
            
        # Option to recalibrate transform if using ZED camera
        if use_zed_camera:
            user_input = input("Recalibrate transform? (y/n): ").lower()
            if user_input == 'y':
                print("Recalibrating ZED-to-robot transform...")
                if executor.recalibrate_transform():
                    print("Recalibration successful!")
                    transform_status = executor.get_transform_status()
                    print(f"New transform status: {transform_status}")
                else:
                    print("Recalibration failed!")
        else:
            print("Transform calibration not available with RealSense camera")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        if 'executor' in locals():
            executor.shutdown()


if __name__ == '__main__':
    main()