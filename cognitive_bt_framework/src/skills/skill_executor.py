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

# Import the CuRobo motion planner
from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface import CuRoboMotionPlanner 

# Import the updated stereo ArUco detector
from cognitive_bt_framework.src.vision.aruco_tf_v3 import StereoArucoDetector

DETECTION_RETRIES = 1

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
                 calibrate_transform=True, use_zed_camera=True, fast_mode=False):
        """
        Initialize the direct skill executor with configurable camera selection
        
        Args:
            robot_ip: IP address of the xArm robot
            camera_params: Camera configuration parameters
            show_debug_windows: Whether to show debug visualization windows
            calibrate_transform: Whether to perform camera-to-robot transform calibration on startup
            use_zed_camera: If True, use ZED camera as primary. If False, use RealSense camera as primary
            fast_mode: If True, skip extensive verification and use minimal debugging
        """
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
        self.action_timeout = 240
        
        # Transform calibration (only relevant when using ZED camera)
        self.zed_to_robot_transform = None
        self.transform_confidence = 0.0
        self.stereo_detector = None
        
        # Initialize valid skills
        self.valid_skills = {
            'open': 1, 'close': 1, 'pickup': 1,
            'place': 2, 'switchon': 1, 'switchoff': 1,
            'twist': 1  # Add twist to valid skills
        }
        
        # Initialize image handling
        self.latest_color_image = None
        self.latest_depth_image = None
        self.execution_start_time = None
        self.perception_system = None
        
        # Camera references (will be set based on use_zed_camera flag)
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
            self.stereo_detector = StereoArucoDetector(
                robot_camera=self.secondary_camera,  # RealSense camera
                zed_camera=self.main_camera,         # ZED camera
                T_robot_base_to_camera_link=T_robot_base_to_camera_link,
                marker_size=0.05  # 5cm markers
            )
            
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
            llm_interface = LLMInterfaceOpenAI()
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
    
    def adjust_tcp_pose_for_depth_centering(self, target_position, gripper_width_pixels=40):
        """
        Adjust TCP pose position using depth image to center the selected depth region within the gripper.
        
        Args:
            target_position: Original target position [x, y, z] in camera/world coordinates
            gripper_width_pixels: Approximate gripper width in pixels (default 40 pixels)
            
        Returns:
            adjusted_position: Position adjusted to center depth region within gripper
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
            
            # Define search region around target pixel (gripper-sized region)
            half_width = gripper_width_pixels // 2
            x_min = max(0, pixel_x - half_width)
            x_max = min(w, pixel_x + half_width)
            y_min = max(0, pixel_y - half_width) 
            y_max = min(h, pixel_y + half_width)
            
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
                
            # Find the centroid of valid depth points
            valid_depths = depth_region[valid_mask]
            y_coords, x_coords = np.where(valid_mask)
            
            self.logger.info(f"DEBUG - Valid depth points: {len(valid_depths)}")
            self.logger.info(f"DEBUG - Valid depth range: {valid_depths.min():.1f} to {valid_depths.max():.1f} mm")
            
            # Add offset to get absolute pixel coordinates
            x_coords = x_coords + x_min
            y_coords = y_coords + y_min
            
            # Calculate weighted centroid (closer objects get higher weight)
            weights = 1.0 / (valid_depths + 1)  # Inverse depth weighting
            centroid_x = np.average(x_coords, weights=weights)
            centroid_y = np.average(y_coords, weights=weights)
            centroid_depth = np.average(valid_depths, weights=weights)
            
            self.logger.info(f"DEBUG - Centroid pixel: ({centroid_x:.1f}, {centroid_y:.1f})")
            self.logger.info(f"DEBUG - Centroid depth: {centroid_depth:.1f} mm")
            
            # Convert depth using depth scale (RealSense depth_scale converts raw to meters)
            centroid_depth_m = centroid_depth * depth_scale
            
            self.logger.info(f"DEBUG - Centroid depth in meters: {centroid_depth_m:.4f} m")
            
            # Convert centroid pixel back to 3D camera coordinates
            centroid_x_3d = (centroid_x - ppx) * centroid_depth_m / fx
            centroid_y_3d = (centroid_y - ppy) * centroid_depth_m / fy
            
            self.logger.info(f"DEBUG - Centroid 3D coordinates: ({centroid_x_3d:.4f}, {centroid_y_3d:.4f}, {centroid_depth_m:.4f})")
            
            # Only adjust X coordinate to center the gripper, maintain original depth for contact
            # This keeps the grasp point on the same surface (e.g., handle) while centering
            original_pos = np.array(target_position[:3])
            
            # Calculate X adjustment only (for gripper centering)
            x_adjustment = centroid_x_3d - original_pos[0]
            
            # Limit X adjustment to reasonable values
            max_x_adjustment = 0.05  # Maximum 5cm adjustment in X direction
            x_adjustment = np.clip(x_adjustment, -max_x_adjustment, max_x_adjustment)
            
            # Create adjusted position: adjusted X, original Y, original Z (maintain contact depth)
            adjusted_position = [
                original_pos[0] + x_adjustment,  # Adjusted X for centering
                original_pos[1],                 # Keep original Y
                original_pos[2]                  # Keep original Z (maintain contact depth)
            ]
            
            self.logger.info(f"DEBUG - X adjustment for centering: {x_adjustment:.4f} m")
            self.logger.info(f"DEBUG - Y coordinate unchanged: {original_pos[1]:.4f} m")
            self.logger.info(f"DEBUG - Z coordinate unchanged (maintaining contact): {original_pos[2]:.4f} m")
            self.logger.info(f"DEBUG - Centroid depth was: {centroid_depth_m:.4f} m (not used for Z)")
            
            # Calculate final adjustment offset for logging
            final_offset = np.array(adjusted_position) - np.array(target_position[:3])
            final_offset_magnitude = np.linalg.norm(final_offset)
            
            self.logger.info(f"Depth-based centering adjustment: offset={final_offset}, magnitude={final_offset_magnitude:.4f}m")
            self.logger.info(f"Original position: {target_position[:3]}")
            self.logger.info(f"Adjusted position: {adjusted_position}")
            
            return adjusted_position
            
        except Exception as e:
            self.logger.error(f"Error in depth centering adjustment: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return target_position

    def execute_action(self, action: ExecutableAction, timeout=120, initial_cam_tf=None) -> bool:
        """Execute a skill action using the CuRobo motion planner with camera poses"""
        try:
            start_time = time.time()
            self.logger.info(f"Executing action: {vars(action)} with timeout {timeout}s")
            
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
                force = action.parameters.get('force', 10.0)
                return self.motion_planner.close_gripper(wait=True, timeout=min(timeout, 10.0))
                    
            elif action.action_type == 'open_gripper':
                return self.motion_planner.open_gripper(wait=True, timeout=min(timeout, 10.0))
                    
            elif action.action_type == 'retract_gripper':
                # Move gripper back by specified distance
                distance = 0.1
                speed = action.parameters.get('speed', 0.3)
                
                # Get current pose
                pose = self.motion_planner.get_robot_tcp_pose()
                success = self.motion_planner.retract_gripper()
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
                angular_velocity = action.parameters.get('angular_velocity', 0.5)
                rotation_angle = action.parameters.get('rotation_angle', 2 * np.pi)  # Default 90 degrees
                
                self.logger.info(f"Executing twist {direction} with angle {rotation_angle:.2f} rad")
                
                # Calculate remaining time for twist
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0.5:
                    self.logger.error("Insufficient time remaining for twist")
                    return False
                
                # Execute the twist motion
                success = self.motion_planner.execute_wrist_twist(
                    direction=direction,
                    rotation_angle=rotation_angle,
                    speed_factor=angular_velocity,
                    timeout=min(remaining_time, 30.0)
                )
                
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
                # Position handling depends on camera type
                target_position = action.position.tolist() if isinstance(action.position, np.ndarray) else action.position
                target_orientation = action.orientation
                
                # Apply depth-based centering adjustment before sending to robot interface
                self.logger.info("Applying depth-based centering adjustment to TCP pose")
                adjusted_position = self.adjust_tcp_pose_for_depth_centering(target_position)
                
                # Calculate remaining time for move
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0.5:
                    self.logger.error("Insufficient time remaining for move_gripper_to_pose")
                    return False
                                
                # Use the motion planner to move to pose with adjusted position
                success, _, _ = self.motion_planner.move_to_pose_with_preparation(
                    target_position=adjusted_position,
                    target_orientation=target_orientation,
                    execute=True,
                    planning_timeout=min(remaining_time * 0.8, 10.0),
                    speed_factor=1.0,
                    is_camera_frame=True
                )
                
                return success
                
            elif action.action_type in ['push', 'pull']:
                # Handle push/pull actions
                distance = action.parameters.get('distance', 0.1)
                force_magnitude = action.parameters.get('force_magnitude', 10.0)
                
                if action.parameters.get('is_button', False):
                    distance = 0.01
                    self.motion_planner.close_gripper()
                    
                speed = action.parameters.get('speed', 0.3)
                is_parallel = action.parameters.get('is_parallel', False)
                surface_normal = action.parameters.get('surface_normal', None)
                pivot_position = action.parameters.get('pivot_position', None)
                
                # Surface normal conversion only needed for ZED camera
                if surface_normal is not None and self.use_zed_camera and self.zed_to_robot_transform is not None:
                    # Convert surface normal from ZED frame to robot base frame
                    surface_normal_homogeneous = np.array([surface_normal[0], surface_normal[1], surface_normal[2], 0])
                    surface_normal_robot = self.zed_to_robot_transform @ surface_normal_homogeneous
                    surface_normal = surface_normal_robot[:3]
                    print(f"Surface normal converted from ZED to robot frame: {surface_normal}")
                
                # Calculate pivot point based on x offset and current gripper position
                pivot_point = None
                if pivot_position is not None:
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
                        
                        # Calculate radius (distance from current position to pivot)
                        radius = np.linalg.norm(np.array(current_position) - pivot_point_array)
                        
                        self.logger.info(f"Executing pivot pull: current_pos={current_position}, pivot={pivot_point_array}, radius={radius:.3f}")
                        
                        # Use conservative settings like in the test method
                        success = self.motion_planner.execute_pivot_pull_direct_xarm(
                            pivot_point=pivot_point_array,
                            current_position=current_position,
                            current_orientation=current_orientation,
                            radius=radius,
                            arc_angle_degrees=30.0,  # Use conservative angle like in test
                            segments=10,
                            speed_factor=0.05,  # Use ultra conservative speed like in test
                            is_quat=False  # Match test method setting
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
            
            # Validate skill command
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
                    object_info
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
            pcd = self.main_camera.get_point_cloud()
            if pcd is not None:
                self.logger.info(f"Retrieved point cloud from {camera_type} camera")
                # Note: Point cloud coordinates depend on camera type
                # self.motion_planner.update_dynamic_collision_objects(pcd)
            
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
        executor.move_to_home()
        
        # Execute a skill using the selected camera
        print(f"Executing skill with {camera_type} camera...")
        success, message = executor.execute_skill("open", "cabinet")
        
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