#!/usr/bin/env python3
"""
Streamlined Skill Executor - Optimized for Task Planner Integration

This is a streamlined version of DirectSkillExecutor that includes only the methods
required by TaskPlanner, with all deprecated and unused methods removed.

Required Methods (16 total):
1. execute_skill_sequence() - Main execution method
2. set_data_recorder() - Data recording integration
3. get_execution_data() - Returns execution data
4. validate_skill_command() - Validates skill commands
5. execute_action() - Executes individual actions
6. adjust_tcp_pose_for_depth_centering() - Depth-based centering
7. estimate_pixel_to_meter_ratio() - Pixel-to-meter conversion
8. get_latest_images() - Gets camera frames
9. shutdown() - Cleanup method
10. Attributes: camera, motion_planner, perception_system, skill_handler
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import time
import sys
import logging
import torch
from scipy.spatial.transform import Rotation

# Import the Camera classes
from cognitive_bt_framework.src.vision.realsense import Camera as RealSenseCamera
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig, ObjectInfo
from cognitive_bt_framework.src.skills import SkillGenerator, SkillHandler, ExecutableAction
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI

# Import the CuRobo motion planner
from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface_streamlined import CuRoboMotionPlanner

DETECTION_RETRIES = 10


class DirectSkillExecutor:
    """
    Streamlined skill executor optimized for TaskPlanner integration.
    Uses RealSense camera only, no ZED camera support.
    """

    def __init__(self, robot_ip="192.168.1.224", camera_params=None,
                 show_debug_windows=False, calibrate_transform=False,
                 use_zed_camera=False, fast_mode=True):
        """
        Initialize the streamlined skill executor

        Args:
            robot_ip: IP address of the xArm robot
            camera_params: Camera configuration parameters
            show_debug_windows: Whether to show debug windows (deprecated, ignored)
            calibrate_transform: Transform calibration (deprecated, ignored)
            use_zed_camera: ZED camera flag (deprecated, ignored - always uses RealSense)
            fast_mode: Fast mode flag (always True in streamlined version)
        """
        print('Starting streamlined skill execution system initialization')

        # Initialize logging
        self.setup_logging()

        # Set configuration parameters
        self.camera_params = camera_params or {
            'width': 640,
            'height': 480,
            'fps': 30
        }

        # Force streamlined settings
        self.use_zed_camera = False  # Always use RealSense
        self.show_debug_windows = False  # No debug windows
        self.calibrate_transform = False  # No transform calibration needed
        self.fast_mode = True  # Always fast mode
        self.action_timeout = 240

        # Initialize valid skills
        self.valid_skills = {
            'open': 1, 'close': 1, 'pickup': 1,
            'place': 2, 'switchon': 1, 'switchoff': 1,
            'twist': 1, 'detect_object': 1
        }

        # Initialize image handling
        self.latest_color_image = None
        self.latest_depth_image = None
        self.execution_start_time = None
        self.perception_system = None
        self.current_object_info = None
        self.current_interaction_point_pixel = None
        self.current_interaction_point_world = None

        # Data recording integration
        self.data_recorder = None

        # Initialize CuRobo motion planner
        print("Initializing CuRobo motion planner...")
        self.motion_planner = CuRoboMotionPlanner(robot_ip=robot_ip)
        print("CuRobo motion planner initialized")

        # Setup camera (RealSense only)
        self.setup_camera()

        # Initialize perception system
        self.initialize_perception_system()
        print('Perception system initialized with RealSense camera')

        print('Streamlined skill executor initialization completed')
        self.motion_planner.open_gripper()

    def setup_logging(self):
        """Configure logging to console"""
        self.logger = logging.getLogger('streamlined_skill_executor')
        self.logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)

    def setup_camera(self):
        """Setup RealSense camera (main camera only)"""
        print("Initializing RealSense camera...")
        self.camera = RealSenseCamera(
            width=self.camera_params['width'],
            height=self.camera_params['height'],
            fps=self.camera_params['fps'],
            depth_averaging_frames=3,
            debug=False
        )

        # Start the RealSense camera
        if not self.camera.start():
            raise RuntimeError("Failed to start RealSense camera")
        print("RealSense camera started successfully")

    def initialize_perception_system(self):
        """Initialize perception system with RealSense camera"""
        try:
            # Get camera matrix from RealSense
            camera_matrix = np.array([
                [self.camera.intrinsics.fx, 0, self.camera.intrinsics.ppx],
                [0, self.camera.intrinsics.fy, self.camera.intrinsics.ppy],
                [0, 0, 1]
            ])
            print("Using camera matrix from RealSense camera")

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

            # Get depth scale from RealSense
            depth_scale = getattr(self.camera, 'depth_scale', 1000.0)

            # Initialize PerceptionSystem
            self.perception_system = PerceptionSystem(
                fast_sam_config=fs_config,
                camera_matrix=camera_matrix,
                depth_scale=depth_scale,
                debug=True
            )

            # Initialize LLM and skill handler
            llm_interface = LLMInterfaceOpenAI()
            skill_generator = SkillGenerator(llm_interface=llm_interface)
            self.skill_handler = SkillHandler(skill_generator, self.perception_system)

            self.logger.info('Perception system initialized with RealSense camera')

        except Exception as e:
            self.logger.error(f'Failed to initialize perception system: {str(e)}')
            raise

    def get_latest_images(self, timeout: float = 2.0):
        """Get latest images from RealSense camera"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            color_image, depth_image = self.camera.get_frames()

            if color_image is not None and depth_image is not None:
                self.latest_color_image = color_image
                self.latest_depth_image = depth_image
                return color_image, depth_image

            time.sleep(0.05)

        self.logger.warning('Timeout waiting for RealSense camera images')
        return None, None

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
        Adjust TCP pose position using depth clustering to center on main object cluster

        Args:
            target_position: Original target position [x, y, z]
            search_radius_m: Physical search radius in meters (default 0.05 = 5cm)
            depth_threshold_ratio: Ratio of average depth for clustering (default 0.1)
            cluster_min_size: Minimum cluster size (default 10 pixels)

        Returns:
            adjusted_position: Position adjusted to center on peak depth region
        """
        try:
            if self.latest_depth_image is None or self.latest_color_image is None:
                self.logger.warning("No depth/color image available for centering adjustment")
                return target_position

            # Get camera intrinsics
            fx = self.camera.intrinsics.fx
            fy = self.camera.intrinsics.fy
            ppx = self.camera.intrinsics.ppx
            ppy = self.camera.intrinsics.ppy

            # Get depth scale
            depth_scale = getattr(self.camera, 'depth_scale', 1000.0)

            # Project target position to pixel coordinates
            u = int((target_position[0] * fx / target_position[2]) + ppx)
            v = int((target_position[1] * fy / target_position[2]) + ppy)

            # Calculate search radius in pixels
            pixel_per_meter = fx / target_position[2]
            search_radius_px = int(search_radius_m * pixel_per_meter)

            # Define circular search region
            y_indices, x_indices = np.ogrid[:self.latest_depth_image.shape[0], :self.latest_depth_image.shape[1]]
            mask = ((x_indices - u)**2 + (y_indices - v)**2) <= search_radius_px**2

            # Extract depths in search region
            search_depths = self.latest_depth_image[mask]
            valid_depths = search_depths[search_depths > 0]

            if len(valid_depths) < cluster_min_size:
                self.logger.warning(f"Insufficient valid depth points ({len(valid_depths)}) in search region")
                return target_position

            # Find peak depth cluster
            avg_depth = np.mean(valid_depths)
            depth_threshold = avg_depth * depth_threshold_ratio

            # Create depth-based mask
            depth_mask = np.abs(self.latest_depth_image - avg_depth) < depth_threshold
            combined_mask = mask & depth_mask

            # Find centroid of peak depth cluster
            y_coords, x_coords = np.where(combined_mask)

            if len(x_coords) < cluster_min_size:
                self.logger.warning("Peak depth cluster too small")
                return target_position

            # Calculate centroid
            centroid_u = int(np.mean(x_coords))
            centroid_v = int(np.mean(y_coords))
            centroid_depth = self.latest_depth_image[centroid_v, centroid_u]

            if centroid_depth <= 0:
                return target_position

            # Convert back to 3D coordinates
            centroid_depth_m = centroid_depth / depth_scale
            adjusted_x = (centroid_u - ppx) * centroid_depth_m / fx
            adjusted_y = (centroid_v - ppy) * centroid_depth_m / fy
            adjusted_z = centroid_depth_m

            adjusted_position = [adjusted_x, adjusted_y, adjusted_z]

            offset = np.linalg.norm(np.array(adjusted_position) - np.array(target_position))
            self.logger.info(f"TCP depth centering: offset={offset*1000:.1f}mm, cluster_size={len(x_coords)}")

            return adjusted_position

        except Exception as e:
            self.logger.error(f"Depth centering adjustment failed: {e}")
            return target_position

    def estimate_pixel_to_meter_ratio(self, world_position):
        """
        Estimate pixel-to-meter ratio at given world position

        Args:
            world_position: World position [x, y, z]

        Returns:
            float: Approximate meters per pixel at that depth
        """
        try:
            if self.camera is None:
                return 0.001  # Default fallback

            # Get camera focal length
            fx = self.camera.intrinsics.fx

            # Calculate pixel-to-meter ratio based on depth
            z_depth = world_position[2] if len(world_position) > 2 else 0.5

            if z_depth <= 0:
                z_depth = 0.5  # Fallback depth

            # Ratio: meters per pixel at this depth
            pixel_to_meter = z_depth / fx

            return pixel_to_meter

        except Exception as e:
            self.logger.warning(f"Failed to estimate pixel-to-meter ratio: {e}")
            return 0.001  # Default fallback

    def execute_action(self, action: ExecutableAction, timeout=120, initial_cam_tf=None, is_place=False) -> bool:
        """Execute a skill action using the CuRobo motion planner"""
        try:
            start_time = time.time()

            def is_timeout_approaching():
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining < 0.5:
                    self.logger.warning(f"Action timeout approaching: {elapsed:.2f}s elapsed, {remaining:.2f}s remaining")
                    return True
                return False

            # Wait for robot to be ready
            print(f"################## ARM STATE: {self.motion_planner.arm.get_state()}")
            while self.motion_planner.arm.get_state()[1] in [1, 4, 5]:
                print(f"################## ARM STATE: {self.motion_planner.arm.get_state()}")
                if self.motion_planner.arm.get_state()[1] == 5:
                    self.motion_planner.arm.set_state(0)
                time.sleep(0.5)

            # Handle gripper actions
            if action.action_type == 'close_gripper':
                if self.data_recorder:
                    self.data_recorder.start_robot_motion(f"close_gripper")
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

            elif action.action_type == 'retract_gripper':
                pose = self.motion_planner.get_robot_tcp_pose()
                if self.data_recorder:
                    self.data_recorder.start_robot_motion(f"retract_gripper")
                speed_factor = 2.0 if self.fast_mode else 1.5
                success = self.motion_planner.retract_gripper(speed_factor=speed_factor)
                if self.data_recorder:
                    self.data_recorder.end_robot_motion(f"retract_gripper")
                return success

            # Handle twist action
            elif action.action_type == 'twist':
                direction = action.parameters.get('direction', 'clockwise')
                angular_velocity = 2
                rotation_angle = np.pi

                self.logger.info(f"Executing twist {direction} with angle {rotation_angle:.2f} rad")

                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0.5:
                    self.logger.error("Insufficient time remaining for twist")
                    return False

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
                action.orientation = (0, 0.7071, 0, 0.7071)
                self.logger.info("Using side orientation for grasp")

            if is_timeout_approaching():
                self.logger.error("Timeout approaching after action setup")
                return False

            # Execute move_gripper_to_pose action
            if action.action_type == 'move_gripper_to_pose':
                # Store interaction point coordinates
                if action.pixel_position is not None:
                    self.current_interaction_point_pixel = action.pixel_position
                    self.logger.info(f"Stored interaction point pixel coordinates: {self.current_interaction_point_pixel}")

                if action.position is not None:
                    self.current_interaction_point_world = action.position.tolist() if isinstance(action.position, np.ndarray) else action.position
                    self.logger.info(f"Stored interaction point world coordinates: {self.current_interaction_point_world}")

                target_position = action.position.tolist() if isinstance(action.position, np.ndarray) else action.position
                target_orientation = action.orientation

                # Apply depth-based centering adjustment
                if not action.is_top_down_grasp:
                    search_radius = 0.05 if action.is_side_grasp else 0.03
                    self.logger.info(f"Applying depth-based centering adjustment to TCP pose (radius: {search_radius}m)")
                    adjusted_position = self.adjust_tcp_pose_for_depth_centering(target_position, search_radius_m=search_radius)
                else:
                    search_radius = 0.05
                    self.logger.info(f"Top-down grasp will be adjusted in robot interface with search radius: {search_radius}m")
                    adjusted_position = target_position

                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0.5:
                    self.logger.error("Insufficient time remaining for move_gripper_to_pose")
                    return False

                # Check if top-down grasp
                is_top_down_param = action.parameters.get('is_top_down_grasp', False)
                is_top_down_orientation = target_orientation is not None and abs(target_orientation[1]) > 0.9
                is_top_down = is_top_down_param or is_top_down_orientation

                self.logger.info(f"Surface adjustment check: is_top_down_param={is_top_down_param}, "
                               f"is_top_down_orientation={is_top_down_orientation}, "
                               f"final_is_top_down={is_top_down}")

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
                if action.object_info is not None:
                    if hasattr(action.object_info, 'mask'):
                        object_mask = action.object_info.mask
                        mask_shape = object_mask.shape if object_mask is not None else None
                        self.logger.info(f"Using object mask for surface-based TCP adjustment, shape: {mask_shape}")

                # Start motion tracking
                if self.data_recorder:
                    object_name = 'unknown_object'
                    if hasattr(action, 'object_info') and action.object_info and hasattr(action.object_info, 'name'):
                        object_name = action.object_info.name
                    self.data_recorder.start_robot_motion(f"{action.action_type}_{object_name}")

                success, _, _ = self.motion_planner.move_to_pose_with_preparation(
                    target_position=adjusted_position,
                    target_orientation=target_orientation,
                    execute=True,
                    planning_timeout=min(remaining_time * 0.8, 10.0),
                    speed_factor=1.0,
                    is_camera_frame=True,
                    is_place=is_place,
                    depth_image=self.latest_depth_image,
                    object_mask=object_mask,
                    adjust_tcp_for_surface=is_top_down,
                    tcp_standoff_m=0.00,
                    search_radius_m=search_radius if is_top_down else 0.05
                )

                # End motion tracking
                if self.data_recorder:
                    object_name = 'unknown_object'
                    if hasattr(action, 'object_info') and action.object_info and hasattr(action.object_info, 'name'):
                        object_name = action.object_info.name
                    self.data_recorder.end_robot_motion(f"{action.action_type}_{object_name}")

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
                hinge_location = action.parameters.get('hinge_location', None)

                # Calculate pivot point based on hinge_location or pivot_position
                pivot_point = None

                if hinge_location and hinge_location in ['top', 'bottom', 'left', 'right']:
                    self.logger.info(f"Using hinge location: {hinge_location}")
                    current_pose = self.motion_planner.get_robot_tcp_pose()
                    if current_pose is not None:
                        current_position, _ = current_pose

                        # Flatten position if nested
                        if hasattr(current_position, 'flatten'):
                            current_position = current_position.flatten()
                        elif isinstance(current_position, (list, tuple)) and len(current_position) > 0:
                            if isinstance(current_position[0], (list, tuple, np.ndarray)):
                                current_position = current_position[0]

                        # Calculate pivot using object bounding box
                        if hasattr(self, 'current_object_info') and self.current_object_info:
                            bbox = None
                            if self.current_object_info.bbox:
                                bbox = self.current_object_info.bbox
                            elif hasattr(self.current_object_info, 'mask') and self.current_object_info.mask is not None:
                                mask = self.current_object_info.mask
                                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    x, y, w, h = cv2.boundingRect(largest_contour)
                                    bbox = [x, y, w, h]
                                    self.logger.info(f"Calculated bounding box from mask: {bbox}")

                            if bbox:
                                bbox_x, bbox_y, bbox_w, bbox_h = bbox
                                pixel_to_meter = self.estimate_pixel_to_meter_ratio(current_position)

                                # Calculate pivot using 3D world coordinates
                                if hasattr(self, 'current_interaction_point_world') and self.current_interaction_point_world:
                                    interaction_world_x, interaction_world_y, interaction_world_z = self.current_interaction_point_world

                                    bbox_width_world = bbox_w * pixel_to_meter
                                    bbox_height_world = bbox_h * pixel_to_meter

                                    if hasattr(self, 'current_interaction_point_pixel') and self.current_interaction_point_pixel:
                                        interaction_px, interaction_py = self.current_interaction_point_pixel

                                        if hinge_location == 'left':
                                            pixel_offset_x = bbox_x - interaction_px
                                            world_offset_y = pixel_offset_x * pixel_to_meter
                                            pivot_point = [interaction_world_x, interaction_world_y + world_offset_y, interaction_world_z]
                                        elif hinge_location == 'right':
                                            pixel_offset_x = (bbox_x + bbox_w) - interaction_px
                                            world_offset_y = pixel_offset_x * pixel_to_meter
                                            pivot_point = [interaction_world_x, interaction_world_y + world_offset_y, interaction_world_z]
                                        elif hinge_location == 'top':
                                            pixel_offset_y = bbox_y - interaction_py
                                            world_offset_x = pixel_offset_y * pixel_to_meter
                                            pivot_point = [interaction_world_x - world_offset_x, interaction_world_y, interaction_world_z]
                                        elif hinge_location == 'bottom':
                                            pixel_offset_y = (bbox_y + bbox_h) - interaction_py
                                            world_offset_x = pixel_offset_y * pixel_to_meter
                                            pivot_point = [interaction_world_x - world_offset_x, interaction_world_y, interaction_world_z]

                                        self.logger.info(f"Calculated pivot using 3D world coordinates: hinge={hinge_location}, pivot={pivot_point}")

                elif pivot_position is not None:
                    self.logger.info("Using legacy pivot_position parameter")
                    current_pose = self.motion_planner.get_robot_tcp_pose()
                    if current_pose is not None:
                        current_position, _ = current_pose

                        if hasattr(current_position, 'flatten'):
                            current_position = current_position.flatten()
                        elif isinstance(current_position, (list, tuple)) and len(current_position) > 0:
                            if isinstance(current_position[0], (list, tuple, np.ndarray)):
                                current_position = current_position[0]

                        if isinstance(pivot_position, (list, tuple, np.ndarray)):
                            x_offset = pivot_position[0]
                        else:
                            x_offset = pivot_position

                        pivot_point = [
                            current_position[0],
                            current_position[1] + x_offset,
                            current_position[2]
                        ]
                        self.logger.info(f"Pivot point calculated: pivot={pivot_point}")

                if is_timeout_approaching():
                    self.logger.error("Timeout approaching, cannot execute push/pull action")
                    return False

                remaining_time = timeout - (time.time() - start_time)
                if remaining_time < 0.5:
                    self.logger.error("Insufficient time remaining for push/pull")
                    return False

                is_push = (action.action_type == 'push')

                # Use direct pivoted pull if pivot point provided and it's a pull action
                if not is_push and pivot_point is not None:
                    self.logger.info(f"Using direct pivoted pull with pivot point: {pivot_point}")

                    current_pose = self.motion_planner.get_tcp_pose_api()
                    if current_pose is not None:
                        current_position, current_orientation = current_pose

                        if hasattr(current_position, 'flatten'):
                            current_position = current_position.flatten()

                        pivot_point_array = np.array(pivot_point)

                        if hinge_location and hinge_location in ['top', 'bottom', 'left', 'right']:
                            radius = np.linalg.norm(np.array(current_position) - pivot_point_array)
                            self.logger.info(f"Using hinge-based radius: hinge_location={hinge_location}, radius={radius:.3f}")
                        else:
                            radius = np.linalg.norm(np.array(current_position) - pivot_point_array)
                            self.logger.info(f"Using legacy pivot radius: radius={radius:.3f}")

                        self.logger.info(f"Executing pivot pull: current_pos={current_position}, pivot={pivot_point_array}, radius={radius:.3f}")

                        success = self.motion_planner.execute_pivot_pull_direct_xarm(
                            pivot_point=pivot_point_array,
                            current_position=current_position,
                            current_orientation=current_orientation,
                            radius=radius,
                            arc_angle_degrees=70.0,
                            segments=5,
                            speed_factor=0.05,
                            is_quat=False,
                            hinge_location=hinge_location,
                            is_push=is_push
                        )
                    else:
                        self.logger.error("Could not get current robot pose for pivot pull")
                        success = False

                    if not success:
                        self.logger.warning("Direct pivoted pull failed, falling back to standard plan_push_pull")
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
                    # Standard push/pull
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
            self.logger.info(f"Starting skill sequence execution with RealSense camera: {len(skill_sequence)} skills")

            # Dictionary to track detected objects
            detected_objects = {}

            # List to track executed skills for context
            executed_skills = []

            # Reset skill generation file tracking
            if hasattr(self, 'skill_handler') and self.skill_handler:
                if hasattr(self.skill_handler, 'skill_generator') and self.skill_handler.skill_generator:
                    self.skill_handler.skill_generator.reset_generation_tracking()
                    self.logger.info("Reset skill generation file tracking for new task sequence")

            # Execute skills in sequence
            for i, (skill_name, parameters) in enumerate(skill_sequence):
                self.logger.info(f"Executing skill {i+1}/{len(skill_sequence)}: {skill_name} {parameters}")
                time.sleep(1)

                # Handle detect_object skill
                if skill_name.lower() == 'detect_object':
                    target_object = parameters.strip()
                    if not target_object:
                        error_msg = f"detect_object requires an object name parameter for skill {i+1}"
                        self.logger.error(error_msg)
                        return False, error_msg

                    # Get camera images for object detection
                    color_image, depth_image = self.camera.get_frames()
                    if color_image is None or depth_image is None:
                        error_msg = f"Failed to get RealSense camera images for detect_object in skill {i+1}"
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

                    executed_skills.append({
                        'skill_command': f"detect_object {target_object}",
                        'skill_name': 'detect_object',
                        'parameters': target_object,
                        'target_object': target_object
                    })
                    continue

                # Handle manipulation skills
                is_valid, error_msg, params = self.validate_skill_command(skill_name.lower(), parameters)
                if not is_valid:
                    error_msg = f"Validation failed for skill {i+1} ({skill_name}): {error_msg}"
                    self.logger.error(error_msg)
                    return False, error_msg

                # Handle place skill specially
                if skill_name.lower() == 'place':
                    source_object = params[0]
                    target_location = params[1]
                    target_location = target_location.strip()

                    if target_location not in detected_objects:
                        error_msg = f"Target location '{target_location}' has not been detected for place skill {i+1}. Use detect_object {target_location} first."
                        self.logger.error(error_msg)
                        return False, error_msg

                    target_object = target_location
                    object_info = detected_objects[target_location]
                    self.logger.info(f"Using detected object info for target location: {target_location}")
                else:
                    target_object = params[0]
                    if target_object not in detected_objects:
                        error_msg = f"Target object '{target_object}' has not been detected for skill {i+1} ({skill_name}). Use detect_object {target_object} first."
                        self.logger.error(error_msg)
                        return False, error_msg

                    object_info = detected_objects[target_object]
                    self.logger.info(f"Using detected object info for {target_object}")

                # Get camera images for skill generation
                color_image, depth_image = self.camera.get_frames()
                if color_image is None or depth_image is None:
                    error_msg = f"Failed to get RealSense camera images for skill {i+1}"
                    self.logger.error(error_msg)
                    return False, error_msg

                # Generate skill command
                skill_command = (f"place {params[0]} on {params[1]}"
                            if skill_name == 'place'
                            else f"{skill_name} {params[0]}")

                # Generate skill
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

                    # Execute the skill
                    _, initial_cam_tf = self.motion_planner.get_camera_transform()

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

            # Log success
            elapsed_time = time.time() - start_time
            success_msg = f"Successfully executed skill sequence of {len(skill_sequence)} skills using RealSense camera in {elapsed_time:.2f}s"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Skill sequence execution failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

        finally:
            self.execution_start_time = None

    def set_data_recorder(self, data_recorder):
        """Set the data recorder for motion tracking"""
        self.data_recorder = data_recorder
        self.logger.info("Data recorder set for motion tracking")

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

            if hasattr(self.current_object_info, 'image') and self.current_object_info.image is not None:
                surface_images['main_object'] = self.current_object_info.image

            if hasattr(self.current_object_info, 'surface_masks') and self.current_object_info.surface_masks:
                for surface_name, mask in self.current_object_info.surface_masks.items():
                    if mask is not None and hasattr(self.current_object_info, 'image'):
                        masked_image = self.current_object_info.image.copy()
                        masked_image[~mask] = 0
                        surface_images[surface_name] = masked_image

            if surface_images:
                execution_data['surface_images'] = surface_images

        # Add timing information
        if hasattr(self, 'execution_start_time') and self.execution_start_time:
            execution_data['execution_start_time'] = self.execution_start_time
            execution_data['current_time'] = time.time()
            execution_data['execution_duration'] = time.time() - self.execution_start_time

        # Add camera type and robot info
        execution_data['camera_type'] = "RealSense"
        execution_data['robot_ip'] = getattr(self.motion_planner.config, 'robot_ip', 'unknown')

        # Collect skill generator file information
        if hasattr(self, 'skill_handler') and self.skill_handler:
            if hasattr(self.skill_handler, 'skill_generator') and self.skill_handler.skill_generator:
                skill_gen_files = self.skill_handler.skill_generator.get_last_generation_files()
                execution_data['skill_generation_files'] = skill_gen_files

        return execution_data

    def shutdown(self):
        """Clean shutdown of the system"""
        self.logger.info("Shutting down streamlined skill execution system")

        # Stop the camera
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()
            self.logger.info("RealSense camera stopped")

        # Disconnect from robot
        if hasattr(self, 'motion_planner'):
            self.motion_planner.disconnect_robot()
            self.logger.info("Robot disconnected")
