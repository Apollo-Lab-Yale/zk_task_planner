from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import cv2
import asyncio
from pathlib import Path
import traceback
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig, ObjectInfo
from cognitive_bt_framework.src.skills.primitive_parser import PrimitiveParser, ParsedPrimitive
from cognitive_bt_framework.src.skills.skill_generator import SkillGenerator, PointOfInterest, Skill
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI

@dataclass
class ExecutableAction:
    """Data class for an executable primitive action with concrete parameters"""
    action_type: str  # e.g., 'push', 'pull', etc.
    position: np.ndarray  # 3D position [x, y, z]
    orientation: np.ndarray  # 3D orientation [roll, pitch, yaw]
    pixel_position: Optional[Tuple[int, int]]  # 2D pixel coordinates [x, y]
    parameters: Dict[str, Any]  # Additional parameters like force magnitude, speed, etc.
    is_top_down_grasp: bool
    is_side_grasp: bool
    
@dataclass
class InstantiatedSkill:
    """Data class for a skill instantiated for a specific object"""
    skill_name: str
    target_object: str
    action_sequence: List[ExecutableAction]
    execution_parameters: Dict[str, Any]
    object_pixel_pose: Tuple[int, int]  # 2D pixel coordinates of object center

    
    
class SyncSkillHandler:
    def __init__(self, skill_generator, perception_system):
        self.skill_handler = SkillHandler(skill_generator, perception_system)
        
    def instantiate_skill(self, skill_command, target_object, color_image=None, depth_image=None):
        """Synchronous wrapper for instantiate_skill"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            skill = loop.run_until_complete(
                self.skill_handler.instantiate_skill(
                    skill_command,
                    target_object,
                    color_image,
                    depth_image
                )
            )
            return skill
        finally:
            loop.close()

class SkillHandler:
    def __init__(self, skill_generator: SkillGenerator, perception_system: PerceptionSystem):
        """
        Initialize skill handler with required components
        
        Args:
            skill_generator: SkillGenerator instance for generating/retrieving skills
            perception_system: System for getting 3D scene information and object detection
        """
        self.skill_generator = skill_generator
        self.perception_system = perception_system
        self.primitive_parser = PrimitiveParser()
        self.debug = False
        
    @staticmethod
    def _get_alpha_id(num):
        """
        Convert a number to an alphabetical ID (a, b, c, ... aa, ab, ...)
        """
        if num <= 0:
            return ""
        
        letters = ""
        while num > 0:
            num, remainder = divmod(num - 1, 26)
            letters = chr(97 + remainder) + letters  # 97 is ASCII for 'a'
        
        return letters
        
    def instantiate_skill(
        self, 
        abstract_action: str, 
        target_object: str,
        image: np.ndarray,
        depth_image: Optional[np.ndarray] = None
    ) -> Optional[InstantiatedSkill]:
        """
        Generate and instantiate a skill for the given action and object
        
        Args:
            abstract_action: High-level action to perform
            target_object: Name of the target object
            image: RGB image of the scene
            depth_image: Optional depth image for better pose estimation
            
        Returns:
            Instantiated skill if successful, None otherwise
        """
        # Get object information from perception system
        object_info = self.perception_system.detect_object(
            target_object=target_object,
            image=image,
            depth_image=depth_image
        )
        
        if object_info is None:
            print(f"Could not detect object: {target_object}")
            return None
        
        # Detect regions of interest on the object
        roi_results = object_info.points
        
        # Create points of interest dictionary
        points_of_interest = {}
        for i, (pixel_x, pixel_y) in enumerate(roi_results['pixel_coords']):
            # Create uppercase alphabetical label
            label = roi_results['ids'][i]
            
            # Convert to normalized coordinates
            norm_x = pixel_x / image.shape[1]
            norm_y = pixel_y / image.shape[0]
            
            # Store in dictionary
            points_of_interest[label] = PointOfInterest(
                label=label,
                position=(norm_x, norm_y),
                description=f"Interest point {label} at normalized position {(norm_x, norm_y)}",
                pixel_coords=(pixel_x,pixel_y)
            )
        
        # Get or generate skill using SkillGenerator
        # skill = await self.skill_generator.find_similar_skill(
        #     image=object_info.image,
        #     points_of_interest=points_of_interest,
        #     abstract_action=abstract_action,
        #     target_object=target_object
        # )
        skill = None
        if skill is None:
            skill = self.skill_generator.generate_skill(
                image=object_info.image,
                points_of_interest=points_of_interest,
                abstract_action=abstract_action,
                target_object=target_object,
                object_info=object_info
            )
            
        if skill is None:
            print(f"Failed to generate skill for {abstract_action} on {target_object}")
            return None
            
        # Instantiate the skill with concrete parameters
        return self._instantiate_skill(skill, object_info)

        
    def _instantiate_skill(self, skill, object_info: ObjectInfo) -> Optional[InstantiatedSkill]:
        """Convert abstract skill to concrete executable actions"""
        try:
            parsed_primitives = self.primitive_parser.validate_primitive_sequence(
                skill.primitive_sequence
            )
        except ValueError as e:
            print(f"Failed to parse skill primitives: {e}")
            return None
        
        # Get 3D and pixel information about the object
        object_pose = object_info.pose if object_info.pose is not None else np.zeros(6)
        object_pixel_pose = object_info.pixel_pose
        
        # Map labeled points to 3D coordinates
        points_3d = {}
        points_pixel = {}
        
        h, w = object_info.image.shape[:2]
        
        for label, point in skill.points_of_interest.items():
            # Get normalized coordinates
            norm_x, norm_y = point.position
            
            # Convert to pixel coordinates
            pixel_x = int(norm_x * w)
            pixel_y = int(norm_y * h)
            
            # Store pixel coordinates
            points_pixel[label] = (pixel_x, pixel_y)
            
            # Convert to 3D coordinates if depth image is available
            if hasattr(object_info, 'depth_image') and object_info.depth_image is not None:
                try:
                    # Get depth at this point (with some averaging for robustness)
                    depth_roi = object_info.depth_image[
                        max(0, pixel_y-2):min(object_info.depth_image.shape[0], pixel_y+3),
                        max(0, pixel_x-2):min(object_info.depth_image.shape[1], pixel_x+3)
                    ]
                    # Filter out zero/invalid depths
                    points_3d[label], _ = self.perception_system._estimate_point_pose(points_pixel[label],
                                                                                   object_info.depth_image)
                except Exception as e:
                    print(f"Error estimating 3D position for point {label}: {e}")
                    points_3d[label] = object_pose[:3]
            else:
                # No depth image, use object pose
                points_3d[label] = object_pose[:3]
        
        # Convert each parsed primitive to executable action
        action_sequence = []
        for primitive in parsed_primitives:
            executable_action = self._convert_point_based_primitive(
                primitive,
                object_pose,
                object_pixel_pose,
                points_3d,
                points_pixel,
                skill.parameters
            )
            if executable_action is None:
                print(f"Failed to convert primitive: {primitive.raw_string}")
                return None
            action_sequence.append(executable_action)
        
        # Create execution parameters
        execution_parameters = self._generate_execution_parameters(
            skill.parameters,
            object_pose,
            points_3d
        )
        
        return InstantiatedSkill(
            skill_name=skill.name,
            target_object=skill.target_object,
            action_sequence=action_sequence,
            execution_parameters=execution_parameters,
            object_pixel_pose=object_pixel_pose
        )

    def _convert_point_based_primitive(self,
                              parsed_primitive,
                              object_pose,
                              object_pixel_pose,
                              points_3d,
                              points_pixel,
                              skill_parameters,
                              object_info=None) -> Optional[ExecutableAction]:
        """
        Convert a parsed primitive using surface labels or point labels to an executable action
        
        Args:
            parsed_primitive: The parsed primitive with action type and parameters
            object_pose: 3D pose of the object
            object_pixel_pose: Pixel coordinates of the object
            points_3d: Dictionary mapping point labels to 3D positions
            points_pixel: Dictionary mapping point labels to pixel positions
            skill_parameters: Parameters for the skill execution
            object_info: ObjectInfo containing surface masks and depth image
            
        Returns:
            ExecutableAction if conversion is successful, None otherwise
        """
        action_type = parsed_primitive.action_type
        parameters = parsed_primitive.parameters
        print(parsed_primitive)
        
        # Create case-insensitive lookup dictionaries
        points_3d_ci = {k.upper(): v for k, v in points_3d.items()}
        points_pixel_ci = {k.upper(): v for k, v in points_pixel.items()}
        
        # Handle push/pull actions that now use surface labels
        if action_type in ['push', 'pull']:
            # Get target surface with case-insensitive lookup
            surface_label = parameters.get('surface_label', '')
            surface_label_upper = surface_label.upper()
            
            # Calculate surface normal if object_info is available
            surface_normal = None
            surface_centroid_position = None
            surface_centroid_pixel = None
            
            if object_info is not None and object_info.surface_masks is not None and object_info.depth_image is not None:
                # Find the surface mask based on the label
                surface_mask = None
                
                # Look for the surface by label in format "surface_X"
                for name, mask in object_info.surface_masks.items():
                    # Extract the alphabetical ID from the surface name
                    if "_" in name:
                        mask_id = name.split("_")[1].upper()
                        if mask_id == surface_label_upper:
                            surface_mask = mask
                            break
                
                # If we found a matching surface mask
                if surface_mask is not None:
                    # Calculate centroid of the surface mask
                    y_coords, x_coords = np.where(surface_mask)
                    if len(y_coords) > 0:
                        # Calculate pixel centroid
                        centroid_y = int(np.mean(y_coords))
                        centroid_x = int(np.mean(x_coords))
                        surface_centroid_pixel = (centroid_x, centroid_y)
                        
                        # Get depth at the centroid
                        if 0 <= centroid_y < object_info.depth_image.shape[0] and 0 <= centroid_x < object_info.depth_image.shape[1]:
                            depth = object_info.depth_image[centroid_y, centroid_x]
                            
                            # Convert to 3D position using camera intrinsics
                            # Note: This assumes we have access to camera intrinsics
                            # which should be available in the perception system
                            # For now, we'll use the object's own conversion method if available
                            if hasattr(self, '_pixel_to_3d'):
                                surface_centroid_position = self._pixel_to_3d(centroid_x, centroid_y, depth)
                            else:
                                # Fallback: Estimate position using simple pinhole model
                                # This assumes fx, fy, cx, cy are available somewhere
                                # You may need to adjust this based on your actual implementation
                                fx = 429.92523193359375  # Default from perception system
                                fy = 429.92523193359375
                                cx = 431.7160339355469
                                cy = 233.39739990234375
                                depth_scale = 0.001  # Default scale factor (m/unit)
                                
                                # Convert to meters
                                z = depth * depth_scale
                                x = (centroid_x - cx) * z / fx
                                y = (centroid_y - cy) * z / fy
                                surface_centroid_position = np.array([x, y, z])
                        
                        # Calculate surface normal using the depth image and mask
                        surface_normal = self._calculate_surface_normal(
                            object_info.depth_image,
                            surface_mask,
                            depth_scale=0.001  # Default scale factor (m/unit)
                        )
            
            # If we couldn't find the surface or calculate normal, use fallback
            if surface_centroid_position is None or surface_normal is None:
                print(f"Surface '{surface_label}' not found or normal calculation failed")
                
                # Fall back to using a point if available
                if points_3d:
                    print(f"Using first available point as fallback")
                    fallback_label = list(points_3d.keys())[0]
                    surface_centroid_position = points_3d[fallback_label]
                    surface_centroid_pixel = points_pixel[fallback_label]
                    
                    # Create a default normal pointing along Z-axis
                    surface_normal = np.array([0, 0, 1])
                else:
                    return None
            
            # Get pivot point if specified
            pivot_position = None
            if parameters.get('has_pivot', False) and 'pivot_point_label' in parameters:
                pivot_label = parameters['pivot_point_label']
                if pivot_label:
                    pivot_label_upper = pivot_label.upper()
                    if pivot_label_upper in points_3d_ci:
                        pivot_position = points_3d_ci[pivot_label_upper]
                    else:
                        print(f"Pivot point label '{pivot_label}' not found in points_3d")
            
            # Create parameters dictionary
            action_params = {
                'force_magnitude': self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium')),
                'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium')),
                'precision': skill_parameters.get('precision_required', 'medium'),
                'is_parallel': parameters.get('force_direction', 'perpendicular') == 'parallel',
                'is_button': parameters.get('is_button', False),
                'has_pivot': parameters.get('has_pivot', False),
                'surface_label': surface_label,
                'surface_normal': surface_normal  # Add the calculated normal to parameters
            }
            
            # Add pivot position if available
            if pivot_position is not None:
                action_params['pivot_position'] = pivot_position
                action_params['pivot_label'] = parameters.get('pivot_point_label')
            
            # Determine approach direction and orientation based on surface normal and parameters
            orientation = self._calculate_orientation_from_normal(
                surface_normal,
                pivot_position if pivot_position is not None else None,
                action_params['is_parallel'],
                action_type
            )
            
            return ExecutableAction(
                action_type=action_type,
                position=surface_centroid_position,
                orientation=orientation,
                pixel_position=surface_centroid_pixel,
                parameters=action_params,
                is_top_down_grasp=False,
                is_side_grasp=False
            )
        
        # Handle move_gripper_to_pose (which still uses point labels)
        elif action_type == 'move_gripper_to_pose':
            # Get target point with case-insensitive lookup
            point_label = parameters.get('point_label', '')
            point_label_upper = point_label.upper()
            
            if point_label_upper not in points_3d_ci:
                print(f"Point label '{point_label}' not found in points_3d")
                # Check if we have any points at all
                if points_3d:
                    # Use the first available point as fallback
                    print(f"Using first available point as fallback")
                    fallback_label = list(points_3d.keys())[0]
                    target_position = points_3d[fallback_label]
                    target_pixel = points_pixel[fallback_label]
                else:
                    return None
            else:
                target_position = points_3d_ci[point_label_upper]
                target_pixel = points_pixel_ci[point_label_upper]
            
            # Create parameters dictionary
            action_params = {
                'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium')),
                'precision': skill_parameters.get('precision_required', 'medium'),
                'point_label': point_label
            }
            
            # Get grasp approach parameters
            is_top_down_grasp = parameters.get('is_top_down_grasp', True)
            is_side_grasp = parameters.get('is_side_grasp', False)
            
            # Calculate orientation based on grasp approach
            if is_top_down_grasp:
                # Top-down approach - gripper aligned with Z-axis
                orientation = [0, 0, -1, 0]  # Quaternion for pointing down
            elif is_side_grasp:
                # Side approach - gripper aligned with X or Y axis
                # The exact orientation should be calculated based on object position
                orientation = self._calculate_side_grasp_orientation(target_position, object_pose)
            else:
                # Default orientation
                orientation = [0, 0, 0, 1]  # Identity quaternion
            
            return ExecutableAction(
                action_type=action_type,
                position=target_position,
                orientation=orientation,
                pixel_position=target_pixel,
                parameters=action_params,
                is_top_down_grasp=is_top_down_grasp,
                is_side_grasp=is_side_grasp
            )
        
        # Handle simple actions without points
        elif action_type in ['close_gripper', 'open_gripper', 'retract_gripper']:
            return ExecutableAction(
                action_type=action_type,
                position=None,
                orientation=None,
                pixel_position=None,
                parameters={},
                is_top_down_grasp=False,
                is_side_grasp=False
            )
        
        else:
            print(f"Unsupported action type: {action_type}")
            return None

    
    def _get_force_magnitude(self, force_threshold: str) -> float:
        """Convert force threshold to concrete value"""
        force_values = {
            'low': 5.0,
            'medium': 10.0,
            'high': 20.0
        }
        return force_values.get(force_threshold.lower(), 10.0)

    def _get_speed_value(self, speed_requirement: str) -> float:
        """Convert speed requirement to concrete value"""
        speed_values = {
            'slow': 0.1,
            'medium': 0.3,
            'fast': 0.5
        }
        return speed_values.get(speed_requirement.lower(), 0.3)

    def _generate_execution_parameters(self,
                                    skill_parameters: Dict[str, Any],
                                    object_pose: np.ndarray,
                                    points_3d: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate concrete execution parameters from abstract skill parameters"""
        execution_params = {}
        
        # Convert abstract parameters to concrete values
        if 'speed_requirement' in skill_parameters:
            execution_params['max_speed'] = self._get_speed_value(skill_parameters['speed_requirement'])
            
        if 'precision_required' in skill_parameters:
            precision = skill_parameters['precision_required'].lower()
            execution_params['position_tolerance'] = {
                'low': 0.02,    # 2 cm
                'medium': 0.01, # 1 cm
                'high': 0.005   # 5 mm
            }.get(precision, 0.01)
            
            execution_params['orientation_tolerance'] = {
                'low': 0.1,    # ~5.7 degrees
                'medium': 0.05, # ~2.9 degrees
                'high': 0.02   # ~1.1 degrees
            }.get(precision, 0.05)
            
        if 'force_threshold' in skill_parameters:
            execution_params['max_force'] = self._get_force_magnitude(skill_parameters['force_threshold'])
            
        # Add object-specific parameters
        execution_params['object_pose'] = object_pose
        execution_params['points_3d'] = points_3d
        
        # Add safety parameters
        execution_params['force_monitoring'] = True
        execution_params['collision_detection'] = True
        execution_params['velocity_scaling'] = 0.8  # 80% of maximum speed for safety
        
        return execution_params

    def _calculate_approach_orientation_for_torque(self, axis: str, object_pose: np.ndarray) -> np.ndarray:
        """Calculate approach orientation for torque application"""
        orientation = np.zeros(3)
        if axis == 'clockwise' or axis == 'counterclockwise':
            # Approach perpendicular to the rotation axis
            orientation[2] = 0 if axis == 'clockwise' else np.pi
            
        # Adjust based on object pose
        orientation += object_pose[3:]
        return orientation

    def visualize_skill(self, skill: InstantiatedSkill, image: np.ndarray, object_info: ObjectInfo) -> np.ndarray:
        """
        Visualize skill execution on image using object pixel pose
        
        Args:
            skill: InstantiatedSkill to visualize
            image: RGB image to visualize on
            object_info: Object information including pixel pose
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Define colors for different action types
        COLOR_MAP = {
            'push': (0, 255, 0),      # Green
            'pull': (0, 200, 0),      # Darker green
            'move_gripper_to_pose': (255, 0, 0),  # Red
            'close_gripper': (0, 0, 255),      # Blue
            'release': (255, 165, 0),          # Orange
            'retract_gripper': (128, 0, 128)    # Purple
        }
        
        # Draw object center and pose axes
        object_center = object_info.pixel_pose
        cv2.circle(vis_image, object_center, 7, (255, 255, 255), -1)  # White circle for object center
        cv2.circle(vis_image, object_center, 5, (0, 0, 0), -1)  # Black center
        
        # Draw each action in sequence
        text_offset = 20  # Initial vertical offset for text
        for i, action in enumerate(skill.action_sequence):
            # Get color for this action type
            color = COLOR_MAP.get(action.action_type, (200, 200, 200))
            
            # For actions with position
            if not np.all(action.position == 0):
                # Get action position relative to object center
                rel_x = action.position[0] * image.shape[1]
                rel_y = action.position[1] * image.shape[0]
                
                # Calculate position relative to object center in pixel coordinates
                pos_2d = (
                    int(object_center[0] + rel_x),
                    int(object_center[1] + rel_y)
                )
                
                # Ensure point is within image bounds
                pos_2d = (
                    max(0, min(pos_2d[0], image.shape[1] - 1)),
                    max(0, min(pos_2d[1], image.shape[0] - 1))
                )
                
                # Draw action point
                cv2.circle(vis_image, pos_2d, 5, color, -1)
                
                # Draw connection line to object center for move_gripper_to_pose
                if action.action_type == 'move_gripper_to_pose':
                    cv2.line(vis_image, object_center, pos_2d, color, 1, cv2.LINE_AA)
                
                # Draw action type and parameters
                label = f"{i+1}. {action.action_type}"
                
                # Add specific parameters based on action type
                if action.action_type == 'move_gripper_to_pose':
                    if action.parameters.get('is_grasp', False):
                        label += " (grasp)"
                        # Draw gripper fingers
                        angle = np.arctan2(pos_2d[1] - object_center[1], 
                                        pos_2d[0] - object_center[0])
                        finger_length = 15
                        finger_spread = 5
                        
                        # Calculate finger positions perpendicular to grasp direction
                        perp_angle = angle + np.pi/2
                        finger1_start = (
                            int(pos_2d[0] + finger_spread * np.cos(perp_angle)),
                            int(pos_2d[1] + finger_spread * np.sin(perp_angle))
                        )
                        finger2_start = (
                            int(pos_2d[0] - finger_spread * np.cos(perp_angle)),
                            int(pos_2d[1] - finger_spread * np.sin(perp_angle))
                        )
                        
                        # Draw fingers
                        cv2.line(vis_image, 
                                finger1_start,
                                (int(finger1_start[0] - finger_length * np.cos(angle)),
                                int(finger1_start[1] - finger_length * np.sin(angle))),
                                color, 2)
                        cv2.line(vis_image,
                                finger2_start,
                                (int(finger2_start[0] - finger_length * np.cos(angle)),
                                int(finger2_start[1] - finger_length * np.sin(angle))),
                                color, 2)
                
                cv2.putText(
                    vis_image,
                    label,
                    (pos_2d[0] + 10, pos_2d[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
                
                # Draw arrows for push/pull actions
                if action.action_type in ['push', 'pull']:
                    # Get distance from parameters
                    distance = action.parameters.get('distance', 0.1)
                    # Scale distance for visualization (20 pixels per 0.1m)
                    arrow_length = int(distance * 200)
                    
                    # Get approach orientation
                    orientation = action.orientation
                    if not np.all(orientation == 0):
                        # Convert orientation to direction vector
                        # Simplified - in production would use proper rotation matrix
                        pitch, yaw = orientation[1], orientation[2]
                        direction = np.array([
                            np.cos(pitch) * np.cos(yaw),
                            np.cos(pitch) * np.sin(yaw),
                            np.sin(pitch)
                        ])
                        
                        # Project 3D direction to 2D image plane (simplified)
                        angle = np.arctan2(direction[1], direction[0])
                    else:
                        # Fallback to simple direction from object center
                        angle = np.arctan2(pos_2d[1] - object_center[1], 
                                        pos_2d[0] - object_center[0])
                    
                    # Adjust direction based on push/pull
                    if action.action_type == 'pull':
                        angle += np.pi  # Reverse direction for pull
                    
                    end_point = (
                        int(pos_2d[0] + arrow_length * np.cos(angle)),
                        int(pos_2d[1] + arrow_length * np.sin(angle))
                    )
                    
                    cv2.arrowedLine(
                        vis_image,
                        pos_2d,
                        end_point,
                        color,
                        2,
                        tipLength=0.3
                    )
                    
                    # Add distance label
                    cv2.putText(
                        vis_image,
                        f"{distance:.2f}m",
                        (end_point[0] + 5, end_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1
                    )
                    
                    # Add surface information
                    if 'surface_keywords' in action.parameters and action.parameters['surface_keywords']:
                        surface_info = f"Surface: {', '.join(action.parameters['surface_keywords'])}"
                        cv2.putText(
                            vis_image,
                            surface_info,
                            (pos_2d[0] + 10, pos_2d[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1
                        )
                    
                    # Show parallel/perpendicular indication
                    if 'is_parallel_surface' in action.parameters:
                        approach_type = "Parallel" if action.parameters['is_parallel_surface'] else "Perpendicular"
                        cv2.putText(
                            vis_image,
                            approach_type,
                            (pos_2d[0] + 10, pos_2d[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1
                        )
                    
                    # Draw pivot point if available
                    if action.parameters.get('has_pivot', False) and 'pivot_position' in action.parameters:
                        pivot_pos = action.parameters['pivot_position']
                        
                        # Convert 3D position to 2D pixel coordinates (simplified)
                        pivot_pixel = (
                            int(object_center[0] + pivot_pos[0] * image.shape[1]),
                            int(object_center[1] + pivot_pos[1] * image.shape[0])
                        )
                        
                        # Ensure within image bounds
                        pivot_pixel = (
                            max(0, min(pivot_pixel[0], image.shape[1] - 1)),
                            max(0, min(pivot_pixel[1], image.shape[0] - 1))
                        )
                        
                        # Draw pivot point
                        cv2.circle(vis_image, pivot_pixel, 5, (0, 255, 255), -1)  # Yellow for pivot
                        cv2.putText(
                            vis_image,
                            "Pivot",
                            (pivot_pixel[0] + 5, pivot_pixel[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 255),
                            1
                        )
                        
                        # Draw line from interaction point to pivot
                        cv2.line(vis_image, pos_2d, pivot_pixel, (0, 255, 255), 1, cv2.LINE_AA)
            
            else:
                # For actions without position (close_gripper, release, retract_gripper)
                cv2.putText(
                    vis_image,
                    f"{i+1}. {action.action_type}",
                    (10, text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
                text_offset += 20
        
        # Add legend
        legend_offset = 20
        legend_x = image.shape[1] - 150
        for action_type, color in COLOR_MAP.items():
            cv2.putText(
                vis_image,
                action_type,
                (legend_x, legend_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            legend_offset += 15
        
        return vis_image
    
    def visualize_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5,
        show_contour: bool = True,
        contour_thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize a segmentation mask on an image
        
        Args:
            image: RGB image to visualize on
            mask: Binary segmentation mask
            color: RGB color for the mask overlay
            alpha: Transparency of the mask overlay (0-1)
            show_contour: Whether to show mask contour
            contour_thickness: Thickness of contour lines
            
        Returns:
            Visualization image with the mask overlay
        """
        if self.debug:
            self.profiler.start("visualize_mask")
        
        try:
            # Check inputs
            if image is None or mask is None:
                if self.debug:
                    print("Cannot visualize mask: image or mask is None")
                return image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Convert mask to binary if not already
            binary_mask = mask.astype(bool)
            
            # Create a copy of the image for visualization
            vis_image = image.copy()
            
            # Create colored mask overlay
            mask_overlay = np.zeros_like(vis_image)
            mask_overlay[binary_mask] = color
            
            # Blend the mask overlay with the original image
            vis_image = cv2.addWeighted(vis_image, 1.0, mask_overlay, alpha, 0)
            
            # Add contour if requested
            if show_contour:
                # Find contours of the mask
                mask_uint8 = binary_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours on the visualization
                cv2.drawContours(vis_image, contours, -1, color, contour_thickness)
            
            return vis_image
            
        except Exception as e:
            if self.debug:
                print(f"Error in visualize_mask: {str(e)}")
                traceback.print_exc()
            # Return original image on error
            return image.copy()
            
        finally:
            if self.debug:
                self.profiler.stop("visualize_mask")
                
    def _calculate_surface_normal(self, depth_image, mask, depth_scale=0.001, window_size=5):
        """
        Calculate the surface normal from a depth image and mask using PCA
        
        Args:
            depth_image: Depth image
            mask: Binary mask of the surface
            depth_scale: Scale factor to convert depth values to meters
            window_size: Size of window for normal calculation
            
        Returns:
            Surface normal vector (3D unit vector)
        """
        # Find all pixel coordinates in the mask
        y_coords, x_coords = np.where(mask)
        
        # If not enough points, return default normal
        if len(y_coords) < 10:
            return np.array([0, 0, 1])  # Default normal along Z axis
        
        # Sample a reasonable number of points for efficiency if there are too many
        max_points = 1000
        if len(y_coords) > max_points:
            indices = np.random.choice(len(y_coords), max_points, replace=False)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        # Set up camera intrinsics (use defaults or get from your system)
        fx = 429.92523193359375  # Default from perception system
        fy = 429.92523193359375
        cx = 431.7160339355469
        cy = 233.39739990234375
        
        # Collect valid 3D points
        points_3d = []
        
        # For each pixel in the mask
        for x, y in zip(x_coords, y_coords):
            # Extract a window around this pixel
            x_min = max(0, x - window_size // 2)
            x_max = min(depth_image.shape[1] - 1, x + window_size // 2)
            y_min = max(0, y - window_size // 2)
            y_max = min(depth_image.shape[0] - 1, y + window_size // 2)
            
            window = depth_image[y_min:y_max+1, x_min:x_max+1]
            
            # If the window has valid depth values
            if np.any(window > 0):
                # Get depth at this pixel
                depth = depth_image[y, x]
                
                # Skip pixels with invalid depth
                if depth <= 0:
                    continue
                
                # Convert to meters
                z = depth * depth_scale
                
                # Apply pinhole camera model to get 3D coordinates
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                z_3d = z
                
                points_3d.append([x_3d, y_3d, z_3d])
        
        # If we don't have enough points for PCA, return default normal
        if len(points_3d) < 3:
            return np.array([0, 0, 1])
        
        # Convert to numpy array
        points_3d = np.array(points_3d)
        
        # Calculate covariance matrix
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        covariance_matrix = np.dot(centered_points.T, centered_points) / centered_points.shape[0]
        
        try:
            # Use SVD to find the normal (eigenvector with smallest eigenvalue)
            u, s, vh = np.linalg.svd(covariance_matrix)
            normal = u[:, 2]  # The last column of U contains the normal vector
            
            # Ensure the normal points towards the camera (negative Z direction)
            if normal[2] > 0:
                normal = -normal
            
            # Normalize the vector
            normal = normal / np.linalg.norm(normal)
            
            return normal
            
        except np.linalg.LinAlgError:
            # Fallback to default normal
            return np.array([0, 0, 1])

    def _calculate_orientation_from_normal(self, normal, pivot_position=None, is_parallel=False, action_type='push'):
        """
        Calculate approach orientation based on surface normal
        
        Args:
            normal: Surface normal vector (3D unit vector)
            pivot_position: Position of pivot point (if applicable)
            is_parallel: Whether the force should be parallel to the surface
            action_type: Type of action ('push' or 'pull')
            
        Returns:
            Orientation as a quaternion or rotation matrix
        """
        # Create a coordinate system based on the normal
        z_axis = np.array(normal)
        
        # Ensure it's normalized
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Create orthogonal axes
        # Find a vector not collinear with z_axis
        if abs(z_axis[0]) < abs(z_axis[1]) and abs(z_axis[0]) < abs(z_axis[2]):
            temp = np.array([1, 0, 0])
        elif abs(z_axis[1]) < abs(z_axis[2]):
            temp = np.array([0, 1, 0])
        else:
            temp = np.array([0, 0, 1])
        
        # Calculate y-axis
        y_axis = np.cross(z_axis, temp)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Calculate x-axis
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # If parallel, adjust the approach direction
        if is_parallel:
            # For parallel approach, we need to use a different axis
            if action_type == 'push':
                approach_axis = x_axis  # Use x-axis for parallel push
            else:  # pull
                approach_axis = -x_axis  # Use negative x-axis for parallel pull
        else:
            # For perpendicular approach, use the normal
            if action_type == 'push':
                approach_axis = z_axis  # Use normal for perpendicular push
            else:  # pull
                approach_axis = -z_axis  # Use negative normal for perpendicular pull
        
        # If pivot is specified, adjust approach direction
        if pivot_position is not None:
            # This would require additional geometry calculations
            # For now, we'll use the same approach as without pivot
            pass
        
        # Create rotation matrix where z-axis is the approach direction
        rotation_matrix = np.column_stack((x_axis, y_axis, approach_axis))
        
        # Convert to quaternion (if your system uses quaternions)
        # This is a simplified conversion and may need adjustment
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w] format
        
        # Reorder to [w, x, y, z] if needed
        # quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        
        return quaternion
    
    def _calculate_side_grasp_orientation(self, target_position, object_pose):
        """
        Calculate orientation for a side grasp approach
        
        Args:
            target_position: 3D position of the target point
            object_pose: 3D pose of the object
            
        Returns:
            Orientation quaternion for side grasp
        """
        # Calculate vector from object center to target position
        if object_pose is not None:
            object_position = object_pose[:3]  # Extract position part
            approach_vector = target_position - object_position
        else:
            # If object pose is unknown, default to horizontal approach
            approach_vector = np.array([1, 0, 0])
        
        # Project onto horizontal plane (ignore Z component)
        approach_vector[2] = 0
        
        # If the vector is too small, use a default approach
        if np.linalg.norm(approach_vector) < 0.001:
            approach_vector = np.array([1, 0, 0])
        
        # Normalize the vector
        approach_vector = approach_vector / np.linalg.norm(approach_vector)
        
        # Create a coordinate system
        z_axis = np.array([0, 0, 1])  # Upward
        x_axis = approach_vector  # Approach direction
        y_axis = np.cross(z_axis, x_axis)  # Perpendicular to both
        
        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Convert to quaternion
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w] format
        
        # Reorder to [w, x, y, z] if needed
        # quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        
        return quaternion
    
    def _calculate_side_grasp_orientation(self, target_position, object_pose):
        """
        Calculate orientation for a side grasp approach
        
        Args:
            target_position: 3D position of the target point
            object_pose: 3D pose of the object
            
        Returns:
            Orientation quaternion for side grasp
        """
        # Calculate vector from object center to target position
        if object_pose is not None:
            object_position = object_pose[:3]  # Extract position part
            approach_vector = target_position - object_position
        else:
            # If object pose is unknown, default to horizontal approach
            approach_vector = np.array([1, 0, 0])
        
        # Project onto horizontal plane (ignore Z component)
        approach_vector[2] = 0
        
        # If the vector is too small, use a default approach
        if np.linalg.norm(approach_vector) < 0.001:
            approach_vector = np.array([1, 0, 0])
        
        # Normalize the vector
        approach_vector = approach_vector / np.linalg.norm(approach_vector)
        
        # Create a coordinate system
        z_axis = np.array([0, 0, 1])  # Upward
        x_axis = approach_vector  # Approach direction
        y_axis = np.cross(z_axis, x_axis)  # Perpendicular to both
        
        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Convert to quaternion
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w] format
        
        # Reorder to [w, x, y, z] if needed
        # quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        
        return quaternion

    def visualize_detection(
        self,
        image: np.ndarray,
        object_info: ObjectInfo,
        show_pose: bool = True,
        show_components: bool = True
    ) -> np.ndarray:
        """
        Visualize a detected object with its mask, bounding box, and pose
        
        Args:
            image: RGB image to visualize on
            object_info: ObjectInfo for the detected object
            show_pose: Whether to show pose axes
            show_components: Whether to show object components
            
        Returns:
            Visualization image
        """
        if self.debug:
            self.profiler.start("visualize_detection")
        
        try:
            if image is None or object_info is None:
                if self.debug:
                    print("Cannot visualize detection: image or object_info is None")
                return image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                
            # Start with a copy of the original image
            vis_image = image.copy()
            
            # Draw mask overlay if available
            if object_info.mask is not None:
                # Apply mask visualization with semi-transparency
                vis_image = self.visualize_mask(
                    image=vis_image,
                    mask=object_info.mask,
                    color=(0, 255, 0),  # Green for main object
                    alpha=0.3,
                    show_contour=True
                )
            
            # Draw bounding box
            x, y, w, h = object_info.bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"{object_info.name} ({object_info.confidence:.2f})"
            cv2.putText(
                vis_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            # Draw pose if available and requested
            if show_pose and object_info.pose is not None and object_info.pixel_pose is not None:
                # Use pixel_pose as the origin for visualization
                origin = object_info.pixel_pose
                
                # Get pose angles
                pose = object_info.pose
                
                # Create a pose legend
                pose_text = f"Position: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})m"
                cv2.putText(
                    vis_image,
                    pose_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Draw coordinate axes (X, Y, Z) at the object center
                axis_length = 50  # pixels
                
                # X-axis (red)
                end_x = (
                    int(origin[0] + axis_length * np.cos(pose[5])),
                    int(origin[1] + axis_length * np.sin(pose[5]))
                )
                cv2.arrowedLine(vis_image, origin, end_x, (0, 0, 255), 2)
                
                # Y-axis (green)
                end_y = (
                    int(origin[0] - axis_length * np.sin(pose[5])),
                    int(origin[1] + axis_length * np.cos(pose[5]))
                )
                cv2.arrowedLine(vis_image, origin, end_y, (0, 255, 0), 2)
                
                # Z-axis (blue) - pointing up/down based on pitch
                z_factor = np.cos(pose[4])  # Adjust length based on pitch
                end_z = (
                    int(origin[0] + axis_length * np.sin(pose[4]) * np.cos(pose[5])),
                    int(origin[1] + axis_length * np.sin(pose[4]) * np.sin(pose[5]) - 
                        axis_length * z_factor)
                )
                cv2.arrowedLine(vis_image, origin, end_z, (255, 0, 0), 2)
            
            # Draw components if available and requested
            if show_components and object_info.components is not None and len(object_info.components) > 0:
                # Draw each component
                for comp_name, comp_info in object_info.components.items():
                    # Apply component mask visualization with different color
                    if comp_info.mask is not None:
                        vis_image = self.visualize_mask(
                            image=vis_image,
                            mask=comp_info.mask,
                            color=(255, 0, 0),  # Red for components
                            alpha=0.2,
                            show_contour=True,
                            contour_thickness=1
                        )
                    
                    # Draw component bounding box
                    cx, cy, cw, ch = comp_info.bbox
                    cv2.rectangle(vis_image, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 1)
                    
                    # Add component label
                    comp_label = f"{comp_name} ({comp_info.confidence:.2f})"
                    cv2.putText(
                        vis_image,
                        comp_label,
                        (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )
            
            return vis_image
            
        except Exception as e:
            if self.debug:
                print(f"Error in visualize_detection: {str(e)}")
                traceback.print_exc()
            # Return original image on error
            return image.copy()
            
        finally:
            if self.debug:
                self.profiler.stop("visualize_detection")
    
def create_test_rgb_image(width=640, height=480):
    """Create a test RGB image with a simple shape."""
    # Create a black background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a red "cup-like" shape in the middle
    center_x, center_y = width // 2, height // 2
    
    # Draw cup body (rectangle)
    cup_width, cup_height = 100, 120
    x1 = center_x - cup_width // 2
    x2 = center_x + cup_width // 2
    y1 = center_y - cup_height // 2
    y2 = center_y + cup_height // 2
    
    # Red cup
    image[y1:y2, x1:x2] = [255, 0, 0]
    
    return image

def create_test_depth_image(width=640, height=480):
    """Create a test depth image with a simple shape."""
    # Create a background (far depth)
    depth = np.ones((height, width), dtype=np.float32) * 2.0
    
    # Add closer depth values for the cup shape
    center_x, center_y = width // 2, height // 2
    cup_width, cup_height = 100, 120
    x1 = center_x - cup_width // 2
    x2 = center_x + cup_width // 2
    y1 = center_y - cup_height // 2
    y2 = center_y + cup_height // 2
    
    # Set cup depth (closer to camera)
    depth[y1:y2, x1:x2] = 0.5
    
    return depth


async def main():
    # Initialize systems
    fast_sam_config = FastSAMConfig()
    perception_system = PerceptionSystem(fast_sam_config)
    llm_interface = LLMInterfaceOpenAI()
    skill_generator = SkillGenerator(llm_interface=llm_interface)
    # Initialize skill handler with perception system
    skill_handler = SkillHandler(
        skill_generator=skill_generator,
        perception_system=perception_system
    )

    # Use in workflow
    rgb_image = create_test_rgb_image()
    depth_image = create_test_depth_image()
    target_object = "cup"

    # Detect object
    object_info = perception_system.detect_object(target_object, rgb_image, depth_image)

    # Generate skill
    skill = await skill_handler.instantiate_skill("pick", target_object, rgb_image, depth_image)

if __name__ == '__main__':
    asyncio.run(main())