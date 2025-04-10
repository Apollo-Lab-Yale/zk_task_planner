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
            label = self._get_alpha_id(i + 1)  # A, B, C, ...
            
            # Convert to normalized coordinates
            norm_x = pixel_x / image.shape[1]
            norm_y = pixel_y / image.shape[0]
            
            # Store in dictionary
            points_of_interest[label] = PointOfInterest(
                label=label,
                position=(norm_x, norm_y),
                description=f"Interest point {label}"
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
                              skill_parameters) -> Optional[ExecutableAction]:
        """Convert a parsed primitive using point labels to an executable action"""
        action_type = parsed_primitive.action_type
        parameters = parsed_primitive.parameters
        print(parsed_primitive)
        
        # Create case-insensitive lookup dictionaries
        points_3d_ci = {k.upper(): v for k, v in points_3d.items()}
        points_pixel_ci = {k.upper(): v for k, v in points_pixel.items()}
        
        # Handle push/pull actions
        if action_type in ['push', 'pull']:
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
                'point_label': point_label
            }
            
            # Add pivot position if available
            if pivot_position is not None:
                action_params['pivot_position'] = pivot_position
                action_params['pivot_label'] = parameters.get('pivot_point_label')
            
            # Determine approach direction and orientation based on parameters
            orientation = self._calculate_approach_orientation(
                target_position, 
                pivot_position if pivot_position is not None else None,
                action_params['is_parallel'],
                action_type
            )
            
            return ExecutableAction(
                action_type=action_type,
                position=target_position,
                orientation=orientation,
                pixel_position=target_pixel,
                parameters=action_params,
                is_top_down_grasp=False,
                is_side_grasp=False
            )
            
        elif action_type == 'move_gripper_to_pose':
            # Add debug information about the incoming parameters
            print(f"DEBUG: Converting move_gripper_to_pose primitive")
            print(f"DEBUG: Raw parameters: {parameters}")
            print(f"DEBUG: Available points_3d keys: {list(points_3d.keys())}")
            print(f"DEBUG: Available points_3d_ci keys: {list(points_3d_ci.keys())}")
            
            # Get target point with case-insensitive lookup
            point_label = parameters.get('point_label', '')
            print(f"DEBUG: Extracted point_label: '{point_label}', type: {type(point_label)}")
            
            # Check if point_label is a string before trying to uppercase
            if not isinstance(point_label, str):
                print(f"DEBUG: point_label is not a string! Converting to string first.")
                point_label = str(point_label)
                
            point_label_upper = point_label.upper()
            print(f"DEBUG: Uppercased point_label: '{point_label_upper}'")
            
            if point_label_upper not in points_3d_ci:
                print(f"Point label '{point_label}' not found in points_3d")
                # Check if we have any points at all
                if points_3d:
                    print(f"Using first available point as fallback")
                    fallback_label = list(points_3d.keys())[0]
                    print(f"DEBUG: Fallback to point '{fallback_label}'")
                    target_position = points_3d[fallback_label]
                    target_pixel = points_pixel[fallback_label]
                else:
                    print("DEBUG: No points available at all, returning None")
                    return None
            else:
                print(f"DEBUG: Found target position for point '{point_label_upper}'")
                target_position = points_3d_ci[point_label_upper]
                target_pixel = points_pixel_ci[point_label_upper]
            
            # Check grasp parameters
            is_top_down_grasp = parameters.get('is_top_down_grasp', False)
            is_side_grasp = parameters.get('is_side_grasp', False)
            print(f"DEBUG: is_top_down_grasp: {is_top_down_grasp}, type: {type(is_top_down_grasp)}")
            print(f"DEBUG: is_side_grasp: {is_side_grasp}, type: {type(is_side_grasp)}")
            
            # Add type conversion in case the boolean values are strings
            if isinstance(is_top_down_grasp, str):
                print(f"DEBUG: Converting is_top_down_grasp from string to boolean")
                is_top_down_grasp = is_top_down_grasp.lower() == 'true'
            if isinstance(is_side_grasp, str):
                print(f"DEBUG: Converting is_side_grasp from string to boolean")
                is_side_grasp = is_side_grasp.lower() == 'true'
            
            print(f"DEBUG: After conversion - is_top_down_grasp: {is_top_down_grasp}, is_side_grasp: {is_side_grasp}")
            
            # Create parameters dictionary
            action_params = {
                'point_label': point_label,
                'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium')),
                'precision': skill_parameters.get('precision_required', 'medium')
            }
            
            # Add grasp parameters if applicable
            if is_top_down_grasp or is_side_grasp:
                action_params['force'] = self._get_force_magnitude(
                    skill_parameters.get('force_threshold', 'medium')
                )
                action_params['grasp_planning_required'] = True
            
            # Determine approach orientation based on grasp type
            orientation = np.zeros(3)
            if is_top_down_grasp:
                # Approach from above (negative Z)
                orientation = np.array([0, 0, 0])  # This will be handled by grasp planner
            elif is_side_grasp:
                # Approach from side (horizontal)
                orientation = np.array([np.pi/2, 0, 0])  # This will be handled by grasp planner
            
            print(f"DEBUG: Successfully created executable action for move_gripper_to_pose")
            return ExecutableAction(
                action_type=action_type,
                position=target_position,
                orientation=orientation,
                pixel_position=target_pixel,
                parameters=action_params,
                is_top_down_grasp=is_top_down_grasp,
                is_side_grasp=is_side_grasp
            )
            
        # Handle simpler actions
        elif action_type == 'close_gripper':
            return ExecutableAction(
                action_type='close_gripper',
                position=np.zeros(3),
                orientation=np.zeros(3),
                pixel_position=object_pixel_pose,
                parameters={'force': self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium'))},
                is_top_down_grasp=False,
                is_side_grasp=False
            )
            
        elif action_type == 'open_gripper':
            return ExecutableAction(
                action_type='open_gripper',
                position=np.zeros(3),
                orientation=np.zeros(3),
                pixel_position=object_pixel_pose,
                parameters={},
                is_top_down_grasp=False,
                is_side_grasp=False
            )
            
        elif action_type == 'retract_gripper':
            return ExecutableAction(
                action_type='retract_gripper',
                position=np.zeros(3),
                orientation=np.zeros(3),
                pixel_position=object_pixel_pose,
                parameters={'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium'))},
                is_top_down_grasp=False,
                is_side_grasp=False
            )
            
        return None

    def _calculate_approach_orientation(self, 
                                    target_position, 
                                    pivot_position, 
                                    is_parallel, 
                                    action_type):
        """
        Calculate approach orientation based on target and pivot positions
        
        Args:
            target_position: 3D position of target point
            pivot_position: 3D position of pivot point (or None)
            is_parallel: Whether approach should be parallel to surface
            action_type: 'push' or 'pull'
            
        Returns:
            3D orientation vector [roll, pitch, yaw]
        """
        # Default orientation (approaching directly from front)
        orientation = np.zeros(3)
        
        if pivot_position is not None:
            # Calculate vector from pivot to target
            pivot_to_target = target_position - pivot_position
            pivot_to_target = pivot_to_target / np.linalg.norm(pivot_to_target)
            
            # Calculate approach vector
            if is_parallel:
                # For parallel approach, we want to move perpendicular to pivot_to_target vector
                if abs(pivot_to_target[2]) < 0.9:  # If not predominantly vertical
                    approach_vector = np.cross(pivot_to_target, np.array([0, 0, 1]))
                else:
                    approach_vector = np.cross(pivot_to_target, np.array([1, 0, 0]))
                
                # Normalize
                approach_vector = approach_vector / np.linalg.norm(approach_vector)
            else:
                # For perpendicular approach, we want to move along the pivot_to_target vector
                approach_vector = pivot_to_target
                
            # Adjust for push vs. pull
            if action_type == 'pull':
                approach_vector = -approach_vector
                
            # Convert approach vector to Euler angles
            # This is a simplified conversion - in production would use proper rotation matrices
            pitch = np.arctan2(approach_vector[2], np.sqrt(approach_vector[0]**2 + approach_vector[1]**2))
            yaw = np.arctan2(approach_vector[1], approach_vector[0])
            orientation = np.array([0, pitch, yaw])
        else:
            # No pivot - use simpler orientation
            # For push, approach from negative Z (top-down)
            # For pull, approach from positive Z (bottom-up)
            if action_type == 'push':
                orientation = np.array([0, 0, 0])  # Default orientation
            else:  # pull
                orientation = np.array([np.pi, 0, 0])  # Rotated 180 degrees around X
                
        return orientation

    def _convert_parsed_primitive_to_executable(self,
                                            parsed_primitive: ParsedPrimitive,
                                            object_pose: np.ndarray,
                                            object_pixel_pose: Tuple[int, int],
                                            interaction_points: Dict[str, np.ndarray],
                                            interaction_pixel_points: Dict[str, Tuple[int, int]],
                                            skill_parameters: Dict[str, Any]) -> Optional[ExecutableAction]:
        """Convert parsed primitive to executable action"""
        action_type = parsed_primitive.action_type
        parameters = parsed_primitive.parameters
        
        # Handle push/pull actions
        if action_type in ['push', 'pull']:
            return self._create_push_pull_action(
            action_type=action_type,
            surface_keywords=parameters.get('surface_keywords', []),
            is_parallel_surface=parameters.get('is_parallel_surface', False),
            is_button=parameters.get('is_button', False),
            has_pivot=parameters.get('has_pivot', False),
            pivot_point=parameters.get('pivot_point', None),
            distance=parameters.get('distance', 0.1),
            object_pose=object_pose,
            object_pixel_pose=object_pixel_pose,
            interaction_points=interaction_points,
            interaction_pixel_points=interaction_pixel_points,
            skill_parameters=skill_parameters,
        )
            
        # Handle gripper movement with potential grasp
        elif action_type == 'move_gripper_to_pose':
            return self._create_gripper_pose_action(
                keywords=parameters.get('keywords', []),
                is_side_grasp=parameters.get('is_side_grasp', False),
                is_top_down_grasp=parameters.get('is_top_down_grasp', False),
                interaction_points=interaction_points,
                interaction_pixel_points=interaction_pixel_points,
                skill_parameters=skill_parameters
            )
            
        # Handle simple gripper actions
        elif action_type == 'close_gripper':
            return ExecutableAction(
                action_type='close_gripper',
                position=np.zeros(3),
                orientation=np.zeros(3),
                is_top_down_grasp=False,
                is_side_grasp=False,
                pixel_position=object_pixel_pose,  # Use object center for visualization
                parameters={'force': self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium'))}
            )
            
        elif action_type == 'open_gripper':
            return ExecutableAction(
                action_type='release',
                position=np.zeros(3),
                orientation=np.zeros(3),
                is_top_down_grasp=False,
                is_side_grasp=False,
                pixel_position=object_pixel_pose,  # Use object center for visualization
                parameters={}
            )
            
        elif action_type == 'retract_gripper':
            return ExecutableAction(
                action_type='retract_gripper',
                position=np.zeros(3),
                orientation=np.zeros(3),
                is_top_down_grasp=False,
                is_side_grasp=False,
                pixel_position=object_pixel_pose,  # Use object center for visualization
                parameters={'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium'))}
            )
                
        return None

    def _create_push_pull_action(self,
                       action_type: str,
                       surface_keywords: List[str],
                       is_parallel_surface: bool,
                       is_button: bool,
                       has_pivot: bool,
                       pivot_point: Optional[str],
                       distance: float,
                       object_pose: np.ndarray,
                       object_pixel_pose: Tuple[int, int],
                       interaction_points: Dict[str, np.ndarray],
                       interaction_pixel_points: Dict[str, Tuple[int, int]],
                       skill_parameters: Dict[str, Any]) -> ExecutableAction:
        """Create push or pull action with specified parameters"""
        # Calculate force based on distance and skill parameters
        force_magnitude = self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium'))
        
        # Find target interaction point for the surface
        target_point = np.zeros(3)
        target_pixel = object_pixel_pose  # Default to object center
        
        for keyword in surface_keywords:
            if keyword in interaction_points:
                target_point = interaction_points[keyword]
                target_pixel = interaction_pixel_points[keyword]
                break
        
        # Determine the orientation based on surface relationship
        orientation = np.zeros(3)
        
        # If we have surface information, use it to determine approach angle
        if len(surface_keywords) > 0:
            # Calculate surface normal vector (simplified)
            # In a real system, this would be based on point cloud analysis or depth image
            surface_normal = np.array([0, 0, 1])  # Default points towards camera
            
            # Adjust based on object pose
            if object_pose is not None and len(object_pose) >= 6:
                # Apply rotation from object pose to surface normal
                # This is a simplified approach; in reality would use rotation matrices
                roll, pitch, yaw = object_pose[3:6]
                # Simple rotation transformation - in production would use proper 3D rotation
                surface_normal = np.array([
                    np.sin(pitch) * np.cos(yaw),
                    np.sin(pitch) * np.sin(yaw),
                    np.cos(pitch)
                ])
            
            # Adjust based on is_parallel_surface
            if is_parallel_surface:
                # For parallel approach, we want to move along the surface
                # This requires computing a vector parallel to the surface
                # For simplicity, we'll use a perpendicular vector to the normal
                if abs(surface_normal[2]) < 0.9:  # If normal is not too vertical
                    parallel_vector = np.cross(surface_normal, np.array([0, 0, 1]))
                else:
                    parallel_vector = np.cross(surface_normal, np.array([1, 0, 0]))
                    
                # Normalize
                parallel_vector = parallel_vector / np.linalg.norm(parallel_vector)
                
                # Set approach direction to be along this parallel vector
                approach_vector = parallel_vector
            else:
                # For perpendicular approach, just use the surface normal
                approach_vector = surface_normal
            
            # Adjust for push vs pull direction
            if action_type == 'pull':
                approach_vector = -approach_vector
                
            # Adjust for top vs bottom surface
            if is_button:
                # For bottom surface, we typically approach from below
                approach_vector[2] = -abs(approach_vector[2])
            else:
                # For top surface, we typically approach from above
                approach_vector[2] = abs(approach_vector[2])
            
            # Calculate orientation from approach vector
            # This is a simplified conversion from vector to Euler angles
            # In production, would use proper vector to rotation matrix conversion
            pitch = np.arctan2(approach_vector[2], np.sqrt(approach_vector[0]**2 + approach_vector[1]**2))
            yaw = np.arctan2(approach_vector[1], approach_vector[0])
            orientation = np.array([0, pitch, yaw])  # Roll is set to 0 for simplicity
        
        # Handle pivot points if specified
        pivot_position = None
        if has_pivot and pivot_point:
            # Try to find the pivot point location
            if pivot_point in interaction_points:
                pivot_position = interaction_points[pivot_point]
            else:
                # If pivot point isn't explicitly provided, make an estimate
                # This is a simplified approach - in production would use more sophisticated estimation
                if pivot_point.lower() in ['top', 'upper']:
                    # Estimate pivot at top of object
                    pivot_position = object_pose[:3] + np.array([0, 0, 0.1])  # 10cm above object center
                elif pivot_point.lower() in ['bottom', 'lower']:
                    # Estimate pivot at bottom of object
                    pivot_position = object_pose[:3] + np.array([0, 0, -0.1])  # 10cm below object center
                elif pivot_point.lower() in ['left']:
                    # Estimate pivot at left of object
                    pivot_position = object_pose[:3] + np.array([-0.1, 0, 0])  # 10cm left of object center
                elif pivot_point.lower() in ['right']:
                    # Estimate pivot at right of object
                    pivot_position = object_pose[:3] + np.array([0.1, 0, 0])  # 10cm right of object center
        
        # Create the parameters for the executable action
        action_params = {
            'distance': distance,
            'force_magnitude': force_magnitude,
            'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium')),
            'precision': skill_parameters.get('precision_required', 'medium'),
            'is_parallel_surface': is_parallel_surface,
            'is_button': is_button,
            'has_pivot': has_pivot,
            'surface_keywords': surface_keywords
        }
        
        # Add pivot information if available
        if pivot_position is not None:
            action_params['pivot_position'] = pivot_position
        
        return ExecutableAction(
            action_type=action_type,
            position=target_point,  # Use the identified surface point
            orientation=orientation,  # Orientation calculated based on surface relationship
            pixel_position=target_pixel,  # Store pixel location for visualization
            is_top_down_grasp=False,  # Not a grasp action
            is_side_grasp=False,  # Not a grasp action
            parameters=action_params
        )

    def _create_gripper_pose_action(self,
                              keywords: List[str],
                              is_side_grasp: bool,
                              is_top_down_grasp: bool,
                              interaction_points: Dict[str, np.ndarray],
                              interaction_pixel_points: Dict[str, Tuple[int, int]],
                              skill_parameters: Dict[str, Any]) -> Optional[ExecutableAction]:
        """Create gripper pose action with grasp parameter"""
        target_point = None
        target_pixel = None
        
        for keyword in keywords:
            if keyword in interaction_points:
                target_point = interaction_points[keyword]
                target_pixel = interaction_pixel_points[keyword]
                break
                
        if target_point is None:
            return None
            
        # If this is a grasp action, add additional parameters
        parameters = {
            'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium')),
            'precision': skill_parameters.get('precision_required', 'medium'),
            'keywords': keywords,
            'is_side_grasp': is_side_grasp,
            'is_top_down_grasp': is_top_down_grasp
        }
        
        if is_top_down_grasp or is_side_grasp:
            parameters.update({
                'force': self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium')),
                'grasp_planning_required': True
            })
        
        return ExecutableAction(
            action_type='move_gripper_to_pose',
            position=target_point,
            orientation=np.zeros(3),  # Will be determined by pose/grasp planner
            pixel_position=target_pixel,
            is_top_down_grasp=False,
            is_side_grasp=False,
            parameters=parameters
        )

    
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