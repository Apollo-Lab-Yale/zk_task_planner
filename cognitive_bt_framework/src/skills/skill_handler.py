from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import cv2
import asyncio
from pathlib import Path
import traceback
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig, ObjectInfo
from cognitive_bt_framework.src.skills.primitive_parser import PrimitiveParser, ParsedPrimitive
from cognitive_bt_framework.src.skills.skill_generator import SkillGenerator
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI

@dataclass
class ExecutableAction:
    """Data class for an executable primitive action with concrete parameters"""
    action_type: str  # e.g., 'push', 'pull', etc.
    position: np.ndarray  # 3D position [x, y, z]
    orientation: np.ndarray  # 3D orientation [roll, pitch, yaw]
    pixel_position: Optional[Tuple[int, int]]  # 2D pixel coordinates [x, y]
    parameters: Dict[str, Any]  # Additional parameters like force magnitude, speed, etc.
    
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
        
    async def instantiate_skill(
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
            
        # Get or generate skill using SkillGenerator
        skill = await self.skill_generator.find_similar_skill(
            image=object_info.image,
            mask=object_info.mask,
            abstract_action=abstract_action,
            target_object=target_object
        )
        
        if skill is None:
            skill = await self.skill_generator.generate_skill(
                image=object_info.image,
                mask=object_info.mask,
                abstract_action=abstract_action,
                target_object=target_object
            )
            
        if skill is None:
            print(f"Failed to generate skill for {abstract_action} on {target_object}")
            return None
            
        # Instantiate the skill with concrete parameters
        return self._instantiate_skill(skill, object_info)

    def _identify_interaction_points(self, 
                             object_info: ObjectInfo,
                             keywords: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
        """
        Identify 3D and 2D interaction points on the object for given keywords
        
        Args:
            object_info: Object information from perception system
            keywords: List of keywords to identify
            
        Returns:
            Tuple containing:
            - Dictionary mapping keywords to 3D positions
            - Dictionary mapping keywords to 2D pixel positions
        """
        interaction_points = {}
        pixel_points = {}
        
        # Check if we have the object mask
        if object_info.mask is None:
            print(f"Warning: Object has no mask, using bounding box for interaction point detection")
            # Create a simple mask from the bounding box
            x, y, w, h = object_info.bbox
            temp_mask = np.zeros(object_info.image.shape[:2], dtype=bool)
            temp_mask[y:y+h, x:x+w] = True
            object_info.mask = temp_mask
        
        # Get object bounding box
        x, y, w, h = object_info.bbox
        
        # For each keyword, detect the relevant part
        for keyword in keywords:
            # Try to detect the component within the object
            component = self.perception_system.detect_object_component(
                object_info=object_info,
                component_name=keyword,
                conf_threshold=0.3,  # Use lower threshold for keywords
                depth_image=object_info.depth_image if hasattr(object_info, 'depth_image') else None
            )
            
            if component is not None and component.confidence > 0.3:
                # We found a good component match
                if component.pose is not None and component.pixel_pose is not None:
                    # Use the component's 3D pose and pixel position
                    interaction_points[keyword] = component.pose[:3]  # Use just the position part
                    pixel_points[keyword] = component.pixel_pose
                    print(f"Found component '{keyword}' with confidence {component.confidence:.2f}")
                else:
                    # Component without pose, use centroid of mask
                    if component.mask is not None and np.any(component.mask):
                        y_coords, x_coords = np.where(component.mask)
                        pixel_pos = (int(np.mean(x_coords)), int(np.mean(y_coords)))
                        
                        # Convert to 3D point if we have depth image
                        if hasattr(object_info, 'depth_image') and object_info.depth_image is not None:
                            point_3d, px_pt = self.perception_system._estimate_object_pose(
                                object_info.mask,
                                depth_image=object_info.depth_image
                            )
                            interaction_points[keyword] = point_3d
                        else:
                            # Without depth, use object's pose with adjusted x,y
                            point_3d = object_info.pose[:3] if object_info.pose is not None else np.zeros(3)
                            interaction_points[keyword] = point_3d
                            
                        pixel_points[keyword] = pixel_pos
                        print(f"Using mask centroid for component '{keyword}'")
            else:
                # Try alternative approach by using specific prompts
                # Extract region of interest around the object
                # Expand the region slightly to ensure we capture the full object
                x_expand = max(0, x - int(w * 0.1))
                y_expand = max(0, y - int(h * 0.1))
                w_expand = min(object_info.image.shape[1] - x_expand, int(w * 1.2))
                h_expand = min(object_info.image.shape[0] - y_expand, int(h * 1.2))
                
                # Extract expanded ROI
                roi = object_info.image[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand]
                
                # Try multiple prompts for robustness
                prompts = [
                    f"a {keyword} of the object",
                    f"the {keyword}",
                    f"part of the object that is the {keyword}",
                    f"region of the object that looks like a {keyword}"
                ]
                
                best_score = 0
                best_point = None
                best_pixel = None
                
                # Only process with segmenter if it's available
                if self.perception_system.segmenter is not None:
                    for prompt in prompts:
                        try:
                            labeled_masks, metadata = self.perception_system.segmenter.process_image(
                                roi, 
                                query=prompt
                            )
                            
                            for mask_id, meta in metadata.items():
                                # Create component mask in ROI coordinates
                                component_roi_mask = labeled_masks == mask_id
                                
                                # Create full image mask
                                component_mask = np.zeros(object_info.image.shape[:2], dtype=bool)
                                component_mask[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand] = component_roi_mask
                                
                                # Check if component intersects with object mask
                                if np.logical_and(component_mask, object_info.mask).any():
                                    score = meta.get('clip_score', 0)
                                    if score > best_score:
                                        best_score = score
                                        # Get centroid of the intersection
                                        intersection = np.logical_and(component_mask, object_info.mask)
                                        mask_y_coords, mask_x_coords = np.where(intersection)
                                        if len(mask_x_coords) > 0 and len(mask_y_coords) > 0:
                                            pixel_pos = (
                                                int(np.mean(mask_x_coords)),
                                                int(np.mean(mask_y_coords))
                                            )
                                            
                                            # Convert to 3D point if we have depth image
                                            if hasattr(object_info, 'depth_image') and object_info.depth_image is not None:
                                                point_3d, px_pt = self.perception_system._estimate_object_pose(
                                                    object_info.mask,
                                                    depth_image=object_info.depth_image
                                                )
                                            else:
                                                # Without depth, get pose from mask
                                                pose, pixel_pose = self.perception_system._estimate_object_pose(
                                                    intersection,
                                                    np.ones(object_info.image.shape[:2], dtype=np.float32) * 0.5  # Default depth
                                                )
                                                point_3d = pose[:3]
                                                pixel_pos = pixel_pose
                                                
                                            best_point = point_3d
                                            best_pixel = pixel_pos
                        except Exception as e:
                            print(f"Error processing prompt '{prompt}': {e}")
                            continue
                
                # Use lower threshold for keywords that might be harder to detect
                min_score_threshold = 0.3
                if best_point is not None and best_score > min_score_threshold:
                    interaction_points[keyword] = best_point
                    pixel_points[keyword] = best_pixel
                    print(f"Found '{keyword}' with score {best_score:.2f} using segmentation")
                else:
                    # Fallback: use object center or appropriate region
                    if np.any(object_info.mask):
                        mask_y_coords, mask_x_coords = np.where(object_info.mask)
                        pixel_pos = (
                            int(np.mean(mask_x_coords)),
                            int(np.mean(mask_y_coords))
                        )
                        
                        # For specific keywords, adjust the point based on semantic understanding
                        if keyword.lower() in ['handle', 'knob', 'grip', 'top']:
                            # For handles and tops, try the top region
                            pixel_pos = (
                                pixel_pos[0],
                                int(np.min(mask_y_coords) + (np.max(mask_y_coords) - np.min(mask_y_coords)) * 0.2)
                            )
                        elif keyword.lower() in ['bottom', 'base']:
                            # For bottoms and bases, try the bottom region
                            pixel_pos = (
                                pixel_pos[0],
                                int(np.min(mask_y_coords) + (np.max(mask_y_coords) - np.min(mask_y_coords)) * 0.8)
                            )
                        elif keyword.lower() in ['left', 'left_side']:
                            # For left side
                            pixel_pos = (
                                int(np.min(mask_x_coords) + (np.max(mask_x_coords) - np.min(mask_x_coords)) * 0.2),
                                pixel_pos[1]
                            )
                        elif keyword.lower() in ['right', 'right_side']:
                            # For right side
                            pixel_pos = (
                                int(np.min(mask_x_coords) + (np.max(mask_x_coords) - np.min(mask_x_coords)) * 0.8),
                                pixel_pos[1]
                            )
                        
                        # Convert to 3D point if we have depth image
                        if hasattr(object_info, 'depth_image') and object_info.depth_image is not None:
                            point_3d, px_pt = self.perception_system._estimate_object_pose(
                                object_info.mask,
                                depth_image=object_info.depth_image
                            )
                        else:
                            # Without depth, use object's pose
                            point_3d = object_info.pose[:3] if object_info.pose is not None else np.zeros(3)
                        
                        interaction_points[keyword] = point_3d
                        pixel_points[keyword] = pixel_pos
                        print(f"Using fallback position for '{keyword}'")

        return interaction_points, pixel_points
        
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
        
        # Get all unique keywords from parsed primitives
        all_keywords = set()
        for primitive in parsed_primitives:
            if 'keywords' in primitive.parameters:
                all_keywords.update(primitive.parameters['keywords'])
                
        # Identify interaction points for all keywords at once
        interaction_points, pixel_points = self._identify_interaction_points(
            object_info, 
            list(all_keywords)
        )
        
        # Convert each parsed primitive to executable action
        action_sequence = []
        for primitive in parsed_primitives:
            print(primitive)
            executable_action = self._convert_parsed_primitive_to_executable(
                primitive,
                object_pose,
                object_pixel_pose,
                interaction_points,
                pixel_points,
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
            interaction_points
        )
        
        return InstantiatedSkill(
            skill_name=skill.name,
            target_object=skill.target_object,
            action_sequence=action_sequence,
            execution_parameters=execution_parameters,
            object_pixel_pose=object_pixel_pose
        )

    # [Previous helper methods remain unchanged]
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
                distance=parameters['distance'],
                object_pose=object_pose,
                object_pixel_pose=object_pixel_pose,
                skill_parameters=skill_parameters
            )
            
        # Handle gripper movement with potential grasp
        elif action_type == 'moveGripperToPose':
            return self._create_gripper_pose_action(
                keywords=parameters.get('keywords', []),
                is_grasp=parameters.get('is_grasp', False),
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
                pixel_position=object_pixel_pose,  # Use object center for visualization
                parameters={'force': self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium'))}
            )
            
        elif action_type == 'release':
            return ExecutableAction(
                action_type='release',
                position=np.zeros(3),
                orientation=np.zeros(3),
                pixel_position=object_pixel_pose,  # Use object center for visualization
                parameters={}
            )
            
        elif action_type == 'retractGripper':
            return ExecutableAction(
                action_type='retractGripper',
                position=np.zeros(3),
                orientation=np.zeros(3),
                pixel_position=object_pixel_pose,  # Use object center for visualization
                parameters={'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium'))}
            )
                
        return None

    def _create_push_pull_action(self,
                           action_type: str,
                           distance: float,
                           object_pose: np.ndarray,
                           object_pixel_pose: Tuple[int, int],
                           skill_parameters: Dict[str, Any]) -> ExecutableAction:
        """Create push or pull action with specified distance"""
        # Calculate force based on distance and skill parameters
        force_magnitude = self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium'))
        
        return ExecutableAction(
            action_type=action_type,
            position=np.zeros(3),  # Will use current position
            orientation=np.zeros(3),  # Will use current orientation
            pixel_position=object_pixel_pose,  # Store pixel location for visualization
            parameters={
                'distance': distance,
                'force_magnitude': force_magnitude,
                'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium')),
                'precision': skill_parameters.get('precision_required', 'medium')
            }
        )

    def _create_gripper_pose_action(self,
                              keywords: List[str],
                              is_grasp: bool,
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
            'is_grasp': is_grasp
        }
        
        if is_grasp:
            parameters.update({
                'force': self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium')),
                'grasp_planning_required': True
            })
        
        return ExecutableAction(
            action_type='moveGripperToPose',
            position=target_point,
            orientation=np.zeros(3),  # Will be determined by pose/grasp planner
            pixel_position=target_pixel,
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
                                    interaction_points: Dict[str, np.ndarray]) -> Dict[str, Any]:
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
        execution_params['interaction_points'] = interaction_points
        
        # Add any safety parameters
        execution_params['force_monitoring'] = True
        execution_params['collision_detection'] = True
        execution_params['velocity_scaling'] = 0.8  # 80% of maximum speed for safety
        
        return execution_params

    def _calculate_approach_orientation(self, direction: str, object_pose: np.ndarray) -> np.ndarray:
        """Calculate approach orientation based on direction and object pose"""
        # Basic orientation calculation
        orientation = np.zeros(3)
        if direction == 'up':
            orientation[0] = -np.pi/2
        elif direction == 'down':
            orientation[0] = np.pi/2
        elif direction == 'left':
            orientation[1] = -np.pi/2
        elif direction == 'right':
            orientation[1] = np.pi/2
        elif direction == 'push':
            orientation[0] = 0
        elif direction == 'pull':
            orientation[0] = np.pi
            
        # Adjust based on object pose
        orientation += object_pose[3:]
        return orientation

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
            'moveGripperToPose': (255, 0, 0),  # Red
            'close_gripper': (0, 0, 255),      # Blue
            'release': (255, 165, 0),          # Orange
            'retractGripper': (128, 0, 128)    # Purple
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
                
                # Draw connection line to object center for moveGripperToPose
                if action.action_type == 'moveGripperToPose':
                    cv2.line(vis_image, object_center, pos_2d, color, 1, cv2.LINE_AA)
                
                # Draw action type and parameters
                label = f"{i+1}. {action.action_type}"
                
                # Add specific parameters based on action type
                if action.action_type == 'moveGripperToPose':
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
                    
                    # Calculate push/pull direction relative to object center
                    angle = np.arctan2(pos_2d[1] - object_center[1], 
                                    pos_2d[0] - object_center[0])
                    
                    end_point = (
                        int(pos_2d[0] + (arrow_length * np.cos(angle) if action.action_type == 'push' 
                                    else -arrow_length * np.cos(angle))),
                        int(pos_2d[1] + (arrow_length * np.sin(angle) if action.action_type == 'push'
                                    else -arrow_length * np.sin(angle)))
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
            
            else:
                # For actions without position (close_gripper, release, retractGripper)
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