from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import cv2
import asyncio
from pathlib import Path
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
            target_object,
            image,
            depth_image
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
        
        # Get original mask shape for later resizing
        original_mask_shape = object_info.mask.shape
        
        # Resize object image to match FastSAM input size
        max_image_size = self.perception_system.segmenter.config.max_image_size
        resized_image = cv2.resize(
            object_info.image,
            (max_image_size, max_image_size),
            interpolation=cv2.INTER_LINEAR
        )
        
        # For each keyword, detect the relevant part
        for keyword in keywords:
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
            
            for prompt in prompts:
                labeled_masks, metadata = self.perception_system.segmenter.process_image(
                    resized_image,
                    query=prompt
                )
                
                for mask_id, meta in metadata.items():
                    component_mask = labeled_masks == mask_id
                    
                    # Resize component mask to match original image dimensions
                    component_mask = cv2.resize(
                        component_mask.astype(np.uint8),
                        (original_mask_shape[1], original_mask_shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    if np.logical_and(component_mask, object_info.mask).any():
                        score = meta.get('clip_score', 0)
                        if score > best_score:
                            best_score = score
                            # Get centroid of the intersection
                            intersection = np.logical_and(component_mask, object_info.mask)
                            y_coords, x_coords = np.where(intersection)
                            if len(x_coords) > 0 and len(y_coords) > 0:
                                pixel_pos = np.array((
                                    int(np.mean(x_coords)),
                                    int(np.mean(y_coords))
                                ))
                                # Convert to 3D using perception system
                                point_3d = self.perception_system.get_3d_point(pixel_pos)
                                if point_3d is not None:
                                    best_point = point_3d
                                    best_pixel = pixel_pos

            # Use lower threshold for keywords that might be harder to detect
            min_score_threshold = 0.3
            if best_point is not None and best_score > min_score_threshold:
                interaction_points[keyword] = best_point
                pixel_points[keyword] = best_pixel
            else:
                # Fallback: use object center or appropriate region
                if np.any(object_info.mask):
                    y_coords, x_coords = np.where(object_info.mask)
                    pixel_pos = np.array((
                        int(np.mean(x_coords)),
                        int(np.mean(y_coords))
                    ))
                    # For specific keywords, adjust the point
                    if keyword.lower() in ['handle', 'knob', 'grip']:
                        # For handles, try the top region
                        pixel_pos = np.array((
                            pixel_pos[0],
                            int(np.min(y_coords) + (np.max(y_coords) - np.min(y_coords)) * 0.2)
                        ))
                    
                    point_3d = self.perception_system.get_3d_point(pixel_pos)
                    if point_3d is not None:
                        interaction_points[keyword] = point_3d
                        pixel_points[keyword] = pixel_pos
                        print(f"Using fallback position for {keyword}")

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