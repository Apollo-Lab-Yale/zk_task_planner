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
    action_type: str  # e.g., 'apply_force', 'apply_torque', etc.
    position: np.ndarray  # 3D position [x, y, z]
    orientation: np.ndarray  # 3D orientation [roll, pitch, yaw]
    parameters: Dict[str, Any]  # Additional parameters like force magnitude, speed, etc.
    
@dataclass
class InstantiatedSkill:
    """Data class for a skill instantiated for a specific object"""
    skill_name: str
    target_object: str
    action_sequence: List[ExecutableAction]
    execution_parameters: Dict[str, Any]
    
    
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
                                keywords: List[str]) -> Dict[str, np.ndarray]:
        """
        Identify 3D interaction points on the object for given keywords
        
        Args:
            object_info: Object information from perception system
            keywords: List of keywords to identify
            
        Returns:
            Dictionary mapping keywords to 3D positions
        """
        interaction_points = {}
        
        # Get original mask shape for later resizing
        original_mask_shape = object_info.mask.shape
        
        # Resize object image to match FastSAM input size
        max_image_size = self.perception_system.segmenter.config.max_image_size
        resized_image = cv2.resize(
            object_info.image,
            (max_image_size, max_image_size),
            interpolation=cv2.INTER_LINEAR
        )
        
        # For each keyword, detect the relevant part using perception system
        for keyword in keywords:
            # Process image for specific part detection
            labeled_masks, metadata = self.perception_system.segmenter.process_image(
                resized_image,
                query=f"a {keyword} of the object"
            )
            
            # Find the mask with highest CLIP score
            best_score = 0
            best_point = None
            
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
                        # Get centroid of the component
                        y_coords, x_coords = np.where(component_mask)
                        center_2d = np.array([
                            np.mean(x_coords),
                            np.mean(y_coords)
                        ])
                        # Convert to 3D using perception system
                        best_point = self.perception_system.get_3d_point(center_2d)
                        
            if best_point is not None:
                interaction_points[keyword] = best_point
                
        return interaction_points
        
    def _instantiate_skill(self, skill, object_info: ObjectInfo) -> Optional[InstantiatedSkill]:
        """Convert abstract skill to concrete executable actions"""
        try:
            # First validate and parse all primitives
            parsed_primitives = self.primitive_parser.validate_primitive_sequence(
                skill.primitive_sequence
            )
        except ValueError as e:
            print(f"Failed to parse skill primitives: {e}")
            return None
            
        # Get 3D information about the object - use pose from ObjectInfo if available
        object_pose = object_info.pose if object_info.pose is not None else np.zeros(6)
        
        # Get all unique keywords from parsed primitives
        all_keywords = set()
        for primitive in parsed_primitives:
            if 'keywords' in primitive.parameters:
                all_keywords.update(primitive.parameters['keywords'])
                
        # Identify interaction points for all keywords at once
        interaction_points = self._identify_interaction_points(object_info, list(all_keywords))
        
        # Convert each parsed primitive to executable action
        action_sequence = []
        for primitive in parsed_primitives:
            print(primitive)
            executable_action = self._convert_parsed_primitive_to_executable(
                primitive,
                object_pose,
                interaction_points,
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
            execution_parameters=execution_parameters
        )

    # [Previous helper methods remain unchanged]
    def _convert_parsed_primitive_to_executable(self,
                                              parsed_primitive: ParsedPrimitive,
                                              object_pose: np.ndarray,
                                              interaction_points: Dict[str, np.ndarray],
                                              skill_parameters: Dict[str, Any]) -> Optional[ExecutableAction]:
        """Convert parsed primitive to executable action"""
        action_type = parsed_primitive.action_type
        parameters = parsed_primitive.parameters
        if 'keywords' not in parameters:
            parameters['keywords'] = []
            
        if action_type == 'apply_force':
            return self._create_force_action(
                direction=parameters['direction'],
                keywords=parameters['keywords'],
                object_pose=object_pose,
                interaction_points=interaction_points,
                skill_parameters=skill_parameters
            )
        elif action_type == 'apply_torque':
            return self._create_torque_action(
                axis=parameters['axis'],
                keywords=parameters['keywords'],
                object_pose=object_pose,
                interaction_points=interaction_points,
                skill_parameters=skill_parameters
            )
        elif action_type == 'go_to_obj':
            return self._create_move_action(
                keywords=parameters['keywords'],
                interaction_points=interaction_points,
                skill_parameters=skill_parameters
            )
        elif action_type == 'close_gripper':
            return self._create_grasp_action(
                keywords=parameters['keywords'],
                interaction_points=interaction_points,
                skill_parameters=skill_parameters
            )
        elif action_type == 'moveGripperToPose':
            return self._create_gripper_pose_action(
                keywords=parameters['keywords'],
                interaction_points=interaction_points,
                skill_parameters=skill_parameters
            )
        elif action_type == 'release':
            return ExecutableAction(
                action_type='release',
                position=np.zeros(3),
                orientation=np.zeros(3),
                parameters={}
            )
        elif action_type == 'retractGripper':
            return ExecutableAction(
                action_type='retractGripper',
                position=np.zeros(3),
                orientation=np.zeros(3),
                parameters={}
            )
            
        return None

    def _create_force_action(self,
                           direction: str,
                           keywords: List[str],
                           object_pose: np.ndarray,
                           interaction_points: Dict[str, np.ndarray],
                           skill_parameters: Dict[str, Any]) -> Optional[ExecutableAction]:
        """Create force action with parsed parameters"""
        target_point = None
        for keyword in keywords:
            if keyword in interaction_points:
                target_point = interaction_points[keyword]
                break
                
        if target_point is None:
            return None
            
        force_magnitude = self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium'))
        approach_orientation = self._calculate_approach_orientation(direction, object_pose)
        
        return ExecutableAction(
            action_type='apply_force',
            position=target_point,
            orientation=approach_orientation,
            parameters={
                'force_magnitude': force_magnitude,
                'direction': direction,
                'keywords': keywords
            }
        )

    def _create_move_action(self,
                          keywords: List[str],
                          interaction_points: Dict[str, np.ndarray],
                          skill_parameters: Dict[str, Any]) -> Optional[ExecutableAction]:
        """Create move action with parsed parameters"""
        target_point = None
        for keyword in keywords:
            if keyword in interaction_points:
                target_point = interaction_points[keyword]
                break
                
        if target_point is None:
            return None
            
        speed = self._get_speed_value(skill_parameters.get('speed_requirement', 'medium'))
        
        return ExecutableAction(
            action_type='go_to_obj',
            position=target_point,
            orientation=np.zeros(3),  # Will be determined by subsequent grasp or force action
            parameters={
                'speed': speed,
                'precision': skill_parameters.get('precision_required', 'medium'),
                'keywords': keywords
            }
        )

    def _create_grasp_action(self,
                           keywords: List[str],
                           interaction_points: Dict[str, np.ndarray],
                           skill_parameters: Dict[str, Any]) -> Optional[ExecutableAction]:
        """Create grasp action with parsed parameters"""
        # target_point = None
        # for keyword in keywords:
        #     if keyword in interaction_points:
        #         target_point = interaction_points[keyword]
        #         break
                
        # if target_point is None:
        #     return None
            
        return ExecutableAction(
            action_type='grasp',
            position=target_point,
            orientation=np.zeros(3),  # Will be determined by grasp planner
            parameters={
                'force': self._get_force_magnitude(skill_parameters.get('force_threshold', 'medium')),
                'precision': skill_parameters.get('precision_required', 'medium'),
                'keywords': keywords
            }
        )

    def _create_gripper_pose_action(self,
                                  keywords: List[str],
                                  interaction_points: Dict[str, np.ndarray],
                                  skill_parameters: Dict[str, Any]) -> Optional[ExecutableAction]:
        """Create gripper pose action with parsed parameters"""
        target_point = None
        for keyword in keywords:
            if keyword in interaction_points:
                target_point = interaction_points[keyword]
                break
                
        if target_point is None:
            return None
            
        return ExecutableAction(
            action_type='moveGripperToPose',
            position=target_point,
            orientation=np.zeros(3),  # Will be determined by pose planner
            parameters={
                'speed': self._get_speed_value(skill_parameters.get('speed_requirement', 'medium')),
                'precision': skill_parameters.get('precision_required', 'medium'),
                'keywords': keywords
            }
        )

    def _create_torque_action(self,
                            axis: str,
                            keywords: List[str],
                            object_pose: np.ndarray,
                            interaction_points: Dict[str, np.ndarray],
                            skill_parameters: Dict[str, Any]) -> Optional[ExecutableAction]:
        """Create torque action with parsed parameters"""
        target_point = None
        for keyword in keywords:
            if keyword in interaction_points:
                target_point = interaction_points[keyword]
                break
                
        if target_point is None:
            return None
            
        torque_magnitude = self._get_torque_magnitude(skill_parameters.get('force_threshold', 'medium'))
        approach_orientation = self._calculate_approach_orientation_for_torque(axis, object_pose)
        
        return ExecutableAction(
            action_type='apply_torque',
            position=target_point,
            orientation=approach_orientation,
            parameters={
                'torque_magnitude': torque_magnitude,
                'axis': axis,
                'keywords': keywords
            }
        )

    def _get_force_magnitude(self, force_threshold: str) -> float:
        """Convert force threshold to concrete value"""
        force_values = {
            'low': 5.0,
            'medium': 10.0,
            'high': 20.0
        }
        return force_values.get(force_threshold.lower(), 10.0)

    def _get_torque_magnitude(self, force_threshold: str) -> float:
        """Convert force threshold to torque magnitude"""
        torque_values = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        }
        return torque_values.get(force_threshold.lower(), 1.0)

    def _get_speed_value(self, speed_requirement: str) -> float:
        """Convert speed requirement to concrete value"""
        speed_values = {
            'slow': 0.1,
            'medium': 0.3,
            'fast': 0.5
        }
        return speed_values.get(speed_requirement.lower(), 0.3)

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

    def _generate_execution_parameters(self,
                                    skill_parameters: Dict[str, Any],
                                    object_pose: np.ndarray,
                                    interaction_points: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate concrete execution parameters from abstract skill parameters
        
        Args:
            skill_parameters: Parameters from abstract skill
            object_pose: 6D pose of the object
            interaction_points: Dictionary of interaction points
            
        Returns:
            Dictionary of concrete execution parameters
        """
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
        
    def visualize_skill(self, skill: InstantiatedSkill, image: np.ndarray) -> np.ndarray:
        """
        Visualize skill execution on image
        
        Args:
            skill: InstantiatedSkill to visualize
            image: RGB image to visualize on
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Draw each action in sequence
        for i, action in enumerate(skill.action_sequence):
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            
            # Project 3D position to 2D
            pos_2d = (
                int(action.position[0] * 640),
                int(action.position[1] * 640)
            )
            
            # Draw action point
            cv2.circle(vis_image, pos_2d, 5, color, -1)
            
            # Draw action type text
            cv2.putText(
                vis_image,
                f"{i+1}. {action.action_type}",
                (pos_2d[0] + 10, pos_2d[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            
            # Draw arrows for force/torque actions
            if action.action_type in ['apply_force', 'apply_torque']:
                direction = action.parameters.get('direction', '')
                if direction:
                    end_point = list(pos_2d)
                    if direction == 'up':
                        end_point[1] -= 20
                    elif direction == 'down':
                        end_point[1] += 20
                    elif direction == 'left':
                        end_point[0] -= 20
                    elif direction == 'right':
                        end_point[0] += 20
                    elif direction == 'push':
                        end_point[1] += 20
                    elif direction == 'pull':
                        end_point[1] -= 20
                        
                    cv2.arrowedLine(
                        vis_image,
                        pos_2d,
                        tuple(end_point),
                        color,
                        2,
                        tipLength=0.3
                    )
        
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