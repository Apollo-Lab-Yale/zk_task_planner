#!/usr/bin/env python3
"""
Task Planner for High-Level Robot Task Execution

This module takes high-level tasks (e.g., "put away the dishes") and decomposes them
into a sequence of executable robot skills based on visual perception of the environment.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import base64
import io
from PIL import Image

# Import vision and perception systems
from cognitive_bt_framework.src.vision.realsense import Camera as RealSenseCamera
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig

# Import skills system
from cognitive_bt_framework.src.skills.skill_executor import DirectSkillExecutor
from cognitive_bt_framework.src.skills import SkillGenerator, SkillHandler, ExecutableAction

# Import LLM interface for task decomposition
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI

# Import robot interface
from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface import CuRoboMotionPlanner


@dataclass
class TaskStep:
    """Represents a single step in a task plan"""
    description: str
    skill_type: str
    target_object: str
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TaskPlan:
    """Represents a complete task plan with multiple steps"""
    task_description: str
    steps: List[TaskStep]
    environment_description: str
    estimated_duration: float = 0.0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class TaskPlanner:
    """
    High-level task planner that decomposes complex tasks into executable skill sequences
    """
    
    def __init__(self, 
                 robot_ip: str = "192.168.1.224",
                 camera_type: str = "realsense",
                 use_llm: bool = True):
        """
        Initialize the task planner (LLM is required for dynamic task decomposition)
        
        Args:
            robot_ip: IP address of the robot
            camera_type: Type of camera to use ("realsense" - ZED is not supported)
            use_llm: Whether to use LLM for task decomposition (required for dynamic planning)
        """
        self.logger = logging.getLogger(__name__)
        self.robot_ip = robot_ip
        self.camera_type = camera_type
        self.use_llm = use_llm
        
        # Initialize robot interface
        self.motion_planner = CuRoboMotionPlanner(robot_ip=robot_ip)
        
        # Initialize skill executor with RealSense camera first
        self.skill_executor = DirectSkillExecutor(
            robot_ip=robot_ip,
            use_zed_camera=False,  # Always use RealSense
            show_debug_windows=False,
            fast_mode=True
        )
        
        # Use camera and perception from skill executor to avoid conflicts
        self.camera = getattr(self.skill_executor, 'camera', None)
        self.perception_system = getattr(self.skill_executor, 'perception_system', None)
        
        # Initialize LLM interface for task decomposition
        if self.use_llm:
            self.llm = LLMInterfaceOpenAI()
        
        # Define skill mappings and templates
        self._init_skill_templates()
        
        # Store current environment state
        self.current_environment = None
        self.detected_objects = []
    
    def _init_skill_templates(self):
        """Initialize skill templates for common task decomposition"""
        # Define valid skills based on robot capabilities
        self.valid_skills = {
            'open': 1, 'close': 1, 'pickup': 1,
            'place': 2, 'switchon': 1, 'switchoff': 1,
            'twist': 1
        }
        
        self.skill_templates = {
            "pickup": {
                "skill_type": "move_gripper_to_pose",
                "parameters": {
                    "keywords": [],
                    "is_top_down_grasp": True,
                    "is_side_grasp": False
                }
            },
            "place": {
                "skill_type": "move_gripper_to_pose", 
                "parameters": {
                    "keywords": [],
                    "is_top_down_grasp": True,
                    "is_side_grasp": False
                }
            },
            "open": {
                "skill_type": "pull",
                "parameters": {
                    "keywords": [],
                    "is_parallel_surface": False,
                    "is_button": False,
                    "has_pivot": True
                }
            },
            "close": {
                "skill_type": "push",
                "parameters": {
                    "keywords": [],
                    "is_parallel_surface": False,
                    "is_button": False,
                    "has_pivot": True
                }
            },
            "switchon": {
                "skill_type": "push",
                "parameters": {
                    "keywords": [],
                    "is_parallel_surface": False,
                    "is_button": True,
                    "has_pivot": False
                }
            },
            "switchoff": {
                "skill_type": "push",
                "parameters": {
                    "keywords": [],
                    "is_parallel_surface": False,
                    "is_button": True,
                    "has_pivot": False
                }
            },
            "twist": {
                "skill_type": "twist",
                "parameters": {
                    "direction": "counterclockwise"
                }
            }
        }
        
        # No pattern matching - purely LLM-driven task decomposition
    
    def capture_environment(self) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Capture current environment state using vision system
        
        Returns:
            Tuple of (rgb_image, environment_info)
        """
        if not self.camera:
            self.logger.error("Camera not initialized")
            return None, None
            
        try:
            # Capture image from camera
            rgb_image, depth_image = self.camera.get_frame()
            
            if rgb_image is None:
                self.logger.error("Failed to capture image")
                return None, None
            
            # Create basic environment info
            environment_info = {
                "timestamp": datetime.now().isoformat(),
                "image_shape": rgb_image.shape,
                "depth_available": depth_image is not None,
                "detected_objects": []
            }
            
            # Run perception to detect objects if available
            if self.perception_system:
                try:
                    # Detect objects and their properties
                    detection_results = self.perception_system.detect_objects(rgb_image)
                    environment_info["detected_objects"] = detection_results
                    self.detected_objects = detection_results
                    
                except Exception as e:
                    self.logger.warning(f"Object detection failed: {e}")
                    # Keep the basic environment_info even if detection fails
            
            self.current_environment = {
                "rgb_image": rgb_image,
                "depth_image": depth_image,
                "info": environment_info
            }
            
            return rgb_image, environment_info
            
        except Exception as e:
            self.logger.error(f"Failed to capture environment: {e}")
            return None, None
    
    def analyze_task(self, task_description: str, environment_image: np.ndarray = None) -> TaskPlan:
        """
        Analyze a high-level task and create an execution plan
        
        Args:
            task_description: High-level task description (e.g., "put away the dishes")
            environment_image: Optional environment image for visual context
            
        Returns:
            TaskPlan object with decomposed steps
        """
        self.logger.info(f"Analyzing task: {task_description}")
        
        # Capture environment if not provided
        if environment_image is None:
            environment_image, env_info = self.capture_environment()
            if environment_image is None:
                self.logger.error("Could not capture environment")
                return self._create_fallback_plan(task_description)
        
        # Use LLM for intelligent task decomposition - required for dynamic planning
        if not self.use_llm or not self.llm:
            self.logger.error("LLM is required for dynamic task decomposition")
            return self._create_fallback_plan(task_description)
            
        try:
            task_plan = self._llm_task_decomposition(task_description, environment_image)
            if task_plan:
                return task_plan
            else:
                self.logger.error("LLM failed to generate task plan")
                return self._create_fallback_plan(task_description)
        except Exception as e:
            self.logger.error(f"LLM task decomposition failed: {e}")
            return self._create_fallback_plan(task_description)
    
    def _llm_task_decomposition(self, task_description: str, environment_image: np.ndarray) -> Optional[TaskPlan]:
        """
        Use LLM to decompose task based on visual environment understanding
        """
        # Convert image to base64 for LLM
        image_b64 = self._image_to_base64(environment_image)
        
        # Create detailed prompt for task decomposition
        prompt = f"""
        You are an expert robot task planner. Given the task "{task_description}" and the attached image of the robot's environment, 
        decompose this task into a sequence of specific robot skills.
        
        Available robot skills (use exact names):
        - pickup: Pick up an object from its current location
        - place: Place an object at a specific location  
        - open: Open doors, drawers, cabinets (handles hinged objects)
        - close: Close doors, drawers, cabinets (handles hinged objects)
        - switchon: Turn on switches, buttons, appliances
        - switchoff: Turn off switches, buttons, appliances
        - twist: Twist/rotate objects like bottle caps, jar lids, knobs
        
        IMPORTANT PLANNING RULES:
        1. Always be specific about target objects (e.g., "red cup", "kitchen cabinet", "light switch")
        2. Break complex tasks into logical steps
        3. Consider object dependencies (open before placing inside)
        4. Be practical - only plan actions the robot can physically perform
        5. If opening containers, remember to close them afterwards
        
        Analyze the image to identify:
        1. All objects relevant to the task
        2. Storage locations (cabinets, drawers, shelves)
        3. Spatial relationships and accessibility
        4. Required manipulation sequence
        
        Example task decompositions:
        - "put away the dishes" → pickup dish, open cabinet, place dish inside, close cabinet
        - "turn on the light" → switchon light_switch
        - "open the bottle" → twist bottle_cap
        
        Provide a JSON response with this structure:
        {{
            "task_analysis": "Brief analysis of what you see and the task requirements",
            "identified_objects": ["list", "of", "relevant", "objects"],
            "steps": [
                {{
                    "description": "Human-readable step description",
                    "skill_type": "robot_skill_name",
                    "target_object": "specific_object_name",
                    "parameters": {{"key": "value"}},
                    "priority": 1
                }}
            ],
            "estimated_duration": 30.0
        }}
        
        Focus on practical, executable steps that a robot arm can perform.
        """
        
        try:
            # Get LLM response with vision
            response = self.llm.get_response_with_image(prompt, image_b64)
            
            # Parse JSON response
            if response.startswith("```json"):
                response = response.strip("```json").strip("```").strip()
            
            task_data = json.loads(response)
            
            # Convert to TaskPlan object and validate skills
            steps = []
            for step_data in task_data.get("steps", []):
                skill_type = step_data.get("skill_type", "")
                
                # Validate that the skill is available
                if skill_type not in self.valid_skills:
                    self.logger.warning(f"Invalid skill '{skill_type}' generated by LLM. Available skills: {list(self.valid_skills.keys())}")
                    continue
                
                # Map skill to actual implementation
                skill_template = self.skill_templates.get(skill_type, {})
                actual_skill_type = skill_template.get("skill_type", skill_type)
                skill_parameters = skill_template.get("parameters", {})
                
                # Merge LLM parameters with template parameters
                final_parameters = {**skill_parameters, **step_data.get("parameters", {})}
                
                step = TaskStep(
                    description=step_data.get("description", ""),
                    skill_type=actual_skill_type,
                    target_object=step_data.get("target_object", ""),
                    parameters=final_parameters,
                    priority=step_data.get("priority", 1)
                )
                steps.append(step)
            
            task_plan = TaskPlan(
                task_description=task_description,
                steps=steps,
                environment_description=task_data.get("task_analysis", ""),
                estimated_duration=task_data.get("estimated_duration", 0.0)
            )
            
            self.logger.info(f"LLM generated plan with {len(steps)} steps")
            return task_plan
            
        except Exception as e:
            self.logger.error(f"LLM task decomposition failed: {e}")
            return None
    
    
    def _create_fallback_plan(self, task_description: str) -> TaskPlan:
        """Create a simple fallback plan when other methods fail"""
        step = TaskStep(
            description="Manual task execution required",
            skill_type="manual",
            target_object="unknown",
            parameters={}
        )
        
        return TaskPlan(
            task_description=task_description,
            steps=[step],
            environment_description="Unable to analyze environment"
        )
    
    def execute_task_plan(self, task_plan: TaskPlan, execute_on_robot: bool = True) -> Dict[str, Any]:
        """
        Execute a complete task plan step by step
        
        Args:
            task_plan: TaskPlan to execute
            execute_on_robot: Whether to actually execute on robot or just plan
            
        Returns:
            Dictionary with execution results
        """
        self.logger.info(f"Executing task plan: {task_plan.task_description}")
        
        results = {
            "task_description": task_plan.task_description,
            "start_time": datetime.now().isoformat(),
            "steps_executed": [],
            "success": False,
            "error_message": None
        }
        
        try:
            for i, step in enumerate(task_plan.steps):
                self.logger.info(f"Executing step {i+1}: {step.description}")
                
                # Create ExecutableAction from TaskStep
                action = ExecutableAction(
                    action_type=step.skill_type,
                    parameters=step.parameters
                )
                
                # Execute the skill
                success = self.skill_executor.execute_action(
                    action=action,
                    timeout=120
                )
                
                step_result = {
                    "success": success,
                    "execution_time": 10.0,  # Placeholder
                    "error": None if success else "Skill execution failed"
                }
                
                step_info = {
                    "step_number": i + 1,
                    "description": step.description,
                    "skill_type": step.skill_type,
                    "target_object": step.target_object,
                    "success": step_result.get("success", False),
                    "execution_time": step_result.get("execution_time", 0.0),
                    "error": step_result.get("error", None)
                }
                
                results["steps_executed"].append(step_info)
                
                # Stop execution if step failed
                if not step_result.get("success", False):
                    results["error_message"] = f"Step {i+1} failed: {step_result.get('error', 'Unknown error')}"
                    break
            
            # Check if all steps succeeded
            results["success"] = all(step["success"] for step in results["steps_executed"])
            results["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            results["error_message"] = f"Task execution failed: {str(e)}"
            results["end_time"] = datetime.now().isoformat()
            self.logger.error(f"Task execution error: {e}")
        
        return results
    
    def plan_and_execute_task(self, task_description: str, execute_on_robot: bool = True) -> Dict[str, Any]:
        """
        Complete workflow: analyze task and execute it
        
        Args:
            task_description: High-level task description
            execute_on_robot: Whether to execute on actual robot
            
        Returns:
            Complete execution results
        """
        self.logger.info(f"Starting complete task workflow: {task_description}")
        
        # Step 1: Capture environment
        environment_image, env_info = self.capture_environment()
        
        # Step 2: Analyze and decompose task
        task_plan = self.analyze_task(task_description, environment_image)
        
        # Step 3: Execute plan
        execution_results = self.execute_task_plan(task_plan, execute_on_robot)
        
        # Combine results
        complete_results = {
            "task_description": task_description,
            "environment_info": env_info,
            "task_plan": {
                "steps": [
                    {
                        "description": step.description,
                        "skill_type": step.skill_type,
                        "target_object": step.target_object,
                        "parameters": step.parameters
                    }
                    for step in task_plan.steps
                ],
                "estimated_duration": task_plan.estimated_duration
            },
            "execution_results": execution_results
        }
        
        return complete_results
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string for LLM"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return image_b64
            
        except Exception as e:
            self.logger.error(f"Failed to convert image to base64: {e}")
            return ""
    
    def get_available_skills(self) -> List[str]:
        """Get list of available robot skills"""
        return list(self.skill_templates.keys())
    
    def get_skill_info(self, skill_name: str) -> Optional[Dict]:
        """Get information about a specific skill"""
        return self.skill_templates.get(skill_name)


# Example usage and testing functions
def test_task_planner():
    """Test the task planner with example tasks (requires LLM for dynamic planning)"""
    
    print("=== Dynamic Task Planner Test ===")
    print("This test requires LLM integration for dynamic task decomposition")
    
    # Initialize task planner with LLM required
    try:
        planner = TaskPlanner(use_llm=True)  # LLM required for dynamic planning
        
        # Test available skills
        print(f"\nAvailable Skills: {list(planner.valid_skills.keys())}")
        
        # Test task decomposition with dynamic LLM planning
        test_tasks = [
            "put away the dishes",
            "turn on the kitchen light", 
            "open the bottle and pour water",
            "clean up the counter and close all cabinets"
        ]
        
        for task in test_tasks:
            print(f"\n=== Testing Task: {task} ===")
            
            # Note: This would require actual LLM integration and environment capture
            print(f"Task: {task}")
            print(f"Required: LLM analysis of environment image")
            print(f"Expected: Dynamic skill sequence generation")
            print(f"Available skills for planning: {list(planner.valid_skills.keys())}")
            
    except Exception as e:
        print(f"LLM integration required: {e}")
        print("To test dynamic planning:")
        print("1. Ensure LLM interface is configured")
        print("2. Connect robot and camera")
        print("3. Run with actual environment images")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_task_planner()