#!/usr/bin/env python3
"""
Task Planner for High-Level Robot Task Execution

This module takes high-level tasks (e.g., "put away the dishes") and decomposes them
into a sequence of executable robot skills based on visual perception of the environment.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import base64
import io
from PIL import Image
import time

# Import skills system
from cognitive_bt_framework.src.skills.skill_executor import DirectSkillExecutor

# Import LLM interface for task decomposition
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI

# Import data recording system
from cognitive_bt_framework.src.data_recorder import DataRecorder


class TaskPlanner:
    """
    High-level task planner that decomposes complex tasks into executable skill sequences
    """
    
    def __init__(self, 
                 robot_ip: str = "192.168.1.224",
                 use_llm: bool = True,
                 enable_data_recording: bool = False,
                 recording_timestep: float = 0.1,
                 collect_openvla_data: bool = False):
        """
        Initialize the task planner with proper tool integration
        
        Args:
            robot_ip: IP address of the robot
            use_llm: Whether to use LLM for task decomposition (required for dynamic planning)
            enable_data_recording: Whether to enable comprehensive data recording (default: False for development)
            recording_timestep: Time interval for motion data recording during execution (seconds)
            collect_openvla_data: Whether to collect OpenVLA-compatible demonstration data during execution
        """
        self.logger = logging.getLogger(__name__)
        self.robot_ip = robot_ip
        self.use_llm = use_llm
        self.enable_data_recording = enable_data_recording
        self.recording_timestep = recording_timestep
        self.collect_openvla_data = collect_openvla_data
        
        # Initialize skill executor (it handles everything: camera, perception, skill generation, execution)
        self.skill_executor = DirectSkillExecutor(
            robot_ip=robot_ip,
            camera_params={'width': 640, 'height': 480, 'fps': 30},
            use_zed_camera=False,  # Use RealSense
            show_debug_windows=False,
            calibrate_transform=False,
            fast_mode=True
        )
        
        # Get camera reference for environment capture (for LLM context)
        self.camera = self.skill_executor.camera
        
        # Initialize LLM interface for task decomposition only
        if self.use_llm:
            self.llm = LLMInterfaceOpenAI()
        
        # Initialize data recording system
        if self.enable_data_recording:
            self.data_recorder = DataRecorder(recording_timestep=self.recording_timestep)
            self.logger.info(f"Data recording system enabled with {self.recording_timestep}s timestep")
        else:
            self.data_recorder = None
            self.logger.info("Data recording system disabled (development mode)")
        
        # Define valid skills that can be requested from skill executor
        self.valid_skills = {
            'detect_object': 1, 'open': 1, 'close': 1, 'pickup': 1,
            'place': 2, 'switchon': 1, 'switchoff': 1
        }
    
    
    def capture_environment_image(self) -> Optional[np.ndarray]:
        """
        Capture environment image for LLM context
        
        Returns:
            RGB image for LLM analysis
        """
        try:
            # Use skill executor's camera to capture frames
            if hasattr(self.camera, 'get_frames'):
                frames = self.camera.get_frames()
                if not frames:
                    self.logger.error("Failed to capture frames")
                    return None
                rgb_image, _ = frames
                return rgb_image
            else:
                self.logger.error("Camera does not have get_frames method")
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to capture environment image: {e}")
            return None
    
    def analyze_task(self, task_description: str, record_data: bool = None) -> List[str]:
        """
        Analyze a high-level task and decompose into skill sequence
        
        Args:
            task_description: High-level task description (e.g., "move the bag and open the bottle")
            record_data: Whether to record data for this analysis
            
        Returns:
            List of skill commands to execute
        """
        self.logger.info(f"Analyzing task: {task_description}")
        
        # Determine if we should record data
        should_record = record_data if record_data is not None else self.enable_data_recording
        
        # Start data recording if enabled
        if should_record and self.data_recorder:
            self.data_recorder.start_recording_session(
                natural_language_task=task_description,
                task_name=None,  # Auto-generated from natural language
                robot_ip=self.robot_ip,
                camera_type="realsense",  # Could be made configurable
                execution_mode="real"  # Could be made configurable
            )
        
        # Capture environment image for LLM context
        environment_image = self.capture_environment_image()
        if environment_image is None:
            self.logger.error("Could not capture environment image")
            if record_data and self.data_recorder:
                self.data_recorder.record_planning_complete(["manual execution required"])
            return ["manual execution required"]
        
        # Record environment image
        if should_record and self.data_recorder:
            self.data_recorder.record_environment_image(environment_image)
        
        # Use LLM for task decomposition
        if not self.use_llm or not self.llm:
            self.logger.error("LLM is required for dynamic task decomposition")
            if should_record and self.data_recorder:
                self.data_recorder.record_planning_complete(["manual execution required"])
            return ["manual execution required"]
            
        try:
            skill_sequence = self._llm_task_decomposition(task_description, environment_image)
            result = skill_sequence if skill_sequence else ["manual execution required"]
            
            # Record planning completion
            if should_record and self.data_recorder:
                self.data_recorder.record_planning_complete(result)
            
            return result
        except Exception as e:
            self.logger.error(f"LLM task decomposition failed: {e}")
            if should_record and self.data_recorder:
                self.data_recorder.record_planning_complete(["manual execution required"])
            return ["manual execution required"]
    
    def _llm_task_decomposition(self, task_description: str, environment_image: np.ndarray) -> Optional[List[str]]:
        """
        Use LLM to decompose task into simple skill command sequence
        """
        # Convert image to base64 for LLM
        image_b64 = self._image_to_base64(environment_image)
        
        # Create focused prompt for skill sequence generation
        prompt = f"""
        You are a robot task planner. Given the task "{task_description}" and the attached environment image, 
        generate a sequence of robot skill commands.
        
        IMPORTANT: Use the provided image to understand the current environment and available objects. 
        Base your task decomposition on what you can see in the image - identify objects, their locations, 
        and their current states to determine the optimal sequence of actions.
        
        Available skills:
        - detect_object <object>: Detect and locate an object in the environment
        - pickup <object>: Pick up an object
        - place <object>,<location>: Place object at location
        - open <object>: Open doors, drawers, containers
        - close <object>: Close doors, drawers, containers
        - switchon <object>: Turn on switches/buttons
        - switchoff <object>: Turn off switches/buttons
        
        OBJECT DETECTION STRATEGY:
        - Use detect_object for ALL objects that need to be manipulated, UNLESS they are clearly obscured or contained (inside closed drawers, cabinets, etc.)
        - Objects should be detected when they are visible and accessible in the current environment
        - If an object becomes visible after opening a container, detect it immediately after the container is opened
        - verify that all objects are in the reference image provided
        - separate multi word objects with spaces ie "top shelf" etc
        - IT is impossible to detect an object while holding it
        
        PLACEMENT OBJECT STRATEGY:
        - verify that the selected object is both in the image and clear of clutter for placement
        - choose the best surface for a successful placement, maximize clear area and ease of detection for models like CLIP based off the limited information in the image
        
        ROBOT LIMITATIONS:
        - the robot has a single arm so it cannot be holding anything when attempting to manipulate an object
        - DO NOT ATTEMPT TO PICK UP AN OBJECT BEFORE MANIPULATING IT THIS WILL FAIL
        - Do not attempt to detect an object you are holding because this will fail
        
        Try to provide the simplest plan possible.
        Analyze the image carefully and provide ONLY a simple list of commands, one per line:
        
        Example for "move the <obstructing object> and open the <object>":
        detect_object <obstructing object>
        detect_object <clear surface in image>
        pickup <obstructing object>
        place <obstructing object>,<clear surface in image>
        detect_object <object>
        open <object>
        
        Example for "open the <object>":
        detect_object <object>
        open <object>
        
        EXAMPLE object with lid "open the <object> and put the lid in the <placement area>":
        detect_object <object>
        detect_object <place area>
        open <object>
        place <object lid>,<place area>
        
        in the above example the robot opens the object and is still holding the lid
        
        FAILURE EXAMPLE "open the <object>":
        detect_object <object>
        pickup <object>
        open <object>
        
        FAILURE EXAMPLE object with lid "open the <object> and put the lid in the <placement area>":
        detect_object <object>
        detect_object <place area>
        open <object>
        detect_object <object lid>
        place <object lid>,<place area>
        
        The above fails because the object will be in the robots gripper, and there is no second 
        arm to open the bottle so the manipulation fails
        
        Generate commands for: {task_description}
        """
        
        try:
            # Get LLM response with vision
            response = self.llm.get_response_with_image(prompt, image_b64)
            
            # Record raw LLM response if data recording is enabled
            if self.data_recorder:
                if not hasattr(self.data_recorder, 'current_record') or not self.data_recorder.current_record:
                    pass  # No active session
                else:
                    self.data_recorder.current_record.llm_responses['task_decomposition'] = response or "No response"
            
            if not response:
                return None
                
            # Parse response into list of commands
            commands = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                    
                # Extract skill name to validate
                skill_name = line.split()[0] if line.split() else ""
                if skill_name in self.valid_skills:
                    commands.append(line)
                else:
                    self.logger.warning(f"Invalid skill in command: {line}")
            
            self.logger.info(f"LLM generated {len(commands)} skill commands")
            return commands if commands else None
            
        except Exception as e:
            self.logger.error(f"LLM task decomposition failed: {e}")
            return None
    
    def execute_task(self, task_description: str, execute_on_robot: bool = True, record_data: bool = None) -> Dict[str, Any]:
        """
        Execute a complete task by decomposing it and using skill executor's full pipeline
        
        Args:
            task_description: High-level task description
            execute_on_robot: Whether to actually execute on robot or just simulate
            
        Returns:
            Dictionary with execution results
        """
        self.logger.info(f"Executing task: {task_description}")
        
        results = {
            "task_description": task_description,
            "start_time": datetime.now().isoformat(),
            "skill_commands": [],
            "skills_generated": [],
            "execution_results": [],
            "success": False,
            "error_message": None
        }
        
        try:
            # Determine if we should record data
            should_record = record_data if record_data is not None else self.enable_data_recording
            
            # Step 1: Decompose task into skill commands using LLM + environment image
            skill_commands = self.analyze_task(task_description, record_data=should_record)
            results["skill_commands"] = skill_commands
            
            if skill_commands == ["manual execution required"]:
                results["error_message"] = "Task decomposition failed - manual execution required"
                results["end_time"] = datetime.now().isoformat()
                
                # Record failure and finalize if recording
                if should_record and self.data_recorder:
                    self.data_recorder.record_execution_complete(
                        success=False, 
                        failure_messages=["Task decomposition failed - manual execution required"]
                    )
                    self.data_recorder.finalize_session(prompt_for_save=True)
                
                return results
            
            # Record start of inference phase
            if should_record and self.data_recorder:
                self.data_recorder.record_inference_start()
            
            # Step 2: Execute all skills as a sequence using skill executor's optimized pipeline
            if not execute_on_robot:
                # Simulate execution
                success, error_msg = True, "Simulation mode - would generate and execute skill sequence"
                
                # Record simulated skill generation
                simulated_skills = []
                for command in skill_commands:
                    results["skills_generated"].append(f"Would generate skill for: {command}")
                    simulated_skills.append({
                        "command": command,
                        "skill_name": command.split()[0] if command.split() else "unknown",
                        "parameters": " ".join(command.split()[1:]) if len(command.split()) > 1 else "",
                        "status": "simulated"
                    })
                    step_result = {
                        "command": command,
                        "skill_generated": True,
                        "success": True,
                        "error": None
                    }
                    results["execution_results"].append(step_result)
                
                # Record inference completion for simulation
                if should_record and self.data_recorder:
                    self.data_recorder.record_inference_complete(
                        skills_generated=simulated_skills,
                        llm_responses={"execution_mode": "simulation"}
                    )
            else:
                # Parse all skill commands into (skill_name, parameters) tuples
                skill_sequence = []
                for command in skill_commands:
                    parts = command.split()
                    skill_name = parts[0]
                    parameters = " ".join(parts[1:]) if len(parts) > 1 else ""
                    skill_sequence.append((skill_name, parameters))
                
                self.logger.info(f"Executing skill sequence with {len(skill_sequence)} skills")
                print(f"Executing skill sequence: {skill_sequence}")
                
                # Record skills generated during inference phase
                generated_skills = []
                for skill_name, parameters in skill_sequence:
                    generated_skills.append({
                        "command": f"{skill_name} {parameters}".strip(),
                        "skill_name": skill_name,
                        "parameters": parameters,
                        "status": "ready_for_execution"
                    })
                
                # Record inference completion
                if should_record and self.data_recorder:
                    self.data_recorder.record_inference_complete(
                        skills_generated=generated_skills,
                        llm_responses={"execution_mode": "real_robot"}
                    )
                    
                    # Record start of execution phase (but don't start recording yet)
                    self.data_recorder.record_execution_start()
                    
                    # Set up motion recording interfaces (but don't start recording yet)
                    robot_interface = None
                    if hasattr(self.skill_executor, 'motion_planner'):
                        robot_interface = self.skill_executor.motion_planner
                    
                    camera_interface = self.camera
                    
                    self.data_recorder.set_motion_interfaces(
                        robot_interface=robot_interface,
                        camera_interface=camera_interface
                    )
                
                # Start motion recording right before robot execution
                if should_record and self.data_recorder:
                    print("Starting motion data recording before robot execution...")
                    self.data_recorder.start_motion_recording()
                    
                    # Start OpenVLA data collection if enabled
                    if self.collect_openvla_data:
                        self._start_openvla_collection(task_description)
                
                # Execute entire skill sequence using optimized pipeline:
                # - Extract unique objects and detect them all upfront
                # - Generate all skills using cached detections
                # - Execute all generated skills in sequence
                success, error_msg = self.skill_executor.execute_skill_sequence(skill_sequence, task_context=task_description)
                
                # Collect OpenVLA timesteps during execution if enabled
                if should_record and self.collect_openvla_data:
                    # Collect a few timesteps during execution for demonstration data
                    for i in range(3):  # Collect 3 timesteps as examples
                        time.sleep(0.5)  # Small delay between collections
                        self.collect_openvla_timestep()
                
                # Collect any additional data from skill executor
                if should_record and self.data_recorder and hasattr(self.skill_executor, 'get_execution_data'):
                    execution_data = self.skill_executor.get_execution_data()
                    if 'points_of_interest' in execution_data:
                        self.data_recorder.record_points_of_interest(execution_data['points_of_interest'])
                    if 'surface_images' in execution_data:
                        self.data_recorder.record_surface_images(execution_data['surface_images'])
                    if 'skill_generation_files' in execution_data:
                        self.data_recorder.record_skill_generation_files(execution_data['skill_generation_files'])
                
                # Record results for all commands
                for i, command in enumerate(skill_commands):
                    if success:
                        results["skills_generated"].append(f"Generated and executed skill for: {command}")
                    else:
                        results["skills_generated"].append(f"Failed in skill sequence for: {command}")
                    
                    step_result = {
                        "command": command,
                        "skill_generated": success,
                        "success": success,
                        "error": error_msg if not success else None
                    }
                    results["execution_results"].append(step_result)
                
                # Stop motion recording if it was started
                if should_record and self.data_recorder:
                    self.data_recorder.stop_motion_recording()
                    
                    # Stop OpenVLA data collection if it was started
                    if self.collect_openvla_data:
                        self._stop_openvla_collection()
                
                # Record execution completion
                failure_messages = [error_msg] if not success and error_msg else []
                if should_record and self.data_recorder:
                    self.data_recorder.record_execution_complete(
                        success=success,
                        failure_messages=failure_messages,
                        robot_errors=failure_messages  # For now, treat all errors as robot errors
                    )
                
                if not success:
                    results["error_message"] = f"Skill sequence execution failed: {error_msg}"
            
            # Check if all steps succeeded
            results["success"] = all(step["success"] for step in results["execution_results"])
            results["end_time"] = datetime.now().isoformat()
            
            # Add summary
            total_commands = len(skill_commands)
            successful_commands = sum(1 for step in results["execution_results"] if step["success"])
            
            self.logger.info(f"Task execution summary: {successful_commands}/{total_commands} commands succeeded")
            
            # Finalize data recording session
            if should_record and self.data_recorder:
                try:
                    # Export OpenVLA dataset if demonstration data was collected (before finalization)
                    if self.collect_openvla_data and len(self.data_recorder.current_record.demonstration_data) > 0:
                        try:
                            openvla_path = self.export_openvla_dataset()
                            if openvla_path:
                                results["openvla_dataset_path"] = openvla_path
                        except Exception as e:
                            self.logger.warning(f"Failed to export OpenVLA dataset: {e}")
                    
                    session_path = self.data_recorder.finalize_session(prompt_for_save=True)
                    results["session_data_path"] = session_path
                    
                except Exception as recording_error:
                    self.logger.warning(f"Failed to finalize recording session: {recording_error}")
            
        except Exception as e:
            error_message = f"Task execution failed: {str(e)}"
            results["error_message"] = error_message
            results["end_time"] = datetime.now().isoformat()
            self.logger.error(f"Task execution error: {e}")
            
            # Record execution failure and finalize
            if should_record and self.data_recorder:
                try:
                    # Stop motion recording if it was started
                    self.data_recorder.stop_motion_recording()
                    
                    # Stop OpenVLA data collection if it was started
                    if self.collect_openvla_data:
                        self._stop_openvla_collection()
                    
                    self.data_recorder.record_execution_complete(
                        success=False,
                        failure_messages=[error_message],
                        robot_errors=[str(e)]
                    )
                    session_path = self.data_recorder.finalize_session(prompt_for_save=True)
                    results["session_data_path"] = session_path
                except Exception as recording_error:
                    self.logger.warning(f"Failed to record execution failure: {recording_error}")
        
        return results
    
    def plan_and_execute_task(self, task_description: str, execute_on_robot: bool = True, record_data: bool = None) -> Dict[str, Any]:
        """
        Complete workflow: analyze task and execute it
        
        Args:
            task_description: High-level task description  
            execute_on_robot: Whether to execute on actual robot
            record_data: Whether to record data (None uses class default)
            
        Returns:
            Complete execution results
        """
        # Just use the execute_task method (it handles both planning and execution)
        return self.execute_task(task_description, execute_on_robot, record_data)
    
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
    
    def set_data_recording(self, enabled: bool):
        """
        Enable or disable data recording at runtime
        
        Args:
            enabled: True to enable recording, False to disable
        """
        old_state = self.enable_data_recording
        self.enable_data_recording = enabled
        
        if enabled and not old_state:
            # Enabling recording
            if self.data_recorder is None:
                self.data_recorder = DataRecorder()
            self.logger.info("Data recording enabled")
        elif not enabled and old_state:
            # Disabling recording
            self.logger.info("Data recording disabled")
        
    def is_data_recording_enabled(self) -> bool:
        """
        Check if data recording is currently enabled
        
        Returns:
            True if data recording is enabled, False otherwise
        """
        return self.enable_data_recording
    
    def set_recording_timestep(self, timestep: float):
        """
        Set the timestep for motion data recording
        
        Args:
            timestep: Time interval between recordings in seconds
        """
        self.recording_timestep = timestep
        if self.data_recorder:
            self.data_recorder.recording_timestep = timestep
            self.logger.info(f"Recording timestep set to {timestep}s")
    
    def get_recording_timestep(self) -> float:
        """
        Get the current recording timestep
        
        Returns:
            Current recording timestep in seconds
        """
        return self.recording_timestep
    
    def pause_data_recording(self):
        """
        Pause data recording (called when robot stops for planning/perception)
        
        This should be called by the skill execution system when the robot
        stops moving for replanning, perception updates, or skill generation.
        """
        if self.data_recorder:
            self.data_recorder.pause_motion_recording()
            self.logger.debug("Data recording paused - robot stationary for planning")
    
    def resume_data_recording(self):
        """
        Resume data recording (called when robot starts moving again)
        
        This should be called by the skill execution system when the robot
        resumes motion after a planning/perception phase.
        """
        if self.data_recorder:
            self.data_recorder.resume_motion_recording()
            self.logger.debug("Data recording resumed - robot motion starting")
    
    def _start_openvla_collection(self, task_description: str):
        """
        Start OpenVLA demonstration data collection during execution
        
        Args:
            task_description: Natural language task description
        """
        if not self.data_recorder:
            return
        
        # OpenVLA collection is handled by the data recorder thread
        # We just need to track some state here
        self.logger.info("OpenVLA data collection started")
    
    def _stop_openvla_collection(self):
        """Stop OpenVLA demonstration data collection"""
        if not self.data_recorder:
            return
        
        self.logger.info("OpenVLA data collection stopped")
    
    def collect_openvla_timestep(self) -> bool:
        """
        Manually collect a single OpenVLA timestep (for use during skill execution)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.data_recorder or not self.data_recorder.current_record:
            return False
        
        try:
            # Get current robot state
            robot_interface = None
            if hasattr(self.skill_executor, 'motion_planner'):
                robot_interface = self.skill_executor.motion_planner
            
            joint_positions = None
            end_effector_pose = None
            gripper_state = None
            
            if robot_interface and hasattr(robot_interface, 'get_robot_joint_state'):
                joint_state = robot_interface.get_robot_joint_state()
                if joint_state:
                    joint_positions = joint_state[:7] if len(joint_state) >= 7 else joint_state
                    gripper_state = joint_state[6] if len(joint_state) > 6 else 0.0
                    
                    # Get end effector pose (placeholder - would need actual forward kinematics)
                    end_effector_pose = [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]  # [x,y,z,rx,ry,rz]
            
            # Get camera images
            wrist_image = None
            external_image = None
            
            if self.camera and hasattr(self.camera, 'get_frames'):
                frames = self.camera.get_frames()
                if frames:
                    wrist_image, _ = frames
            
            # For now, use zero delta pose (would need to track previous poses for real deltas)
            delta_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            gripper_action = gripper_state if gripper_state is not None else 0.0
            
            # Add demonstration timestep
            success = self.data_recorder.add_demonstration_timestep(
                wrist_image=wrist_image,
                external_image=external_image,
                joint_positions=joint_positions,
                end_effector_pose=end_effector_pose,
                gripper_state=gripper_state,
                delta_pose=delta_pose,
                gripper_action=gripper_action,
                language_instruction=self.data_recorder.current_record.natural_language_task
            )
            
            if success:
                self.logger.debug("OpenVLA timestep collected")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to collect OpenVLA timestep: {e}")
            return False
    
    def export_openvla_dataset(self, output_path: str = None) -> Optional[str]:
        """
        Export collected OpenVLA demonstration data
        
        Args:
            output_path: Output file path (default: auto-generated)
            
        Returns:
            Path to exported dataset file, or None if failed
        """
        if not self.data_recorder:
            self.logger.error("No data recorder available")
            return None
        
        try:
            return self.data_recorder.export_openvla_dataset(output_path)
        except Exception as e:
            self.logger.error(f"Failed to export OpenVLA dataset: {e}")
            return None
    
    def get_available_skills(self) -> List[str]:
        """Get list of available robot skills"""
        return list(self.valid_skills.keys())
    
    def get_skill_info(self, skill_name: str) -> Optional[Dict]:
        """Get information about a specific skill"""
        return {"name": skill_name} if skill_name in self.valid_skills else None
    
    def shutdown(self):
        """Properly shutdown the task planner and all components"""
        try:
            # The skill executor handles camera cleanup
            if hasattr(self, 'skill_executor') and self.skill_executor:
                # Check if camera exists and is running before stopping
                if (hasattr(self.skill_executor, 'camera') and 
                    self.skill_executor.camera and 
                    hasattr(self.skill_executor.camera, '_running') and
                    self.skill_executor.camera._running):
                    self.skill_executor.camera.stop()
                    self.logger.info("Skill executor camera stopped")
        except Exception as e:
            self.logger.warning(f"Error stopping skill executor camera: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.shutdown()


# Example usage and testing functions
def test_task_planner():
    """Test the task planner with example tasks (requires LLM for dynamic planning)"""
    
    print("=== Dynamic Task Planner Test ===")
    print("This test requires LLM integration for dynamic task decomposition")
    
    # Initialize task planner with LLM required
    planner = None
    try:
        planner = TaskPlanner(
            use_llm=True, 
            enable_data_recording=True, 
            recording_timestep=0.2,
            collect_openvla_data=True  # Enable OpenVLA data collection
        )
        
        # Test available skills
        print(f"\nAvailable Skills: {list(planner.valid_skills.keys())}")
        print(f"Data Recording Enabled: {planner.is_data_recording_enabled()}")
        print(f"OpenVLA Collection Enabled: {planner.collect_openvla_data}")
        print(f"Recording Timestep: {planner.get_recording_timestep()}s")
        
        # Test task decomposition with dynamic LLM planning
        test_tasks = [
            # "move the paper bag to the stove and open the bottle",
            # "open the bottle and put the cap in the bag."
            "open the bottle"
            # "turn on the kitchen light", 
            # "open the bottle and pour water",
            # "clean up the counter and close all cabinets"
        ]
        
        for task in test_tasks:
            print(f"\n=== Testing Task: {task} ===")
            print(f"Task: {task}")
            print(f"Required: LLM analysis of environment image")
            print(f"Expected: Dynamic skill sequence generation")
            print(f"Available skills for planning: {list(planner.valid_skills.keys())}")
            
            try:
                # Actually run the task planning and execution with recording enabled
                results = planner.execute_task(task, execute_on_robot=True, record_data=True)
                
                print(f"Generated {len(results['skill_commands'])} skill commands:")
                for i, command in enumerate(results['skill_commands']):
                    print(f"  Command {i+1}: {command}")
                
                print(f"\nSkill Generation & Execution Results:")
                for i, result in enumerate(results['execution_results']):
                    status = "✓" if result['success'] else "✗"
                    print(f"  {status} {result['command']}")
                    if result.get('skill_generated', False):
                        print(f"    → Skill generated and executed successfully")
                    if result.get('error'):
                        print(f"    → Error: {result['error']}")
                
                if 'skills_generated' in results:
                    print(f"\nDetailed Skill Generation:")
                    for skill_info in results['skills_generated']:
                        print(f"  • {skill_info}")
                
                print(f"\nOverall Success: {results['success']}")
                if not results['success']:
                    print(f"Error: {results['error_message']}")
                
                # Show data recording results
                if 'session_data_path' in results:
                    print(f"\nData Recording Results:")
                    print(f"  Session data saved: {results['session_data_path']}")
                if 'openvla_dataset_path' in results:
                    print(f"  OpenVLA dataset saved: {results['openvla_dataset_path']}")
                if 'session_data_path' not in results and 'openvla_dataset_path' not in results:
                    print(f"\nNo data was recorded for this execution.")
                    
                # Show summary
                total = len(results['skill_commands'])
                successful = sum(1 for r in results['execution_results'] if r['success'])
                print(f"Summary: {successful}/{total} skills executed successfully")
                    
            except Exception as e:
                print(f"Task execution failed: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"Task planner initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print("To test dynamic planning:")
        print("1. Ensure LLM interface is configured")
        print("2. Connect robot and camera")
        print("3. Run with actual environment images")
    finally:
        # Ensure proper cleanup
        if planner:
            planner.shutdown()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_task_planner()