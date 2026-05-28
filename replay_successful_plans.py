#!/usr/bin/env python3
"""
Replay Successful Plans Script

This script loads successful plans from saved session data, skips the LLM planning step,
and directly executes the generated skill specifications to collect new trajectory data.
"""

import json
import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import cv2

# Add the cognitive_bt_framework to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_bt_framework', 'src'))

from cognitive_bt_framework.src.skills.skill_executor import DirectSkillExecutor
from cognitive_bt_framework.src.data_recorder import DataRecorder
from cognitive_bt_framework.src.vision.realsense import Camera as RealSenseCamera


class PlanReplaySystem:
    """System for replaying successful plans without LLM generation"""
    
    def __init__(self, robot_ip: str = "192.168.1.224", enable_data_recording: bool = True, 
                 depth_capture_delay: float = 1.0):
        """
        Initialize the replay system
        
        Args:
            robot_ip: IP address of the robot
            enable_data_recording: Whether to record trajectory data during replay
            depth_capture_delay: Delay in seconds before capturing depth data for skill execution
        """
        self.logger = logging.getLogger(__name__)
        self.robot_ip = robot_ip
        self.enable_data_recording = enable_data_recording
        
        # Initialize skill executor
        self.skill_executor = DirectSkillExecutor(
            robot_ip=robot_ip,
            camera_params={'width': 640, 'height': 480, 'fps': 30},
            use_zed_camera=False,  # Use RealSense
            show_debug_windows=False,
            calibrate_transform=False,
            fast_mode=True,
            depth_capture_delay=depth_capture_delay
        )
        
        # Get camera reference
        self.camera = self.skill_executor.camera
        
        # Initialize data recording system
        if self.enable_data_recording:
            self.data_recorder = DataRecorder(recording_timestep=0.1)
            self.logger.info("Data recording system enabled for replay")
        else:
            self.data_recorder = None
            self.logger.info("Data recording system disabled")
    
    def load_session_data(self, session_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from a JSON file
        
        Args:
            session_file_path: Path to the session JSON file
            
        Returns:
            Loaded session data or None if failed
        """
        try:
            with open(session_file_path, 'r') as f:
                session_data = json.load(f)
            
            self.logger.info(f"Loaded session data: {session_data['task_name']}")
            self.logger.info(f"Task decomposition: {session_data['task_decomposition']}")
            self.logger.info(f"Skills generated: {len(session_data.get('skills_generated', []))}")
            
            return session_data
            
        except Exception as e:
            self.logger.error(f"Failed to load session data from {session_file_path}: {e}")
            return None
    
    def load_environment_image(self, session_data: Dict[str, Any], session_dir: str) -> Optional[np.ndarray]:
        """
        Load the environment image from the session data
        
        Args:
            session_data: Loaded session data
            session_dir: Directory containing the session file
            
        Returns:
            Environment image as numpy array or None if failed
        """
        try:
            env_image_path = session_data.get('environment_image_path')
            if not env_image_path:
                self.logger.warning("No environment image path in session data")
                return None
            
            # Convert relative path to absolute path
            if not os.path.isabs(env_image_path):
                env_image_path = os.path.join(os.path.dirname(session_dir), env_image_path)
            
            if not os.path.exists(env_image_path):
                self.logger.error(f"Environment image not found: {env_image_path}")
                return None
            
            # Load image
            image = cv2.imread(env_image_path)
            if image is None:
                self.logger.error(f"Failed to load environment image: {env_image_path}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.logger.info(f"Loaded environment image: {env_image_path}")
            
            return image_rgb
            
        except Exception as e:
            self.logger.error(f"Failed to load environment image: {e}")
            return None
    
    def load_points_of_interest(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Load points of interest from session data
        
        Args:
            session_data: Loaded session data
            
        Returns:
            List of points of interest
        """
        return session_data.get('points_of_interest', [])
    
    def load_stored_skills(self, session_data: Dict[str, Any], session_dir: str) -> List[str]:
        """
        Load the stored skill files from the original session
        
        Args:
            session_data: Loaded session data
            session_dir: Directory containing the session file
            
        Returns:
            List of paths to stored skill JSON files
        """
        stored_skill_paths = session_data.get('stored_skill_paths', [])
        absolute_skill_paths = []
        
        # Ensure we maintain the original order from the session
        self.logger.info(f"Original stored skill paths order: {stored_skill_paths}")
        
        for skill_path in stored_skill_paths:
            # Convert relative path to absolute path
            if not os.path.isabs(skill_path):
                abs_path = os.path.join(os.path.dirname(session_dir), skill_path)
            else:
                abs_path = skill_path
            
            if os.path.exists(abs_path):
                absolute_skill_paths.append(abs_path)
                self.logger.info(f"Found stored skill file: {abs_path}")
            else:
                self.logger.warning(f"Stored skill file not found: {abs_path}")
        
        return absolute_skill_paths
    
    def execute_stored_skills(self, stored_skill_paths: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Execute pre-generated skills directly without re-running perception
        
        Args:
            stored_skill_paths: List of paths to stored skill JSON files
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not stored_skill_paths:
                return False, "No stored skills to execute"
            
            self.logger.info(f"Executing {len(stored_skill_paths)} stored skills")
            self.logger.info(f"Execution order will be: {[os.path.basename(p) for p in stored_skill_paths]}")
            
            # Execute each stored skill file directly
            for i, skill_path in enumerate(stored_skill_paths):
                self.logger.info(f"Executing stored skill {i+1}/{len(stored_skill_paths)}: {os.path.basename(skill_path)}")
                
                # Use the skill executor's direct execution method for stored skills
                if hasattr(self.skill_executor, 'execute_stored_skill'):
                    success, error_msg = self.skill_executor.execute_stored_skill(skill_path)
                elif hasattr(self.skill_executor, 'skill_handler') and hasattr(self.skill_executor.skill_handler, 'execute_stored_skill'):
                    # Try via skill handler
                    success, error_msg = self.skill_executor.skill_handler.execute_stored_skill(skill_path)
                else:
                    # Fallback: load and execute the skill JSON manually
                    success, error_msg = self._execute_skill_json(skill_path)
                
                if not success:
                    self.logger.error(f"Stored skill execution failed: {error_msg}")
                    return False, f"Skill {os.path.basename(skill_path)} failed: {error_msg}"
                else:
                    self.logger.info(f"✓ Stored skill executed successfully: {os.path.basename(skill_path)}")
            
            self.logger.info("All stored skills executed successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to execute stored skills: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _execute_skill_json(self, skill_json_path: str) -> Tuple[bool, Optional[str]]:
        """
        Fallback method to execute a skill JSON file manually
        
        Args:
            skill_json_path: Path to the skill JSON file
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Load the skill JSON file
            with open(skill_json_path, 'r') as f:
                skill_data = json.load(f)
            
            # Extract the primitive sequence from the skill data
            if 'primitive_sequence' not in skill_data:
                return False, f"No primitive_sequence found in skill file: {skill_json_path}"
            
            primitive_sequence = skill_data['primitive_sequence']
            points_of_interest = skill_data.get('points_of_interest', {})
            surface_info = skill_data.get('surface_info', {})
            
            self.logger.info(f"Executing skill: {skill_data.get('name', 'unknown')}")
            self.logger.info(f"Primitive sequence: {primitive_sequence}")
            if surface_info:
                self.logger.info(f"Surface info available: {list(surface_info.keys())}")
            
            # Execute the primitive sequence using the skill executor directly
            if hasattr(self.skill_executor, 'execute_primitive_command'):
                # Execute each primitive in the sequence using the skill executor
                for primitive in primitive_sequence:
                    self.logger.info(f"Executing primitive: {primitive}")
                    success = self.skill_executor.execute_primitive_command(
                        primitive,
                        points_of_interest,
                        surface_info
                    )
                    if not success:
                        return False, f"Failed to execute primitive: {primitive}"
                
                return True, None
            
            # Fallback to skill handler if available
            elif hasattr(self.skill_executor, 'skill_handler') and self.skill_executor.skill_handler:
                skill_handler = self.skill_executor.skill_handler
                
                # Check if skill handler has a method to execute primitives directly
                if hasattr(skill_handler, 'execute_primitive_sequence'):
                    success = skill_handler.execute_primitive_sequence(
                        primitive_sequence, 
                        points_of_interest
                    )
                    return success, None if success else "Primitive sequence execution failed"
                
                elif hasattr(skill_handler, 'execute_action_from_string'):
                    # Execute each primitive in the sequence
                    for primitive in primitive_sequence:
                        self.logger.info(f"Executing primitive: {primitive}")
                        success = skill_handler.execute_action_from_string(
                            primitive,
                            points_of_interest
                        )
                        if not success:
                            return False, f"Failed to execute primitive: {primitive}"
                    
                    return True, None
                else:
                    return False, "Skill handler does not support primitive execution"
            else:
                return False, "No skill handler available for primitive execution"
                
        except Exception as e:
            return False, f"Failed to execute skill JSON {skill_json_path}: {str(e)}"
    
    def capture_current_environment(self) -> Optional[np.ndarray]:
        """
        Capture current environment image for comparison/verification
        
        Returns:
            Current environment image as numpy array
        """
        try:
            if hasattr(self.camera, 'get_frames'):
                frames = self.camera.get_frames()
                if not frames:
                    self.logger.error("Failed to capture current frames")
                    return None
                rgb_image, _ = frames
                return rgb_image
            else:
                self.logger.error("Camera does not have get_frames method")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to capture current environment: {e}")
            return None
    
    def replay_plan(self, session_file_path: str, verify_environment: bool = True, auto_save: bool = False, use_stored_skills: bool = True) -> Dict[str, Any]:
        """
        Replay a successful plan from session data
        
        Args:
            session_file_path: Path to the session JSON file
            verify_environment: Whether to capture current environment for comparison
            auto_save: If True, automatically save all data without user prompts
            use_stored_skills: If True, use stored skill files; if False, regenerate skills
            
        Returns:
            Replay results dictionary
        """
        results = {
            "session_file": session_file_path,
            "start_time": time.time(),
            "task_description": None,
            "task_decomposition": [],
            "skills_executed": [],
            "success": False,
            "error_message": None,
            "session_data_path": None
        }
        
        try:
            # Load session data
            session_data = self.load_session_data(session_file_path)
            if not session_data:
                results["error_message"] = "Failed to load session data"
                return results
            
            results["task_description"] = session_data.get("natural_language_task", "Unknown task")
            results["task_decomposition"] = session_data.get("task_decomposition", [])
            
            # Start data recording session if enabled
            if self.enable_data_recording and self.data_recorder:
                self.data_recorder.start_recording_session(
                    natural_language_task=results["task_description"],
                    task_name=session_data.get("task_name", "replayed_task"),
                    robot_ip=self.robot_ip,
                    camera_type="realsense",
                    execution_mode="replay"
                )
            
            # Load original environment image
            session_dir = os.path.dirname(session_file_path)
            original_env_image = self.load_environment_image(session_data, session_dir)
            
            # Capture current environment if requested
            current_env_image = None
            if verify_environment:
                current_env_image = self.capture_current_environment()
                if current_env_image is not None:
                    self.logger.info("Captured current environment for comparison")
            
            # Record environment images
            if self.enable_data_recording and self.data_recorder:
                if current_env_image is not None:
                    self.data_recorder.record_environment_image(current_env_image)
                elif original_env_image is not None:
                    self.data_recorder.record_environment_image(original_env_image)
            
            # Load points of interest from the original session
            points_of_interest = self.load_points_of_interest(session_data)
            if points_of_interest and self.enable_data_recording and self.data_recorder:
                # Convert to the format expected by data recorder
                poi_records = []
                for poi in points_of_interest:
                    if isinstance(poi, dict):
                        poi_records.append(poi)
                self.data_recorder.record_points_of_interest(poi_records)
            
            if use_stored_skills:
                # Load stored skills from the original session
                stored_skill_paths = self.load_stored_skills(session_data, session_dir)
                if not stored_skill_paths:
                    results["error_message"] = "No stored skill files found in session data"
                    return results
                
                # Get the skill info from the original session for logging
                skills_generated = session_data.get("skills_generated", [])
                skill_names = [skill.get("skill_name", "unknown") for skill in skills_generated]
                
                self.logger.info(f"Replaying stored skills: {[os.path.basename(p) for p in stored_skill_paths]}")
                self.logger.info(f"Original skill sequence was: {skill_names}")
            else:
                # Regenerate skills mode - use original skill sequence
                skills_generated = session_data.get("skills_generated", [])
                if not skills_generated:
                    results["error_message"] = "No skills found in session data for regeneration"
                    return results
                
                # Convert skills to the format expected by skill executor
                skill_sequence = []
                for skill_info in skills_generated:
                    skill_name = skill_info.get("skill_name", "")
                    parameters = skill_info.get("parameters", "")
                    if skill_name:
                        skill_sequence.append((skill_name, parameters))
                
                if not skill_sequence:
                    results["error_message"] = "No valid skills found in session data for regeneration"
                    return results
                
                self.logger.info(f"Regenerating skill sequence: {skill_sequence}")
                stored_skill_paths = None  # Not using stored skills
            
            # Record planning completion (reusing original plan)
            if self.enable_data_recording and self.data_recorder:
                self.data_recorder.record_planning_complete(results["task_decomposition"])
                self.data_recorder.record_inference_start()
                self.data_recorder.record_inference_complete(
                    skills_generated=skills_generated,
                    llm_responses={"execution_mode": "replay", "source": "loaded_from_session"}
                )
                self.data_recorder.record_execution_start()
                
                # Set up motion recording interfaces
                robot_interface = None
                if hasattr(self.skill_executor, 'motion_planner'):
                    robot_interface = self.skill_executor.motion_planner
                
                camera_interface = self.camera
                
                self.data_recorder.set_motion_interfaces(
                    robot_interface=robot_interface,
                    camera_interface=camera_interface
                )
                
                # Pass data recorder to skill executor for motion tracking
                if hasattr(self.skill_executor, 'set_data_recorder'):
                    self.skill_executor.set_data_recorder(self.data_recorder)
            
            # Start motion recording
            if self.enable_data_recording and self.data_recorder:
                print("Starting motion data recording for replay...")
                self.data_recorder.start_motion_recording()
            
            # Execute skills based on the selected mode
            if use_stored_skills:
                # Execute the stored skills directly (reusing original perception/planning)
                success, error_msg = self.execute_stored_skills(stored_skill_paths)
            else:
                # Regenerate and execute the skill sequence
                success, error_msg = self.skill_executor.execute_skill_sequence(
                    skill_sequence, 
                    task_context=results["task_description"]
                )
            
            # Stop motion recording
            if self.enable_data_recording and self.data_recorder:
                self.data_recorder.stop_motion_recording()
            
            # Record results based on execution mode
            if use_stored_skills:
                # Record results for executed stored skills
                for skill_path in stored_skill_paths:
                    skill_filename = os.path.basename(skill_path)
                    skill_result = {
                        "stored_skill_file": skill_filename,
                        "skill_path": skill_path,
                        "success": success,  # For now, treat as all-or-nothing
                        "error": error_msg if not success else None
                    }
                    results["skills_executed"].append(skill_result)
            else:
                # Record results for regenerated skills
                for skill_name, parameters in skill_sequence:
                    skill_result = {
                        "skill_name": skill_name,
                        "parameters": parameters,
                        "success": success,  # For now, treat as all-or-nothing
                        "error": error_msg if not success else None
                    }
                    results["skills_executed"].append(skill_result)
            
            results["success"] = success
            if not success:
                results["error_message"] = f"Skill execution failed: {error_msg}"
            
            # Ask user for success confirmation before saving (unless auto_save is enabled)
            user_success_rating = False
            if self.enable_data_recording and self.data_recorder:
                if success:
                    if auto_save:
                        # Auto-save mode: assume technical success is user success
                        user_success_rating = True
                        print(f"✓ Replay completed successfully (auto-saved): {results['task_description']}")
                    else:
                        # Interactive mode: ask user for confirmation
                        print(f"\nReplay completed for task: {results['task_description']}")
                        print("Please evaluate if the robot task was executed successfully.")
                        user_input = input("Was the task executed successfully? (y/n): ").strip().lower()
                        user_success_rating = user_input in ['y', 'yes', '1', 'true']
                        
                        if user_success_rating:
                            print("✓ Marked as successful - data will be saved")
                        else:
                            print("✗ Marked as failed - data will be saved with failure status")
                            # Get optional user notes about why it failed
                            user_notes = input("Optional: Why did it fail? (press enter to skip): ").strip()
                            if user_notes:
                                results["user_failure_notes"] = user_notes
                else:
                    if auto_save:
                        print(f"✗ Replay failed (auto-saved): {results['task_description']} - {error_msg}")
                    else:
                        print(f"\nReplay failed for task: {results['task_description']}")
                        print(f"Error: {error_msg}")
                    user_success_rating = False
                
                # Record execution completion with user rating
                failure_messages = [error_msg] if not success and error_msg else []
                if not user_success_rating and success:
                    # Technical success but user rated as failed
                    failure_messages.append("User rated execution as unsuccessful")
                
                self.data_recorder.record_execution_complete(
                    success=user_success_rating,  # Use user rating instead of technical success
                    failure_messages=failure_messages,
                    robot_errors=failure_messages if not success else []
                )
                
                # Finalize session
                try:
                    session_path = self.data_recorder.finalize_session(prompt_for_save=not auto_save)
                    results["session_data_path"] = session_path
                    results["user_success_rating"] = user_success_rating
                    self.logger.info(f"Replay data saved to: {session_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to finalize recording session: {e}")
            
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]
            
            self.logger.info(f"Plan replay completed. Success: {success}")
            
        except Exception as e:
            error_message = f"Plan replay failed: {str(e)}"
            results["error_message"] = error_message
            results["end_time"] = time.time()
            self.logger.error(error_message)
            
            # Clean up recording if it was started
            if self.enable_data_recording and self.data_recorder:
                try:
                    self.data_recorder.stop_motion_recording()
                    
                    # Handle exception data saving based on auto_save mode
                    if auto_save:
                        # Auto-save mode: always save failure data
                        save_failed_data = True
                        print(f"✗ Replay failed (auto-saved): {results['task_description']} - {error_message}")
                    else:
                        # Interactive mode: ask user for confirmation
                        print(f"\nReplay failed due to exception: {results['task_description']}")
                        print(f"Error: {error_message}")
                        user_input = input("Save this failed execution data? (y/n): ").strip().lower()
                        save_failed_data = user_input in ['y', 'yes', '1', 'true']
                    
                    if save_failed_data:
                        self.data_recorder.record_execution_complete(
                            success=False,
                            failure_messages=[error_message],
                            robot_errors=[str(e)]
                        )
                        session_path = self.data_recorder.finalize_session(prompt_for_save=not auto_save)
                        results["session_data_path"] = session_path
                        results["user_success_rating"] = False
                        if not auto_save:
                            print("✓ Failed execution data saved")
                    else:
                        if not auto_save:
                            print("✗ Failed execution data not saved")
                        
                except Exception as recording_error:
                    self.logger.warning(f"Failed to record execution failure: {recording_error}")
        
        return results
    
    def batch_replay_plans(self, session_files: List[str], verify_environment: bool = True, auto_save: bool = True, use_stored_skills: bool = True) -> List[Dict[str, Any]]:
        """
        Replay multiple plans in batch
        
        Args:
            session_files: List of session file paths
            verify_environment: Whether to capture current environment for comparison
            auto_save: If True, automatically save all data without user prompts
            use_stored_skills: If True, use stored skill files; if False, regenerate skills
            
        Returns:
            List of replay results
        """
        results = []
        
        for i, session_file in enumerate(session_files):
            self.logger.info(f"Replaying plan {i+1}/{len(session_files)}: {session_file}")
            time.sleep(2)
            try:
                result = self.replay_plan(session_file, verify_environment, auto_save, use_stored_skills)
                results.append(result)
                
                # Log summary
                if result["success"]:
                    self.logger.info(f"✓ Plan {i+1} replayed successfully")
                else:
                    self.logger.warning(f"✗ Plan {i+1} failed: {result.get('error_message', 'Unknown error')}")
                
                # Small delay between replays
                time.sleep(1.0)
                
            except Exception as e:
                error_result = {
                    "session_file": session_file,
                    "success": False,
                    "error_message": f"Exception during replay: {str(e)}"
                }
                results.append(error_result)
                self.logger.error(f"Exception replaying plan {i+1}: {e}")
        
        # Print batch summary
        successful = sum(1 for r in results if r.get("success", False))
        self.logger.info(f"Batch replay completed: {successful}/{len(results)} plans successful")
        
        return results
    
    def execute_session_skills(self, session_file_path: str, auto_save: bool = True) -> Dict[str, Any]:
        """
        Execute all skills from a session file directly
        
        Args:
            session_file_path: Path to the session JSON file (can be relative or absolute)
            auto_save: If True, automatically save data without user prompts
            
        Returns:
            Execution results dictionary
        """
        results = {
            "session_file": session_file_path,
            "start_time": time.time(),
            "success": False,
            "error_message": None,
            "session_data_path": None
        }
        
        try:
            import os
            
            # Handle relative paths - check common locations
            possible_paths = [
                session_file_path,
                os.path.join("cognitive_bt_framework/src/task_execution_data/sessions", session_file_path),
                os.path.join("task_execution_data/sessions", session_file_path),
                os.path.join("raw_data", session_file_path, "session_data.json") if not session_file_path.endswith('.json') else None
            ]
            
            # Filter out None values
            possible_paths = [p for p in possible_paths if p is not None]
            
            found_session_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    found_session_path = path
                    break
            
            if not found_session_path:
                results["error_message"] = f"Session file not found: {session_file_path}. Checked: {possible_paths}"
                return results
            
            self.logger.info(f"Executing session from: {found_session_path}")
            
            # Use replay_plan method to execute all skills in session
            replay_result = self.replay_plan(
                found_session_path,
                verify_environment=False,  # Skip environment verification for direct execution
                auto_save=auto_save,
                use_stored_skills=True  # Default to using stored skills
            )
            
            # Copy results from replay_plan
            results.update(replay_result)
            results["session_file"] = session_file_path  # Keep original path reference
            
        except Exception as e:
            results["error_message"] = f"Failed to execute session: {str(e)}"
            self.logger.error(f"Session execution failed: {e}")
        
        return results

    def execute_single_skill(self, skill_file_path: str, auto_save: bool = True) -> Dict[str, Any]:
        """
        Execute a single skill file directly
        
        Args:
            skill_file_path: Path to the skill JSON file (can be relative or absolute)
            auto_save: If True, automatically save data without user prompts
            
        Returns:
            Execution results dictionary
        """
        results = {
            "skill_file": skill_file_path,
            "start_time": time.time(),
            "success": False,
            "error_message": None,
            "session_data_path": None
        }
        
        try:
            import os
            
            # Handle relative paths - check common locations
            if not os.path.isabs(skill_file_path):
                possible_paths = [
                    skill_file_path,  # Current directory
                    os.path.join("cognitive_bt_framework/src/stored_skills", skill_file_path),
                    os.path.join("cognitive_bt_framework/src/task_execution_data/stored_skills", skill_file_path),
                    os.path.join(os.path.dirname(__file__), "cognitive_bt_framework/src/stored_skills", skill_file_path),
                    os.path.join(os.path.dirname(__file__), "cognitive_bt_framework/src/task_execution_data/stored_skills", skill_file_path)
                ]
                
                skill_file_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        skill_file_path = path
                        break
                
                if skill_file_path is None:
                    results["error_message"] = f"Skill file not found in any of the expected locations: {possible_paths}"
                    return results
            
            if not os.path.exists(skill_file_path):
                results["error_message"] = f"Skill file not found: {skill_file_path}"
                return results
            
            self.logger.info(f"Executing skill file: {skill_file_path}")
            
            # Start data recording session if enabled
            if self.enable_data_recording and self.data_recorder:
                self.data_recorder.start_recording_session(
                    natural_language_task=f"Direct skill execution: {os.path.basename(skill_file_path)}",
                    task_name=f"direct_skill_{os.path.basename(skill_file_path).replace('.json', '')}",
                    robot_ip=self.robot_ip,
                    camera_type="realsense",
                    execution_mode="direct_skill"
                )
            
            # Record planning completion (no actual planning for direct skills)
            if self.enable_data_recording and self.data_recorder:
                self.data_recorder.record_planning_complete(["Direct skill execution"])
                self.data_recorder.record_inference_start()
                self.data_recorder.record_inference_complete(
                    skills_generated=[{"skill_name": "direct_skill", "skill_file": skill_file_path}],
                    llm_responses={"execution_mode": "direct_skill", "skill_file": skill_file_path}
                )
                self.data_recorder.record_execution_start()
                
                # Set up motion recording interfaces
                robot_interface = None
                if hasattr(self.skill_executor, 'motion_planner'):
                    robot_interface = self.skill_executor.motion_planner
                
                camera_interface = self.camera
                
                self.data_recorder.set_motion_interfaces(
                    robot_interface=robot_interface,
                    camera_interface=camera_interface
                )
                
                # Pass data recorder to skill executor for motion tracking
                if hasattr(self.skill_executor, 'set_data_recorder'):
                    self.skill_executor.set_data_recorder(self.data_recorder)
            
            # Start motion recording
            if self.enable_data_recording and self.data_recorder:
                print("Starting motion data recording for direct skill execution...")
                self.data_recorder.start_motion_recording()
            
            # Execute the skill file directly
            success, error_msg = self.skill_executor.execute_stored_skill(skill_file_path)
            
            # Stop motion recording
            if self.enable_data_recording and self.data_recorder:
                self.data_recorder.stop_motion_recording()
            
            results["success"] = success
            if not success:
                results["error_message"] = f"Skill execution failed: {error_msg}"
            
            # Ask user for success confirmation before saving (unless auto_save is enabled)
            user_success_rating = False
            if self.enable_data_recording and self.data_recorder:
                if success:
                    if auto_save:
                        user_success_rating = True
                        print(f"✓ Skill executed successfully (auto-saved): {os.path.basename(skill_file_path)}")
                    else:
                        print(f"\nSkill execution completed: {os.path.basename(skill_file_path)}")
                        print("Please evaluate if the skill was executed successfully.")
                        user_input = input("Was the skill executed successfully? (y/n): ").strip().lower()
                        user_success_rating = user_input in ['y', 'yes', '1', 'true']
                        
                        if user_success_rating:
                            print("✓ Marked as successful - data will be saved")
                        else:
                            print("✗ Marked as failed - data will be saved with failure status")
                else:
                    if auto_save:
                        print(f"✗ Skill execution failed (auto-saved): {os.path.basename(skill_file_path)} - {error_msg}")
                    else:
                        print(f"\nSkill execution failed: {os.path.basename(skill_file_path)}")
                        print(f"Error: {error_msg}")
                    user_success_rating = False
                
                # Record execution completion with user rating
                failure_messages = [error_msg] if not success and error_msg else []
                if not user_success_rating and success:
                    failure_messages.append("User rated execution as unsuccessful")
                
                self.data_recorder.record_execution_complete(
                    success=user_success_rating,
                    failure_messages=failure_messages,
                    robot_errors=failure_messages if not success else []
                )
                
                # Finalize session
                try:
                    session_path = self.data_recorder.finalize_session(prompt_for_save=not auto_save)
                    results["session_data_path"] = session_path
                    results["user_success_rating"] = user_success_rating
                    self.logger.info(f"Direct skill execution data saved to: {session_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to finalize recording session: {e}")
            
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]
            
            self.logger.info(f"Direct skill execution completed. Success: {success}")
            
        except Exception as e:
            error_message = f"Direct skill execution failed: {str(e)}"
            results["error_message"] = error_message
            results["end_time"] = time.time()
            self.logger.error(error_message)
            
            # Clean up recording if it was started
            if self.enable_data_recording and self.data_recorder:
                try:
                    self.data_recorder.stop_motion_recording()
                    if auto_save:
                        self.data_recorder.record_execution_complete(
                            success=False,
                            failure_messages=[error_message],
                            robot_errors=[str(e)]
                        )
                        session_path = self.data_recorder.finalize_session(prompt_for_save=False)
                        results["session_data_path"] = session_path
                        print(f"✗ Direct skill execution failed (auto-saved): {error_message}")
                except Exception as recording_error:
                    self.logger.warning(f"Failed to record execution failure: {recording_error}")
        
        return results
    
    def shutdown(self):
        """Clean shutdown of the replay system"""
        try:
            if hasattr(self, 'skill_executor') and self.skill_executor:
                if (hasattr(self.skill_executor, 'camera') and 
                    self.skill_executor.camera and 
                    hasattr(self.skill_executor.camera, '_running') and
                    self.skill_executor.camera._running):
                    self.skill_executor.camera.stop()
                    self.logger.info("Skill executor camera stopped")
        except Exception as e:
            self.logger.warning(f"Error stopping skill executor camera: {e}")


def find_successful_sessions(data_directory: str, task_filter: Optional[str] = None, exact_match: bool = False) -> List[str]:
    """
    Find successful session files in the data directory
    
    Args:
        data_directory: Directory containing session files
        task_filter: Optional task name filter (e.g., "open_the_bottle")
        exact_match: If True, require exact task name match; if False, match as substring with word boundaries
        
    Returns:
        List of paths to successful session files
    """
    session_files = []
    sessions_dir = os.path.join(data_directory, "sessions")
    
    if not os.path.exists(sessions_dir):
        print(f"Sessions directory not found: {sessions_dir}")
        return session_files
    
    for filename in os.listdir(sessions_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(sessions_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    session_data = json.load(f)
                
                # Check if this is a successful session based on user rating
                user_success_rating = session_data.get('user_success_rating', False)
                task_name = session_data.get('task_name', '')
                
                # Apply task filter if specified
                if task_filter:
                    if exact_match:
                        # Exact task name match
                        if task_name.lower() != task_filter.lower():
                            continue
                    else:
                        # Word boundary matching - matches complete words only
                        import re
                        pattern = r'\b' + re.escape(task_filter) + r'\b'
                        if not re.search(pattern, task_name, re.IGNORECASE):
                            continue
                
                # Only include user-rated successful sessions with skills
                skills_generated = session_data.get('skills_generated', [])
                if user_success_rating and len(skills_generated) > 0:
                    session_files.append(filepath)
                    print(f"Found successful session: {task_name} - {filename}")
                
            except Exception as e:
                print(f"Error reading session file {filename}: {e}")
                continue
    
    return sorted(session_files)


def main():
    """Main function for the replay script"""
    parser = argparse.ArgumentParser(description="Replay successful robot task plans")
    parser.add_argument("--data-dir",
                       default=os.path.join(os.environ.get("PROJECT_ROOT", "."), "cognitive_bt_framework/src/task_execution_data"),
                       help="Directory containing session data")
    parser.add_argument("--session-file", 
                       help="Specific session file to replay (overrides batch mode)")
    parser.add_argument("--skill-file",
                       help="Specific skill file to execute directly (e.g., 'open pen_pen_generic_9bc364.json')")
    parser.add_argument("--session-skills",
                       help="Execute all skills from a specific session file (like --skill-file but for sessions)")
    parser.add_argument("--task-filter",
                       help="Filter sessions by task name (e.g., 'open_the_bottle')")
    parser.add_argument("--robot-ip", 
                       default="192.168.1.224",
                       help="Robot IP address")
    parser.add_argument("--no-recording", 
                       action="store_true",
                       help="Disable trajectory data recording")
    parser.add_argument("--no-env-verify", 
                       action="store_true",
                       help="Skip environment verification capture")
    parser.add_argument("--max-sessions", 
                       type=int, 
                       help="Maximum number of sessions to replay")
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--interactive", 
                       action="store_true",
                       help="Enable interactive mode (ask user for success confirmation after each replay)")
    parser.add_argument("--use-stored-skills", 
                       action="store_true", 
                       default=True,
                       help="Use stored skill files instead of regenerating (default: True)")
    parser.add_argument("--regenerate-skills", 
                       action="store_true",
                       help="Regenerate skills instead of using stored files (overrides --use-stored-skills)")
    parser.add_argument("--exact-match", 
                       action="store_true",
                       help="Require exact task name match for filtering (default: word boundary matching)")
    parser.add_argument("--depth-delay", 
                       type=float, 
                       default=1.0,
                       help="Delay in seconds before capturing depth data for skill execution (default: 1.0)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Initialize replay system
    replay_system = None
    try:
        replay_system = PlanReplaySystem(
            robot_ip=args.robot_ip,
            enable_data_recording=not args.no_recording,
            depth_capture_delay=args.depth_delay
        )
        
        # Determine execution mode
        use_stored_skills = not args.regenerate_skills
        execution_mode = "stored skills" if use_stored_skills else "regenerated skills"
        logger.info(f"Execution mode: {execution_mode}")
        
        if args.skill_file:
            # Direct skill file execution
            logger.info(f"Executing skill file directly: {args.skill_file}")
            result = replay_system.execute_single_skill(
                args.skill_file,
                auto_save=not args.interactive
            )
            
            # Print results
            print(f"\nSkill Execution Results:")
            print(f"Skill: {result['skill_file']}")
            print(f"Success: {result['success']}")
            if not result['success']:
                print(f"Error: {result['error_message']}")
            if result.get('session_data_path'):
                print(f"Data saved to: {result['session_data_path']}")
                
        elif args.session_skills:
            # Direct session skills execution (all skills from session)
            logger.info(f"Executing all skills from session: {args.session_skills}")
            result = replay_system.execute_session_skills(
                args.session_skills,
                auto_save=not args.interactive
            )
            
            # Print results
            print(f"\nSession Skills Execution Results:")
            print(f"Session: {result['session_file']}")
            print(f"Task: {result.get('task_description', 'Unknown')}")
            print(f"Success: {result['success']}")
            print(f"Skills executed: {len(result.get('skills_executed', []))}")
            if not result['success']:
                print(f"Error: {result['error_message']}")
            if result.get('session_data_path'):
                print(f"Data saved to: {result['session_data_path']}")
                
        elif args.session_file:
            # Single session replay
            logger.info(f"Replaying single session: {args.session_file}")
            result = replay_system.replay_plan(
                args.session_file, 
                verify_environment=not args.no_env_verify,
                auto_save=not args.interactive,
                use_stored_skills=use_stored_skills
            )
            
            # Print results
            print(f"\nReplay Results:")
            print(f"Task: {result['task_description']}")
            print(f"Success: {result['success']}")
            if not result['success']:
                print(f"Error: {result['error_message']}")
            if result.get('session_data_path'):
                print(f"Data saved to: {result['session_data_path']}")
            
        else:
            # Batch replay
            logger.info(f"Finding successful sessions in: {args.data_dir}")
            session_files = find_successful_sessions(args.data_dir, args.task_filter, args.exact_match)
            
            if not session_files:
                print("No successful sessions found!")
                return
            
            if args.max_sessions:
                session_files = session_files[:args.max_sessions]
            
            print(f"\nFound {len(session_files)} successful sessions to replay")
            
            # Replay all sessions
            results = replay_system.batch_replay_plans(
                session_files, 
                verify_environment=not args.no_env_verify,
                auto_save=not args.interactive,
                use_stored_skills=use_stored_skills
            )
            
            # Print summary
            successful = sum(1 for r in results if r.get("success", False))
            print(f"\nBatch Replay Summary:")
            print(f"Total sessions: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            
            # Print individual results
            for i, result in enumerate(results):
                status = "✓" if result.get("success", False) else "✗"
                task = result.get("task_description", "Unknown")
                print(f"{status} Session {i+1}: {task}")
                if not result.get("success", False):
                    print(f"  Error: {result.get('error_message', 'Unknown error')}")
        
    except KeyboardInterrupt:
        logger.info("Replay interrupted by user")
    except Exception as e:
        logger.error(f"Replay system error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        if replay_system:
            replay_system.shutdown()


if __name__ == "__main__":
    main()