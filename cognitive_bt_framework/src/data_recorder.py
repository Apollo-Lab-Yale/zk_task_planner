#!/usr/bin/env python3
"""
Data Recording System for Task Planner
Records comprehensive data for each task execution run including timing, failures, user feedback,
joint states, RGBD images, and robot commands at configurable timesteps during execution.
"""

import json
import time
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import base64
import uuid
from PIL import Image
import io
import threading
import queue

@dataclass
class TimingData:
    """Store timing information for different phases"""
    planning_start: float
    planning_end: float
    inference_start: float
    inference_end: float
    execution_start: float
    execution_end: float
    
    @property
    def planning_duration(self) -> float:
        return self.planning_end - self.planning_start
    
    @property
    def inference_duration(self) -> float:
        return self.inference_end - self.inference_start
    
    @property
    def execution_duration(self) -> float:
        return self.execution_end - self.execution_start
    
    @property
    def total_duration(self) -> float:
        return self.execution_end - self.planning_start

@dataclass
class MotionDataRecord:
    """Record for robot motion data at a specific timestep"""
    timestamp: float
    joint_positions: List[float]  # Current joint angles
    joint_commands: List[float]   # Commanded joint angles
    gripper_state: float          # Gripper position/state
    gripper_command: float        # Commanded gripper position
    rgb_image_path: Optional[str] = None      # Path to RGB image
    depth_image_path: Optional[str] = None    # Path to depth image

@dataclass
class OpenVLADemonstrationRecord:
    """OpenVLA-compatible demonstration data for each timestep"""
    timestamp: float
    images: Dict[str, str]        # Image paths: {"wrist_cam": path, "external_cam": path}
    robot_state: Dict[str, Any]   # {"joint_positions": [...], "end_effector_pose": [...], "gripper_state": float}
    action: Dict[str, Any]        # {"delta_pose": [...], "gripper_action": float}
    language_instruction: str     # Natural language task description

@dataclass
class PointOfInterestRecord:
    """Record for points of interest detected"""
    label: str
    pixel_coordinates: Tuple[int, int]
    normalized_coordinates: Tuple[float, float]
    detection_method: str
    interaction_type: str
    confidence: float
    score: float
    stability: float
    accessibility: float

@dataclass
class TaskExecutionRecord:
    """Complete record of a task execution"""
    # Basic task information
    session_id: str
    timestamp: str
    natural_language_task: str
    task_name: str  # Short task identifier
    task_decomposition: List[str]
    
    # Generated skills and LLM responses
    skills_generated: List[Dict[str, Any]]
    llm_responses: Dict[str, str]  # Store raw LLM responses
    
    # Executed skills with complete action data including ObjectInfo
    executed_skills: List[Dict[str, Any]]
    
    # Points of interest and images
    points_of_interest: List[PointOfInterestRecord]
    environment_image_path: str
    surface_images_paths: List[str]
    
    # Skill generation files
    skill_generation_image_paths: List[str]
    stored_skill_paths: List[str]
    skill_image_id: Optional[str]
    
    # Motion data recorded during execution
    motion_data: List[MotionDataRecord]
    recording_timestep: float  # Timestep used for recording (seconds)
    
    # OpenVLA-compatible demonstration data
    demonstration_data: List[OpenVLADemonstrationRecord]
    
    # Timing information
    timing: TimingData
    
    # Execution results
    execution_success: bool
    failure_messages: List[str]
    robot_errors: List[str]
    
    # User feedback
    user_success_rating: Optional[bool]  # True for success, False for failure
    user_notes: Optional[str]
    user_explanation: Optional[str]
    
    # Technical details
    robot_ip: str
    camera_type: str
    execution_mode: str  # "simulation" or "real"

class DataRecorder:
    """Records comprehensive data for each task execution"""
    
    def __init__(self, data_dir: str = "task_execution_data", recording_timestep: float = 0.1):
        """
        Initialize data recorder
        
        Args:
            data_dir: Directory to store recorded data
            recording_timestep: Time interval between motion data recordings (seconds)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Motion images directory will be created per session
        self.motion_images_dir = None
        
        # Recording configuration
        self.recording_timestep = recording_timestep
        
        # Current session tracking
        self.current_session_id = None
        self.current_record = None
        self.timing_data = None
        
        # Motion recording state
        self.is_recording_motion = False
        self.motion_recording_thread = None
        self.motion_data_queue = queue.Queue()
        self.stop_recording_event = threading.Event()
        self.pause_recording_event = threading.Event()  # For pausing/resuming recording
        
        # Motion detection parameters
        self.motion_threshold = 0.01  # Minimum joint movement to consider "motion" (radians)
        self.previous_joint_positions = None
        self.stationary_timeout = 2.0  # Stop recording after 2 seconds of no motion
        self.last_motion_time = 0
        self.robot_connected = False  # Track if robot is actually connected
        
        # Robot and camera interfaces (will be set during recording session)
        self.robot_interface = None
        self.camera_interface = None
        
        # Image validation parameters
        self.min_brightness_threshold = 10  # Minimum average pixel value to consider image valid
        self.min_std_threshold = 5  # Minimum standard deviation to avoid completely uniform images
        
    def start_recording_session(self, natural_language_task: str, task_name: str = None, 
                               robot_ip: str = "192.168.1.224", 
                               camera_type: str = "realsense", execution_mode: str = "real") -> str:
        """
        Start a new recording session
        
        Args:
            natural_language_task: The original natural language task
            task_name: Short task identifier (auto-generated if None)
            robot_ip: IP address of the robot
            camera_type: Type of camera being used
            execution_mode: "simulation" or "real"
            
        Returns:
            session_id: Unique identifier for this session
        """
        self.current_session_id = str(uuid.uuid4())
        
        # Create session-specific motion images directory
        safe_session_id = self._sanitize_filename(self.current_session_id)
        self.motion_images_dir = self.images_dir / "motion" / safe_session_id
        self.motion_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate task name if not provided
        if task_name is None:
            task_name = self._generate_task_name(natural_language_task)
        
        # Initialize timing data
        self.timing_data = TimingData(
            planning_start=time.time(),
            planning_end=0,
            inference_start=0,
            inference_end=0,
            execution_start=0,
            execution_end=0
        )
        
        # Initialize record
        self.current_record = TaskExecutionRecord(
            session_id=self.current_session_id,
            timestamp=datetime.now().isoformat(),
            natural_language_task=natural_language_task,
            task_name=task_name,
            task_decomposition=[],
            skills_generated=[],
            llm_responses={},
            executed_skills=[],
            points_of_interest=[],
            environment_image_path="",
            surface_images_paths=[],
            skill_generation_image_paths=[],
            stored_skill_paths=[],
            skill_image_id=None,
            motion_data=[],
            recording_timestep=self.recording_timestep,
            demonstration_data=[],
            timing=self.timing_data,
            execution_success=False,
            failure_messages=[],
            robot_errors=[],
            user_success_rating=None,
            user_notes=None,
            user_explanation=None,
            robot_ip=robot_ip,
            camera_type=camera_type,
            execution_mode=execution_mode
        )
        
        print(f"Started recording session: {self.current_session_id} for task: {task_name}")
        return self.current_session_id
    
    def record_planning_complete(self, task_decomposition: List[str]):
        """Record completion of task planning phase"""
        if self.timing_data:
            self.timing_data.planning_end = time.time()
            print(f"Planning completed in {self.timing_data.planning_duration:.2f}s")
        
        if self.current_record:
            self.current_record.task_decomposition = task_decomposition.copy()
    
    def record_inference_start(self):
        """Record start of skill inference phase"""
        if self.timing_data:
            self.timing_data.inference_start = time.time()
    
    def record_inference_complete(self, skills_generated: List[Dict[str, Any]], llm_responses: Dict[str, str]):
        """Record completion of skill inference phase"""
        if self.timing_data:
            self.timing_data.inference_end = time.time()
            print(f"Inference completed in {self.timing_data.inference_duration:.2f}s")
        
        if self.current_record:
            self.current_record.skills_generated = skills_generated
            self.current_record.llm_responses = llm_responses
    
    def record_execution_start(self):
        """Record start of execution phase"""
        if self.timing_data:
            self.timing_data.execution_start = time.time()
    
    def record_execution_complete(self, success: bool, failure_messages: List[str] = None, robot_errors: List[str] = None):
        """Record completion of execution phase"""
        if self.timing_data:
            self.timing_data.execution_end = time.time()
            print(f"Execution completed in {self.timing_data.execution_duration:.2f}s")
        
        if self.current_record:
            self.current_record.execution_success = success
            self.current_record.failure_messages = failure_messages or []
            self.current_record.robot_errors = robot_errors or []
    
    def record_executed_skill(self, skill_command: str, target_object: str, instantiated_skill):
        """Record an executed skill with complete action data including ObjectInfo"""
        if not self.current_record:
            return
        
        try:
            # Serialize the InstantiatedSkill object including ExecutableActions with ObjectInfo
            skill_data = {
                'skill_command': skill_command,
                'target_object': target_object,
                'skill_name': getattr(instantiated_skill, 'skill_name', 'unknown'),
                'action_sequence': []
            }
            
            # Serialize each ExecutableAction in the sequence
            if hasattr(instantiated_skill, 'action_sequence'):
                for action in instantiated_skill.action_sequence:
                    action_data = {
                        'action_type': action.action_type,
                        'position': action.position.tolist() if action.position is not None else None,
                        'orientation': action.orientation.tolist() if action.orientation is not None else None,
                        'pixel_position': action.pixel_position,
                        'parameters': action.parameters,
                        'is_top_down_grasp': action.is_top_down_grasp,
                        'is_side_grasp': action.is_side_grasp
                    }
                    
                    # Serialize ObjectInfo if present
                    if hasattr(action, 'object_info') and action.object_info is not None:
                        object_info_data = {
                            'bbox': action.object_info.bbox.tolist() if hasattr(action.object_info, 'bbox') and action.object_info.bbox is not None else None,
                            'mask_available': hasattr(action.object_info, 'mask') and action.object_info.mask is not None,
                            'points_available': hasattr(action.object_info, 'points') and action.object_info.points is not None,
                            'image_available': hasattr(action.object_info, 'image') and action.object_info.image is not None,
                            'surface_masks_available': hasattr(action.object_info, 'surface_masks') and action.object_info.surface_masks is not None
                        }
                        action_data['object_info'] = object_info_data
                    else:
                        action_data['object_info'] = None
                    
                    skill_data['action_sequence'].append(action_data)
            
            self.current_record.executed_skills.append(skill_data)
            
        except Exception as e:
            print(f"Warning: Failed to record executed skill data: {e}")
    
    def record_environment_image(self, image: np.ndarray) -> str:
        """
        Record environment image and return path
        
        Args:
            image: RGB image array
            
        Returns:
            str: Path to saved image
        """
        if self.current_session_id is None:
            raise RuntimeError("No active recording session")
        
        # Validate image before saving
        if not self._is_valid_image(image):
            print("Warning: Environment image appears to be black/empty, skipping save")
            return ""
        
        image_filename = f"{self.current_session_id}_environment.jpg"
        image_path = self.images_dir / image_filename
        
        # Save as RGB using PIL to avoid color space conversion issues
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert numpy array to PIL Image and save as RGB
            pil_image = Image.fromarray(image.astype(np.uint8))
            pil_image.save(str(image_path), 'JPEG', quality=95)
        else:
            # Fallback for grayscale or other formats
            cv2.imwrite(str(image_path), image)
        
        if self.current_record:
            self.current_record.environment_image_path = str(image_path)
        
        return str(image_path)
    
    def record_surface_images(self, surface_images: Dict[str, np.ndarray]) -> List[str]:
        """
        Record surface/object cropped images
        
        Args:
            surface_images: Dictionary of surface name to image array
            
        Returns:
            List of paths to saved images
        """
        if self.current_session_id is None:
            raise RuntimeError("No active recording session")
        
        saved_paths = []
        
        for surface_name, image in surface_images.items():
            # Validate image before saving
            if not self._is_valid_image(image):
                print(f"Warning: Surface image '{surface_name}' appears to be black/empty, skipping")
                continue
                
            image_filename = f"{self.current_session_id}_surface_{surface_name}.jpg"
            image_path = self.images_dir / image_filename
            
            # Save as RGB using PIL to avoid color space conversion issues
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert numpy array to PIL Image and save as RGB
                pil_image = Image.fromarray(image.astype(np.uint8))
                pil_image.save(str(image_path), 'JPEG', quality=95)
            else:
                # Fallback for grayscale or other formats
                cv2.imwrite(str(image_path), image)
            saved_paths.append(str(image_path))
        
        if self.current_record:
            self.current_record.surface_images_paths = saved_paths
        
        return saved_paths
    
    def record_points_of_interest(self, points_data: Dict[str, Any]):
        """
        Record detected points of interest
        
        Args:
            points_data: Dictionary containing points information
        """
        if self.current_record is None:
            return
        
        points_records = []
        
        # Handle different formats of points data
        if 'pixel_coords' in points_data and 'ids' in points_data:
            # Format from perception system
            pixel_coords = points_data['pixel_coords']
            ids = points_data['ids']
            scores = points_data.get('scores', [1.0] * len(pixel_coords))
            detection_methods = points_data.get('detection_methods', ['unknown'] * len(pixel_coords))
            interaction_types = points_data.get('interaction_types', ['unknown'] * len(pixel_coords))
            confidences = points_data.get('confidences', [1.0] * len(pixel_coords))
            
            for i, (pixel_x, pixel_y) in enumerate(pixel_coords):
                # Assuming image dimensions for normalization - these should be passed in
                norm_x = pixel_x / 640.0  # Default camera width
                norm_y = pixel_y / 480.0  # Default camera height
                
                record = PointOfInterestRecord(
                    label=ids[i],
                    pixel_coordinates=(int(pixel_x), int(pixel_y)),
                    normalized_coordinates=(norm_x, norm_y),
                    detection_method=detection_methods[i] if i < len(detection_methods) else 'unknown',
                    interaction_type=interaction_types[i] if i < len(interaction_types) else 'unknown',
                    confidence=confidences[i] if i < len(confidences) else 1.0,
                    score=scores[i] if i < len(scores) else 1.0,
                    stability=0.5,  # Default values
                    accessibility=0.5
                )
                points_records.append(record)
        
        self.current_record.points_of_interest = points_records
    
    def record_skill_generation_files(self, skill_gen_files: Dict[str, Any]):
        """
        Record files generated during skill generation
        
        Args:
            skill_gen_files: Dictionary containing skill paths, image paths, and image ID
        """
        if self.current_record is None:
            return
        
        if 'skill_paths' in skill_gen_files:
            self.current_record.stored_skill_paths.extend(skill_gen_files['skill_paths'])
        
        if 'image_paths' in skill_gen_files:
            self.current_record.skill_generation_image_paths.extend(skill_gen_files['image_paths'])
        
        if 'image_id' in skill_gen_files:
            self.current_record.skill_image_id = skill_gen_files['image_id']
    
    def set_motion_interfaces(self, robot_interface=None, camera_interface=None):
        """
        Set robot and camera interfaces for motion data recording
        
        Args:
            robot_interface: Robot interface with get_robot_joint_state() method
            camera_interface: Camera interface with get_frames() method
        """
        self.robot_interface = robot_interface
        self.camera_interface = camera_interface
    
    def start_motion_recording(self, robot_interface=None, camera_interface=None):
        """
        Start recording motion data during execution
        
        Args:
            robot_interface: Robot interface with get_robot_joint_state() method
            camera_interface: Camera interface with get_frames() method
        """
        if not self.current_session_id or not self.current_record:
            print("Warning: No active recording session for motion recording")
            return False
        
        # Set interfaces if provided
        if robot_interface is not None:
            self.robot_interface = robot_interface
        if camera_interface is not None:
            self.camera_interface = camera_interface
        
        # Test robot connection
        if self.robot_interface and hasattr(self.robot_interface, 'get_robot_joint_state'):
            try:
                test_state = self.robot_interface.get_robot_joint_state()
                self.robot_connected = test_state is not None and len(test_state) > 0
                if self.robot_connected:
                    print(f"Robot connected - joint state size: {len(test_state)}")
                else:
                    print("Warning: Robot interface available but no joint state data")
            except Exception as e:
                print(f"Warning: Robot connection test failed: {e}")
                self.robot_connected = False
        else:
            print("Warning: No robot interface available for motion recording")
            self.robot_connected = False
        
        if self.is_recording_motion:
            print("Motion recording already started")
            return True
        
        # Warm up camera before starting recording to avoid black images
        if self.camera_interface and hasattr(self.camera_interface, 'get_frames'):
            print("Warming up camera before starting motion recording...")
            for i in range(5):  # Try to get 5 valid frames
                try:
                    frames = self.camera_interface.get_frames()
                    if frames:
                        rgb_image, _ = frames
                        if rgb_image is not None and self._is_valid_image(rgb_image):
                            print(f"Camera warmed up successfully after {i+1} attempts")
                            break
                    time.sleep(0.1)  # Short delay between attempts
                except Exception as e:
                    print(f"Camera warmup attempt {i+1} failed: {e}")
                    time.sleep(0.1)
            else:
                print("Warning: Camera warmup incomplete, may record black images initially")

        self.is_recording_motion = True
        self.stop_recording_event.clear()
        # Start paused to wait for motion tracking signals
        self.pause_recording_event.set()
        
        # Start motion recording thread
        self.motion_recording_thread = threading.Thread(
            target=self._motion_recording_loop,
            daemon=True
        )
        self.motion_recording_thread.start()
        
        status = "with robot data" if self.robot_connected else "camera only"
        print(f"Started motion recording at {self.recording_timestep}s intervals ({status})")
        print("Motion recording is paused - waiting for motion tracking signals from skill executor")
        return True
    
    def stop_motion_recording(self):
        """Stop recording motion data"""
        if not self.is_recording_motion:
            return
        
        self.is_recording_motion = False
        self.stop_recording_event.set()
        
        # Wait for recording thread to finish
        if self.motion_recording_thread and self.motion_recording_thread.is_alive():
            self.motion_recording_thread.join(timeout=2.0)
        
        # Process any remaining data in queue
        self._process_motion_data_queue()
        
        print(f"Stopped motion recording. Captured {len(self.current_record.motion_data)} motion samples")
    
    def pause_motion_recording(self):
        """Pause motion recording (robot stopped for planning/perception)"""
        if self.is_recording_motion:
            self.pause_recording_event.set()
            print("Motion recording paused (robot stationary for planning)")
    
    def resume_motion_recording(self):
        """Resume motion recording (robot starting to move again)"""
        if self.is_recording_motion:
            self.pause_recording_event.clear()
            print("Motion recording resumed (robot motion detected)")
    
    def start_robot_motion(self, action_name: str = "unknown"):
        """
        Signal that robot is starting a motion (called from skill executor/motion planner)
        
        Args:
            action_name: Name/type of the action being executed
        """
        if self.is_recording_motion:
            self.pause_recording_event.clear()
            print(f"Robot motion started: {action_name}")
    
    def end_robot_motion(self, action_name: str = "unknown"):
        """
        Signal that robot motion has ended (called from skill executor/motion planner)
        
        Args:
            action_name: Name/type of the action that was executed
        """
        if self.is_recording_motion:
            # Continue recording for a short period after motion ends to capture final state
            print(f"Robot motion ended: {action_name}")
            # Note: We don't pause immediately to capture the final settled state
    
    def is_robot_moving(self, current_joint_positions: List[float]) -> bool:
        """
        Detect if robot is currently moving based on joint position changes
        
        Args:
            current_joint_positions: Current robot joint positions
            
        Returns:
            True if robot is moving, False if stationary
        """
        if self.previous_joint_positions is None:
            self.previous_joint_positions = current_joint_positions
            return False
        
        if len(current_joint_positions) != len(self.previous_joint_positions):
            return False
        
        # Calculate joint position differences
        joint_diffs = [
            abs(current - previous) 
            for current, previous in zip(current_joint_positions, self.previous_joint_positions)
        ]
        
        # Check if any joint moved more than threshold
        is_moving = any(diff > self.motion_threshold for diff in joint_diffs)
        
        # Update previous positions
        self.previous_joint_positions = current_joint_positions.copy()
        
        if is_moving:
            self.last_motion_time = time.time()
        
        return is_moving
    
    def _is_valid_image(self, image: np.ndarray) -> bool:
        """
        Check if image is valid (not black/empty)
        
        Args:
            image: RGB image array
            
        Returns:
            True if image contains valid data, False if black/empty
        """
        if image is None or image.size == 0:
            return False
        
        # Check if image has reasonable brightness (not all black)
        mean_brightness = np.mean(image)
        if mean_brightness < self.min_brightness_threshold:
            return False
        
        # Check if image has some variation (not completely uniform)
        std_dev = np.std(image)
        if std_dev < self.min_std_threshold:
            return False
        
        return True
    
    def _motion_recording_loop(self):
        """Main loop for motion data recording (runs in separate thread)"""
        while not self.stop_recording_event.is_set():
            try:
                start_time = time.time()
                
                # Check if recording is paused
                if self.pause_recording_event.is_set():
                    time.sleep(0.1)  # Short sleep when paused
                    continue
                
                # Get robot state
                joint_positions = []
                joint_commands = []
                gripper_state = 0.0
                gripper_command = 0.0
                robot_is_moving = False
                
                if self.robot_interface and hasattr(self.robot_interface, 'get_robot_joint_state'):
                    try:
                        robot_state = self.robot_interface.get_robot_joint_state()
                        if robot_state is not None and len(robot_state) > 0:
                            joint_positions = robot_state[:7] if len(robot_state) >= 7 else robot_state
                            joint_commands = joint_positions.copy()  # Assume commands match positions for now
                            
                            # Get gripper state using dedicated xArm SDK method
                            if hasattr(self.robot_interface, 'arm') and self.robot_interface.arm is not None:
                                try:
                                    gripper_result = self.robot_interface.arm.get_gripper_position()
                                    if gripper_result[0] == 0:  # Success code
                                        gripper_state = gripper_result[1]
                                    else:
                                        gripper_state = 0.0
                                except Exception as e:
                                    gripper_state = 0.0
                            else:
                                gripper_state = 0.0
                            gripper_command = gripper_state
                            
                            # Check if robot is moving
                            robot_is_moving = self.is_robot_moving(joint_positions)
                            
                    except Exception as e:
                        print(f"Error getting robot joint state: {e}")
                
                # Skip recording if robot is not moving and has been stationary for too long
                # Note: With motion tracking hooks, this logic is less critical but kept as backup
                if not robot_is_moving:
                    time_since_motion = time.time() - self.last_motion_time
                    if time_since_motion > self.stationary_timeout and self.last_motion_time > 0:
                        # Robot has been stationary for too long, skip recording
                        time.sleep(self.recording_timestep)
                        continue
                
                # Get camera frames
                rgb_image_path = None
                depth_image_path = None
                
                if self.camera_interface and hasattr(self.camera_interface, 'get_frames'):
                    try:
                        frames = self.camera_interface.get_frames()
                        if frames:
                            rgb_image, depth_image = frames
                            
                            # Save images with timestamp
                            timestamp_str = f"{time.time():.3f}"
                            
                            if rgb_image is not None and self._is_valid_image(rgb_image):
                                rgb_filename = f"{self.current_session_id}_rgb_{timestamp_str}.jpg"
                                rgb_path = self.motion_images_dir / rgb_filename
                                # Save RGB image using PIL to preserve RGB format
                                pil_image = Image.fromarray(rgb_image.astype(np.uint8))
                                pil_image.save(str(rgb_path), 'JPEG', quality=95)
                                rgb_image_path = str(rgb_path)
                            
                            if depth_image is not None:
                                depth_filename = f"{self.current_session_id}_depth_{timestamp_str}.png"
                                depth_path = self.motion_images_dir / depth_filename
                                # Save depth as 16-bit PNG
                                cv2.imwrite(str(depth_path), depth_image.astype(np.uint16))
                                depth_image_path = str(depth_path)
                    
                    except Exception as e:
                        print(f"Error capturing camera frames: {e}")
                
                # Skip recording if no valid camera data (prevents black/empty image records)
                if rgb_image_path is None:
                    time.sleep(self.recording_timestep)
                    continue
                
                # Create motion data record
                motion_record = MotionDataRecord(
                    timestamp=time.time(),
                    joint_positions=joint_positions,
                    joint_commands=joint_commands,
                    gripper_state=gripper_state,
                    gripper_command=gripper_command,
                    rgb_image_path=rgb_image_path,
                    depth_image_path=depth_image_path
                )
                
                # Add to queue for main thread to process
                self.motion_data_queue.put(motion_record)
                
                # Calculate sleep time to maintain timestep
                elapsed = time.time() - start_time
                sleep_time = max(0, self.recording_timestep - elapsed)
                
                if sleep_time > 0:
                    self.stop_recording_event.wait(sleep_time)
            
            except Exception as e:
                print(f"Error in motion recording loop: {e}")
                time.sleep(self.recording_timestep)
    
    def _process_motion_data_queue(self):
        """Process all motion data from queue and add to current record"""
        if not self.current_record:
            return
        
        while not self.motion_data_queue.empty():
            try:
                motion_record = self.motion_data_queue.get_nowait()
                self.current_record.motion_data.append(motion_record)
            except queue.Empty:
                break
    
    def _generate_task_name(self, natural_language_task: str) -> str:
        """Generate a short task name from natural language task"""
        # Take first 50 characters and sanitize
        task_name = natural_language_task[:50].lower()
        task_name = ''.join(c if c.isalnum() or c in ' _-' else '' for c in task_name)
        task_name = '_'.join(task_name.split())  # Replace spaces with underscores
        
        if not task_name:
            task_name = "task"
        
        return task_name
    
    def add_demonstration_timestep(self, 
                                 wrist_image: Optional[np.ndarray] = None,
                                 external_image: Optional[np.ndarray] = None,
                                 joint_positions: Optional[List[float]] = None,
                                 end_effector_pose: Optional[List[float]] = None,
                                 gripper_state: Optional[float] = None,
                                 delta_pose: Optional[List[float]] = None,
                                 gripper_action: Optional[float] = None,
                                 language_instruction: Optional[str] = None):
        """
        Add OpenVLA-compatible demonstration timestep
        
        Args:
            wrist_image: 224x224 RGB wrist camera image
            external_image: 224x224 RGB external camera image (optional)
            joint_positions: 7-DOF xArm joint positions
            end_effector_pose: Current EE pose [x,y,z,rx,ry,rz]
            gripper_state: Current gripper state (0-1 normalized)
            delta_pose: Delta EE movement [dx,dy,dz,drx,dry,drz]
            gripper_action: Target gripper state (0-1 normalized)
            language_instruction: Natural language task instruction
        """
        if not self.current_session_id or not self.current_record:
            print("Warning: No active recording session for demonstration data")
            return False
        
        timestamp = time.time()
        timestamp_str = f"{timestamp:.3f}"
        
        # Save images and get paths
        image_paths = {}
        
        if wrist_image is not None and self._is_valid_image(wrist_image):
            wrist_filename = f"{self.current_session_id}_wrist_{timestamp_str}.jpg"
            wrist_path = self.motion_images_dir / wrist_filename
            # Save RGB image using PIL to preserve RGB format (keep original resolution)
            pil_image = Image.fromarray(wrist_image.astype(np.uint8))
            pil_image.save(str(wrist_path), 'JPEG', quality=95)
            image_paths["wrist_cam"] = str(wrist_path)
        
        if external_image is not None and self._is_valid_image(external_image):
            external_filename = f"{self.current_session_id}_external_{timestamp_str}.jpg"
            external_path = self.motion_images_dir / external_filename
            # Save RGB image using PIL to preserve RGB format (keep original resolution)
            pil_image = Image.fromarray(external_image.astype(np.uint8))
            pil_image.save(str(external_path), 'JPEG', quality=95)
            image_paths["external_cam"] = str(external_path)
        
        # Create robot state
        robot_state = {}
        if joint_positions is not None:
            robot_state["joint_positions"] = joint_positions
        if end_effector_pose is not None:
            robot_state["end_effector_pose"] = end_effector_pose
        if gripper_state is not None:
            robot_state["gripper_state"] = gripper_state
        
        # Create action
        action = {}
        if delta_pose is not None:
            action["delta_pose"] = delta_pose
        if gripper_action is not None:
            action["gripper_action"] = gripper_action
        
        # Use current task instruction if not provided
        if language_instruction is None and self.current_record:
            language_instruction = self.current_record.natural_language_task
        
        # Create demonstration record
        demo_record = OpenVLADemonstrationRecord(
            timestamp=timestamp,
            images=image_paths,
            robot_state=robot_state,
            action=action,
            language_instruction=language_instruction or ""
        )
        
        self.current_record.demonstration_data.append(demo_record)
        return True
    
    def export_openvla_dataset(self, output_path: str = None) -> str:
        """
        Export current session data in OpenVLA-compatible format
        
        Args:
            output_path: Output file path (default: auto-generated)
            
        Returns:
            Path to exported dataset file
        """
        if not self.current_record or len(self.current_record.demonstration_data) == 0:
            raise RuntimeError("No demonstration data available to export")
        
        # Process demonstration data queue first
        self._process_motion_data_queue()
        
        if output_path is None:
            safe_task_name = self._sanitize_filename(self.current_record.task_name)
            output_path = str(self.data_dir / f"openvla_demo_{self.current_session_id}_{safe_task_name}.json")
        
        # Convert demonstration data to OpenVLA format
        openvla_data = {
            "task_description": self.current_record.natural_language_task,
            "episode_data": []
        }
        
        for demo_record in self.current_record.demonstration_data:
            timestep = {
                "images": demo_record.images,
                "robot_state": demo_record.robot_state,
                "action": demo_record.action,
                "language_instruction": demo_record.language_instruction,
                "timestamp": demo_record.timestamp
            }
            openvla_data["episode_data"].append(timestep)
        
        # Save OpenVLA dataset
        with open(output_path, 'w') as f:
            json.dump(openvla_data, f, indent=2, default=self._json_serializer)
        
        print(f"OpenVLA dataset exported: {output_path}")
        print(f"Episode length: {len(openvla_data['episode_data'])} timesteps")
        
        return output_path
    
    def collect_user_feedback(self) -> Tuple[bool, str]:
        """
        Collect user feedback about task execution success
        
        Returns:
            Tuple of (success_rating, user_explanation)
        """
        print("\n" + "="*60)
        print("TASK EXECUTION FEEDBACK")
        print("="*60)
        
        # Get success rating
        while True:
            response = input("\nDid the task execute successfully? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                success_rating = True
                break
            elif response in ['n', 'no']:
                success_rating = False
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
        # Get explanation/notes
        print("\nPlease provide details about the execution:")
        print("(Enter your explanation, then press Enter twice to finish)")
        
        explanation_lines = []
        empty_line_count = 0
        
        while empty_line_count < 2:
            line = input()
            if line.strip() == "":
                empty_line_count += 1
            else:
                empty_line_count = 0
                explanation_lines.append(line)
        
        explanation = "\n".join(explanation_lines).strip()
        
        if not explanation:
            explanation = "No additional details provided."
        
        # Record feedback
        if self.current_record:
            self.current_record.user_success_rating = success_rating
            self.current_record.user_explanation = explanation
        
        print(f"\nFeedback recorded: {'Success' if success_rating else 'Failure'}")
        print(f"Notes: {explanation[:100]}{'...' if len(explanation) > 100 else ''}")
        
        return success_rating, explanation
    
    def finalize_session(self, prompt_for_save: bool = True) -> str:
        """
        Finalize and save the current recording session
        
        Args:
            prompt_for_save: Whether to prompt user before saving motion data
        
        Returns:
            Path to saved session file
        """
        if self.current_session_id is None or self.current_record is None:
            raise RuntimeError("No active recording session to finalize")
        
        # Stop motion recording if still active
        if self.is_recording_motion:
            self.stop_motion_recording()
        
        # Process any remaining motion data
        self._process_motion_data_queue()
        
        # Ask user about saving motion data if recording was enabled
        save_motion_data = True
        if prompt_for_save and len(self.current_record.motion_data) > 0:
            print(f"\nMotion data recorded: {len(self.current_record.motion_data)} samples")
            while True:
                response = input("Do you want to save the motion robot data? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    save_motion_data = True
                    break
                elif response in ['n', 'no']:
                    save_motion_data = False
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
        
        # Clear motion data if user chose not to save
        if not save_motion_data:
            self.current_record.motion_data = []
            print("Motion data cleared from session")
        
        # Collect user feedback
        self.collect_user_feedback()
        
        # Create session file with task name
        safe_task_name = self._sanitize_filename(self.current_record.task_name)
        session_filename = f"{self.current_session_id}_{safe_task_name}.json"
        session_path = self.sessions_dir / session_filename
        
        # Convert record to dictionary for JSON serialization
        record_dict = self._record_to_dict(self.current_record)
        
        # Save session data
        with open(session_path, 'w') as f:
            json.dump(record_dict, f, indent=2, default=self._json_serializer)
        
        # Generate summary
        self._print_session_summary(self.current_record)
        
        # Clear current session
        current_session_id = self.current_session_id
        self.current_session_id = None
        self.current_record = None
        self.timing_data = None
        self.robot_interface = None
        self.camera_interface = None
        
        print(f"\nSession saved to: {session_path}")
        return str(session_path)
    
    def _record_to_dict(self, record: TaskExecutionRecord) -> Dict[str, Any]:
        """Convert TaskExecutionRecord to dictionary for JSON serialization"""
        record_dict = asdict(record)
        
        # Convert timing data
        if record.timing:
            record_dict['timing'] = asdict(record.timing)
        
        # Convert points of interest
        record_dict['points_of_interest'] = [asdict(poi) for poi in record.points_of_interest]
        
        # Convert motion data
        record_dict['motion_data'] = [asdict(motion_record) for motion_record in record.motion_data]
        
        # Convert demonstration data
        record_dict['demonstration_data'] = [asdict(demo_record) for demo_record in record.demonstration_data]
        
        return record_dict
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """
        Sanitize text for use in filename
        
        Args:
            text: Text to sanitize
            max_length: Maximum length of resulting filename part
            
        Returns:
            Sanitized filename-safe text
        """
        import re
        
        # Replace spaces with underscores
        sanitized = text.replace(" ", "_")
        
        # Remove or replace invalid filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', sanitized)
        
        # Remove other non-alphanumeric characters except underscores and hyphens
        sanitized = re.sub(r'[^\w\-_]', '', sanitized)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Ensure it doesn't end with a period (Windows compatibility)
        sanitized = sanitized.rstrip('.')
        
        # Fallback to "task" if sanitization resulted in empty string
        if not sanitized:
            sanitized = "task"
            
        return sanitized.lower()

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and other types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)
    
    def _print_session_summary(self, record: TaskExecutionRecord):
        """Print a summary of the recorded session"""
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Session ID: {record.session_id}")
        print(f"Task Name: {record.task_name}")
        print(f"Task: {record.natural_language_task}")
        print(f"Execution Mode: {record.execution_mode}")
        print(f"Timestamp: {record.timestamp}")
        
        print(f"\nTiming:")
        if record.timing:
            print(f"  Planning: {record.timing.planning_duration:.2f}s")
            print(f"  Inference: {record.timing.inference_duration:.2f}s")
            print(f"  Execution: {record.timing.execution_duration:.2f}s")
            print(f"  Total: {record.timing.total_duration:.2f}s")
        
        print(f"\nTask Decomposition ({len(record.task_decomposition)} commands):")
        for i, cmd in enumerate(record.task_decomposition, 1):
            print(f"  {i}. {cmd}")
        
        print(f"\nSkills Generated: {len(record.skills_generated)}")
        print(f"Points of Interest: {len(record.points_of_interest)}")
        print(f"Stored Skill Files: {len(record.stored_skill_paths)}")
        print(f"Skill Generation Images: {len(record.skill_generation_image_paths)}")
        if record.skill_image_id:
            print(f"Skill Image ID: {record.skill_image_id}")
        
        # Motion data summary
        print(f"\nMotion Data:")
        print(f"  Samples Recorded: {len(record.motion_data)}")
        if len(record.motion_data) > 0:
            print(f"  Recording Timestep: {record.recording_timestep}s")
            duration = record.motion_data[-1].timestamp - record.motion_data[0].timestamp if len(record.motion_data) > 1 else 0
            print(f"  Total Recording Duration: {duration:.2f}s")
            
            # Count RGBD images
            rgb_count = sum(1 for md in record.motion_data if md.rgb_image_path)
            depth_count = sum(1 for md in record.motion_data if md.depth_image_path)
            print(f"  RGB Images: {rgb_count}")
            print(f"  Depth Images: {depth_count}")
        
        # Demonstration data summary
        print(f"\nOpenVLA Demonstration Data:")
        print(f"  Timesteps Recorded: {len(record.demonstration_data)}")
        if len(record.demonstration_data) > 0:
            wrist_images = sum(1 for dd in record.demonstration_data if "wrist_cam" in dd.images)
            external_images = sum(1 for dd in record.demonstration_data if "external_cam" in dd.images)
            print(f"  Wrist Camera Images: {wrist_images}")
            print(f"  External Camera Images: {external_images}")
        
        print(f"\nExecution Success: {record.execution_success}")
        
        if record.failure_messages:
            print(f"Failure Messages: {len(record.failure_messages)}")
            for msg in record.failure_messages:
                print(f"  - {msg}")
        
        if record.robot_errors:
            print(f"Robot Errors: {len(record.robot_errors)}")
            for error in record.robot_errors:
                print(f"  - {error}")
        
        print(f"\nUser Feedback:")
        print(f"  Success Rating: {record.user_success_rating}")
        if record.user_explanation:
            print(f"  Explanation: {record.user_explanation}")
        
        print("="*60)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about recorded sessions"""
        session_files = list(self.sessions_dir.glob("*.json"))
        
        if not session_files:
            return {"total_sessions": 0}
        
        stats = {
            "total_sessions": len(session_files),
            "successful_executions": 0,
            "failed_executions": 0,
            "user_success_rate": 0.0,
            "execution_success_rate": 0.0,
            "average_planning_time": 0.0,
            "average_inference_time": 0.0,
            "average_execution_time": 0.0,
            "most_common_failures": [],
            "sessions_by_date": {}
        }
        
        planning_times = []
        inference_times = []
        execution_times = []
        user_successes = 0
        execution_successes = 0
        failure_messages = []
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Count successes
                if session_data.get('execution_success', False):
                    execution_successes += 1
                else:
                    stats['failed_executions'] += 1
                
                if session_data.get('user_success_rating', False):
                    user_successes += 1
                
                # Collect timing data
                timing = session_data.get('timing', {})
                if timing:
                    if 'planning_duration' in timing or ('planning_end' in timing and 'planning_start' in timing):
                        planning_duration = timing.get('planning_duration', 
                                                     timing.get('planning_end', 0) - timing.get('planning_start', 0))
                        planning_times.append(planning_duration)
                    
                    if 'inference_duration' in timing or ('inference_end' in timing and 'inference_start' in timing):
                        inference_duration = timing.get('inference_duration',
                                                       timing.get('inference_end', 0) - timing.get('inference_start', 0))
                        inference_times.append(inference_duration)
                    
                    if 'execution_duration' in timing or ('execution_end' in timing and 'execution_start' in timing):
                        execution_duration = timing.get('execution_duration',
                                                       timing.get('execution_end', 0) - timing.get('execution_start', 0))
                        execution_times.append(execution_duration)
                
                # Collect failure messages
                failure_messages.extend(session_data.get('failure_messages', []))
                
            except Exception as e:
                print(f"Error reading session file {session_file}: {e}")
        
        # Calculate averages
        stats['successful_executions'] = execution_successes
        stats['execution_success_rate'] = execution_successes / len(session_files) if session_files else 0.0
        stats['user_success_rate'] = user_successes / len(session_files) if session_files else 0.0
        
        if planning_times:
            stats['average_planning_time'] = sum(planning_times) / len(planning_times)
        if inference_times:
            stats['average_inference_time'] = sum(inference_times) / len(inference_times)
        if execution_times:
            stats['average_execution_time'] = sum(execution_times) / len(execution_times)
        
        # Count failure messages
        from collections import Counter
        failure_counter = Counter(failure_messages)
        stats['most_common_failures'] = failure_counter.most_common(5)
        
        return stats