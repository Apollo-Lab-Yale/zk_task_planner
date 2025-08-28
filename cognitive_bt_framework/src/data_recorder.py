#!/usr/bin/env python3
"""
Data Recording System for Task Planner
Records comprehensive data for each task execution run including timing, failures, and user feedback.
"""

import json
import time
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import base64
import uuid
from PIL import Image
import io

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
    task_decomposition: List[str]
    
    # Generated skills and LLM responses
    skills_generated: List[Dict[str, Any]]
    llm_responses: Dict[str, str]  # Store raw LLM responses
    
    # Points of interest and images
    points_of_interest: List[PointOfInterestRecord]
    environment_image_path: str
    surface_images_paths: List[str]
    
    # Skill generation files
    skill_generation_image_paths: List[str]
    stored_skill_paths: List[str]
    skill_image_id: Optional[str]
    
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
    
    def __init__(self, data_dir: str = "task_execution_data"):
        """
        Initialize data recorder
        
        Args:
            data_dir: Directory to store recorded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Current session tracking
        self.current_session_id = None
        self.current_record = None
        self.timing_data = None
        
    def start_recording_session(self, natural_language_task: str, robot_ip: str = "192.168.1.224", 
                               camera_type: str = "realsense", execution_mode: str = "real") -> str:
        """
        Start a new recording session
        
        Args:
            natural_language_task: The original natural language task
            robot_ip: IP address of the robot
            camera_type: Type of camera being used
            execution_mode: "simulation" or "real"
            
        Returns:
            session_id: Unique identifier for this session
        """
        self.current_session_id = str(uuid.uuid4())
        
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
            task_decomposition=[],
            skills_generated=[],
            llm_responses={},
            points_of_interest=[],
            environment_image_path="",
            surface_images_paths=[],
            skill_generation_image_paths=[],
            stored_skill_paths=[],
            skill_image_id=None,
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
        
        print(f"Started recording session: {self.current_session_id}")
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
        
        image_filename = f"{self.current_session_id}_environment.jpg"
        image_path = self.images_dir / image_filename
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        cv2.imwrite(str(image_path), image_bgr)
        
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
            image_filename = f"{self.current_session_id}_surface_{surface_name}.jpg"
            image_path = self.images_dir / image_filename
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            cv2.imwrite(str(image_path), image_bgr)
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
    
    def finalize_session(self) -> str:
        """
        Finalize and save the current recording session
        
        Returns:
            Path to saved session file
        """
        if self.current_session_id is None or self.current_record is None:
            raise RuntimeError("No active recording session to finalize")
        
        # Collect user feedback
        self.collect_user_feedback()
        
        # Create session file
        session_filename = f"{self.current_session_id}.json"
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
        
        return record_dict
    
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
        print(f"Execution Success: {record.execution_success}")
        
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