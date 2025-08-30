#!/usr/bin/env python3
"""
OpenVLA Teleoperation Data Collection Script

This script collects demonstration data in OpenVLA format for training robotic manipulation models.
It captures:
- Wrist camera images (224x224 RGB)
- External camera images (224x224 RGB, optional)  
- 7-DOF xArm joint positions
- End effector poses
- Gripper states
- Delta actions (7-DOF: delta xyz, delta roll/pitch/yaw, gripper)
- Natural language instructions

Usage:
    python openvla_teleoperation_collector.py --task "pick up the red cube" --episodes 10
"""

import sys
import argparse
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# Add the cognitive_bt_framework to Python path
sys.path.append(str(Path(__file__).parent / "cognitive_bt_framework" / "src"))

from data_recorder import DataRecorder
from robot_interface.xarm_curobo_interface import CuRoboMotionPlanner
from vision.realsense import Camera as RealSenseCamera

class OpenVLATeleopCollector:
    """OpenVLA-compatible teleoperation data collector"""
    
    def __init__(self, robot_ip: str = "192.168.1.224"):
        """
        Initialize the teleoperation data collector
        
        Args:
            robot_ip: xArm robot IP address
        """
        self.robot_ip = robot_ip
        
        # Initialize components
        self.data_recorder = DataRecorder(recording_timestep=0.1)
        self.robot_interface = None
        self.wrist_camera = None
        self.external_camera = None
        
        # State tracking
        self.current_session_id = None
        self.previous_ee_pose = None
        self.episode_count = 0
        
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_hardware(self) -> bool:
        """
        Initialize robot and camera hardware
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize robot interface
            self.logger.info("Initializing robot interface...")
            self.robot_interface = CuRoboMotionPlanner(
                robot_ip=self.robot_ip,
                urdf_path=None,  # Use default
                use_real_robot=True
            )
            
            # Initialize wrist camera (primary)
            self.logger.info("Initializing wrist camera...")
            self.wrist_camera = RealSenseCamera(
                width=640, height=480, fps=30,
                debug=False
            )
            if not self.wrist_camera.start():
                self.logger.error("Failed to start wrist camera")
                return False
            
            # Initialize external camera (optional secondary camera)
            try:
                self.logger.info("Initializing external camera...")
                self.external_camera = RealSenseCamera(
                    width=640, height=480, fps=30,
                    debug=False,
                    device_serial=None  # Will use second camera if available
                )
                if not self.external_camera.start():
                    self.logger.warning("External camera not available, continuing with wrist camera only")
                    self.external_camera = None
            except Exception as e:
                self.logger.warning(f"External camera initialization failed: {e}")
                self.external_camera = None
            
            self.logger.info("Hardware initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            return False
    
    def get_robot_state(self) -> Optional[dict]:
        """
        Get current robot state
        
        Returns:
            Dictionary with joint positions, end effector pose, and gripper state
        """
        try:
            # Get joint positions
            joint_positions = self.robot_interface.get_robot_joint_state()
            if joint_positions is None:
                return None
            
            # Get end effector pose using forward kinematics
            # This would need to be implemented in the robot interface
            # For now, we'll use a placeholder
            end_effector_pose = [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]  # [x,y,z,rx,ry,rz]
            
            # Get gripper state (normalized 0-1)
            gripper_state = 0.5  # Placeholder - needs actual gripper state
            
            return {
                "joint_positions": joint_positions[:7] if len(joint_positions) >= 7 else joint_positions,
                "end_effector_pose": end_effector_pose,
                "gripper_state": gripper_state
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get robot state: {e}")
            return None
    
    def get_camera_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get camera images
        
        Returns:
            Tuple of (wrist_image, external_image) - external may be None
        """
        wrist_image = None
        external_image = None
        
        try:
            # Get wrist camera image
            if self.wrist_camera:
                frames = self.wrist_camera.get_frames()
                if frames:
                    wrist_image, _ = frames  # RGB, depth
            
            # Get external camera image
            if self.external_camera:
                frames = self.external_camera.get_frames()
                if frames:
                    external_image, _ = frames  # RGB, depth
            
        except Exception as e:
            self.logger.error(f"Failed to get camera images: {e}")
        
        return wrist_image, external_image
    
    def compute_delta_action(self, current_pose: List[float], target_pose: List[float]) -> List[float]:
        """
        Compute delta action from current to target pose
        
        Args:
            current_pose: Current end effector pose [x,y,z,rx,ry,rz]
            target_pose: Target end effector pose [x,y,z,rx,ry,rz]
            
        Returns:
            Delta action [dx,dy,dz,drx,dry,drz]
        """
        if len(current_pose) != 6 or len(target_pose) != 6:
            return [0.0] * 6
        
        delta = [target - current for target, current in zip(target_pose, current_pose)]
        return delta
    
    def collect_episode(self, task_instruction: str, episode_id: int) -> bool:
        """
        Collect a single demonstration episode
        
        Args:
            task_instruction: Natural language task description
            episode_id: Episode identifier
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Starting episode {episode_id}: {task_instruction}")
        
        # Start recording session
        self.current_session_id = self.data_recorder.start_recording_session(
            natural_language_task=task_instruction,
            task_name=f"episode_{episode_id}",
            robot_ip=self.robot_ip,
            camera_type="realsense_wrist",
            execution_mode="teleoperation"
        )
        
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_id}: {task_instruction}")
        print(f"{'='*60}")
        print("Instructions:")
        print("1. Move the robot using teaching mode or teleoperation interface")
        print("2. Press SPACE to record current timestep")
        print("3. Press 'q' to finish episode")
        print("4. Press 'r' to restart episode")
        print(f"{'='*60}")
        
        timestep_count = 0
        
        try:
            while True:
                # Get user input
                key = input(f"Timestep {timestep_count} - [SPACE: record, q: quit, r: restart]: ").strip().lower()
                
                if key == 'q':
                    if timestep_count > 0:
                        print(f"Episode completed with {timestep_count} timesteps")
                        break
                    else:
                        print("No timesteps recorded. Episode canceled.")
                        return False
                
                elif key == 'r':
                    print("Restarting episode...")
                    timestep_count = 0
                    self.data_recorder.current_record.demonstration_data = []
                    continue
                
                elif key == ' ' or key == '':
                    # Record current timestep
                    success = self.record_timestep(task_instruction)
                    if success:
                        timestep_count += 1
                        print(f"Recorded timestep {timestep_count}")
                    else:
                        print("Failed to record timestep")
                
                else:
                    print("Invalid input. Use SPACE to record, 'q' to quit, 'r' to restart")
            
            # Export OpenVLA dataset
            if timestep_count > 0:
                dataset_path = self.data_recorder.export_openvla_dataset()
                print(f"Episode dataset saved: {dataset_path}")
            
            # Finalize session
            session_path = self.data_recorder.finalize_session(prompt_for_save=False)
            print(f"Session saved: {session_path}")
            
            return timestep_count > 0
            
        except KeyboardInterrupt:
            print("\nEpisode interrupted by user")
            return False
        
        except Exception as e:
            self.logger.error(f"Episode collection failed: {e}")
            return False
    
    def record_timestep(self, task_instruction: str) -> bool:
        """
        Record a single timestep
        
        Args:
            task_instruction: Natural language task description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get robot state
            robot_state = self.get_robot_state()
            if robot_state is None:
                self.logger.error("Failed to get robot state")
                return False
            
            # Get camera images
            wrist_image, external_image = self.get_camera_images()
            if wrist_image is None:
                self.logger.error("Failed to get wrist camera image")
                return False
            
            # Compute delta action (for now, use zero deltas - in real teleoperation,
            # this would be computed from the next commanded pose)
            current_pose = robot_state["end_effector_pose"]
            target_pose = current_pose.copy()  # Placeholder - would be actual target
            delta_pose = self.compute_delta_action(current_pose, target_pose)
            
            # Gripper action (placeholder - would be actual commanded gripper state)
            gripper_action = robot_state["gripper_state"]
            
            # Add demonstration timestep
            success = self.data_recorder.add_demonstration_timestep(
                wrist_image=wrist_image,
                external_image=external_image,
                joint_positions=robot_state["joint_positions"],
                end_effector_pose=robot_state["end_effector_pose"],
                gripper_state=robot_state["gripper_state"],
                delta_pose=delta_pose,
                gripper_action=gripper_action,
                language_instruction=task_instruction
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to record timestep: {e}")
            return False
    
    def collect_dataset(self, task_instruction: str, num_episodes: int) -> bool:
        """
        Collect multiple demonstration episodes
        
        Args:
            task_instruction: Natural language task description
            num_episodes: Number of episodes to collect
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialize_hardware():
            return False
        
        successful_episodes = 0
        
        for episode_id in range(1, num_episodes + 1):
            print(f"\n--- Episode {episode_id}/{num_episodes} ---")
            
            try:
                if self.collect_episode(task_instruction, episode_id):
                    successful_episodes += 1
                    self.logger.info(f"Episode {episode_id} completed successfully")
                else:
                    self.logger.warning(f"Episode {episode_id} failed or was canceled")
                
                # Ask user if they want to continue
                if episode_id < num_episodes:
                    continue_collection = input(f"Continue to episode {episode_id + 1}? [y/N]: ").strip().lower()
                    if continue_collection not in ['y', 'yes']:
                        break
                        
            except KeyboardInterrupt:
                print(f"\nData collection interrupted at episode {episode_id}")
                break
        
        print(f"\nData collection complete!")
        print(f"Successful episodes: {successful_episodes}/{episode_id}")
        
        return successful_episodes > 0
    
    def shutdown(self):
        """Clean up resources"""
        try:
            if self.wrist_camera:
                self.wrist_camera.stop()
            if self.external_camera:
                self.external_camera.stop()
            if self.robot_interface:
                # Robot interface cleanup if needed
                pass
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

def main():
    parser = argparse.ArgumentParser(description="OpenVLA Teleoperation Data Collection")
    parser.add_argument("--task", type=str, required=True,
                        help="Natural language task description")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to collect")
    parser.add_argument("--robot_ip", type=str, default="192.168.1.224",
                        help="xArm robot IP address")
    
    args = parser.parse_args()
    
    collector = OpenVLATeleopCollector(robot_ip=args.robot_ip)
    
    try:
        success = collector.collect_dataset(args.task, args.episodes)
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"Collection failed: {e}")
        exit_code = 1
    finally:
        collector.shutdown()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()