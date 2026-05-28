#!/usr/bin/env python3
import numpy as np
import time
import argparse
import sys
import logging

# Import the CuRobo motion planner - adjust the import path as needed
from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface import CuRoboMotionPlanner

class MotionCommandTester:
    def __init__(self, robot_ip="192.168.1.224"):
        """
        Initialize the motion command tester
        
        Args:
            robot_ip: IP address of the xArm robot
        """
        print(f'Initializing Motion Command Tester, connecting to robot at {robot_ip}')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize CuRobo motion planner
        print("Initializing CuRobo motion planner...")
        self.motion_planner = CuRoboMotionPlanner(robot_ip=robot_ip)
        print("CuRobo motion planner initialized")
    
    def setup_logging(self):
        """Configure logging to console"""
        self.logger = logging.getLogger('motion_command_tester')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(console_handler)

    def test_gripper_commands(self):
        """Test basic gripper open/close commands"""
        self.logger.info("Testing gripper commands...")
        
        # Test open gripper
        self.logger.info("Opening gripper...")
        success = self.motion_planner.open_gripper(wait=True, timeout=10.0)
        self.logger.info(f"Open gripper {'succeeded' if success else 'failed'}")
        
        # Pause before next command
        time.sleep(2)
        
        # Test close gripper
        self.logger.info("Closing gripper...")
        success = self.motion_planner.close_gripper(wait=True, timeout=10.0)
        self.logger.info(f"Close gripper {'succeeded' if success else 'failed'}")
        
        # Pause before next command
        time.sleep(2)
        
        # Open gripper again to prepare for other tests
        self.motion_planner.open_gripper(wait=True, timeout=10.0)
        return success
    
    def test_retract_gripper(self, distance=0.2, speed=0.3):
        """Test gripper retraction along z-axis"""
        self.logger.info(f"Testing gripper retraction with distance={distance}, speed={speed}...")
        
        # Get current pose
        pose = self.motion_planner.get_robot_tcp_pose()
        if pose is None:
            self.logger.error("Could not get current TCP pose for retraction")
            return False
            
        current_position, current_orientation = pose
        self.logger.info(f"Current position: {current_position}")
        
        # Move back along Z-axis
        target_position = np.array(current_position) + np.array([0, 0, distance])
        self.logger.info(f"Target position after retraction: {target_position}")
        
        success, _, _ = self.motion_planner.move_to_pose(
            target_position=target_position.tolist(),
            target_orientation=current_orientation,
            execute=True,
            speed_factor=speed,
            planning_timeout=10.0
        )
        
        self.logger.info(f"Retract gripper {'succeeded' if success else 'failed'}")
        return success
    
    def test_push_pull(self, is_push=True, distance=0.05, axis='z', speed=0.3):
        """
        Test push/pull motion along specified axis
        
        Args:
            is_push: True for push, False for pull
            distance: Distance to push/pull in meters
            axis: Axis to push/pull along ('x', 'y', or 'z')
            speed: Speed factor for the motion
        """
        action_type = "push" if is_push else "pull"
        self.logger.info(f"Testing {action_type} along {axis}-axis with distance={distance}, speed={speed}...")
        
        # Create surface normal based on selected axis
        surface_normal = None
        if axis.lower() == 'x':
            surface_normal = [1.0, 0.0, 0.0]
        elif axis.lower() == 'y':
            surface_normal = [0.0, 1.0, 0.0]
        else:  # default to z-axis
            surface_normal = [0.0, 0.0, 1.0]
            
        self.logger.info(f"Using surface normal: {surface_normal}")
        
        # Execute the push/pull motion
        success, _, _ = self.motion_planner.plan_push_pull(
            distance=distance,
            is_push=is_push,
            custom_normal=surface_normal,
            move_parallel=False,
            planning_timeout=10.0,
            execute=True,
            speed_factor=speed
        )
        
        self.logger.info(f"{action_type.capitalize()} along {axis}-axis {'succeeded' if success else 'failed'}")
        return success
    
    def test_move_to_pose(self, position_offset=None, speed=0.5):
        """
        Test moving to a specific pose relative to current position
        
        Args:
            position_offset: Offset from current position [x, y, z] in meters
            speed: Speed factor for the motion
        """
        if position_offset is None:
            position_offset = [0.05, 0.0, 0.0]  # Default to small x offset
            
        self.logger.info(f"Testing move to pose with offset={position_offset}, speed={speed}...")
        
        # Get current pose
        pose = self.motion_planner.get_robot_tcp_pose()
        if pose is None:
            self.logger.error("Could not get current TCP pose")
            return False
            
        current_position, current_orientation = pose
        self.logger.info(f"Current position: {current_position}")
        
        # Calculate target position
        target_position = np.array(current_position) + np.array(position_offset)
        self.logger.info(f"Target position: {target_position}")
        
        # Move to target position
        success, _, _ = self.motion_planner.move_to_pose_with_preparation(
            target_position=target_position.tolist(),
            target_orientation=current_orientation,
            execute=True,
            planning_timeout=10.0,
            speed_factor=speed
        )
        
        self.logger.info(f"Move to pose {'succeeded' if success else 'failed'}")
        return success
    
    def test_move_to_home(self):
        """Test moving to home position"""
        self.logger.info("Testing move to home position...")
        success, _, _ = self.motion_planner.move_to_home(execute=True)
        self.logger.info(f"Move to home {'succeeded' if success else 'failed'}")
        return success
    
    def test_get_robot_status(self):
        """Test getting robot status"""
        self.logger.info("Getting robot status...")
        status = self.motion_planner.get_robot_status()
        if status:
            self.logger.info(f"Robot status: Mode={status['mode']}, State={status['state']}")
        else:
            self.logger.error("Failed to get robot status")
        return status
    
    def run_custom_command_sequence(self, commands):
        """
        Run a custom sequence of commands
        
        Args:
            commands: List of (command_name, args_dict) tuples
        """
        self.logger.info(f"Running custom command sequence with {len(commands)} commands")
        
        for i, (cmd_name, args) in enumerate(commands, 1):
            self.logger.info(f"Command {i}/{len(commands)}: {cmd_name} with args {args}")
            
            if cmd_name == 'open_gripper':
                success = self.motion_planner.open_gripper(wait=True, timeout=args.get('timeout', 10.0))
            elif cmd_name == 'close_gripper':
                success = self.motion_planner.close_gripper(wait=True, timeout=args.get('timeout', 10.0))
            elif cmd_name == 'retract_gripper':
                success = self.test_retract_gripper(
                    distance=args.get('distance', 0.2),
                    speed=args.get('speed', 0.3)
                )
            elif cmd_name == 'push_pull':
                success = self.test_push_pull(
                    is_push=args.get('is_push', True),
                    distance=args.get('distance', 0.05),
                    axis=args.get('axis', 'z'),
                    speed=args.get('speed', 0.3)
                )
            elif cmd_name == 'move_to_pose':
                success = self.test_move_to_pose(
                    position_offset=args.get('position_offset', [0.05, 0.0, 0.0]),
                    speed=args.get('speed', 0.5)
                )
            elif cmd_name == 'move_to_home':
                success = self.test_move_to_home()
            elif cmd_name == 'wait':
                time.sleep(args.get('duration', 1.0))
                success = True
            else:
                self.logger.warning(f"Unknown command: {cmd_name}")
                success = False
                
            self.logger.info(f"Command {i} {'succeeded' if success else 'failed'}")
            
            if not success and not args.get('continue_on_failure', False):
                self.logger.error(f"Stopping sequence due to command failure")
                return False
                
            # Pause between commands
            time.sleep(args.get('pause_after', 1.0))
            
        self.logger.info("Custom command sequence completed")
        return True
    
    def shutdown(self):
        """Clean shutdown of the system"""
        self.logger.info("Shutting down motion command tester")
        
        # Disconnect from robot
        if hasattr(self, 'motion_planner'):
            self.motion_planner.disconnect_robot()
            self.logger.info("Robot disconnected")


def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description='Test robot motion commands')
    parser.add_argument('--ip', type=str, default="192.168.1.224", help='Robot IP address')
    parser.add_argument('--test', type=str, default="all", 
                      help='Test to run (gripper, retract, push_x, push_y, push_z, pull_x, pull_y, pull_z, home, status, custom)')
    parser.add_argument('--distance', type=float, default=0.05, help='Distance for push/pull/retract (meters)')
    parser.add_argument('--speed', type=float, default=0.3, help='Speed factor for motions')
    
    args = parser.parse_args()
    
    try:
        # Initialize the motion command tester
        tester = MotionCommandTester(robot_ip=args.ip)
        
        # Check robot status first
        status = tester.test_get_robot_status()
        if not status:
            print("Failed to get robot status, aborting tests")
            return
        
        # Run the selected test
        if args.test == 'all':
            print("Running all basic tests...")
            # tester.test_gripper_commands()
            # time.sleep(1)
            # tester.test_retract_gripper(distance=args.distance, speed=args.speed)
            # time.sleep(1)
            tester.test_push_pull(is_push=False, distance=args.distance, axis='x', speed=args.speed)
            time.sleep(1)
            tester.test_move_to_home()
            
        elif args.test == 'gripper':
            tester.test_gripper_commands()
            
        elif args.test == 'retract':
            tester.test_retract_gripper(distance=args.distance, speed=args.speed)
            
        elif args.test.startswith('push_'):
            axis = args.test[-1].lower()
            tester.test_push_pull(is_push=True, distance=args.distance, axis=axis, speed=args.speed)
            
        elif args.test.startswith('pull_'):
            axis = args.test[-1].lower()
            tester.test_push_pull(is_push=False, distance=args.distance, axis=axis, speed=args.speed)
            
        elif args.test == 'home':
            tester.test_move_to_home()
            
        elif args.test == 'status':
            tester.test_get_robot_status()
            
        elif args.test == 'custom':
            # Example custom command sequence
            commands = [
                ('move_to_home', {}),
                ('open_gripper', {}),
                ('wait', {'duration': 1.0}),
                ('push_pull', {'is_push': True, 'distance': 0.05, 'axis': 'x', 'speed': 0.3}),
                ('wait', {'duration': 1.0}),
                ('push_pull', {'is_push': False, 'distance': 0.05, 'axis': 'x', 'speed': 0.3}),
                ('wait', {'duration': 1.0}),
                ('retract_gripper', {'distance': 0.1, 'speed': 0.3}),
                ('close_gripper', {}),
                ('move_to_home', {})
            ]
            tester.run_custom_command_sequence(commands)
            
        else:
            print(f"Unknown test: {args.test}")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'tester' in locals():
            tester.shutdown()


if __name__ == '__main__':
    main()