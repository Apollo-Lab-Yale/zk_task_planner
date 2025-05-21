import numpy as np
import time
import torch
import math
import threading
from typing import List, Dict, Optional, Tuple, Union, Any
import os
import yaml

# CuRobo imports
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision
from curobo.geom.types import Cuboid, WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.robot import RobotConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen, 
    MotionGenConfig, 
    MotionGenPlanConfig, 
    MotionGenResult, 
    MotionGenStatus
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# xArm SDK import
from xarm.wrapper import XArmAPI


class RobotConfig:
    def __init__(self):
        self.dof = 6
        self.robot_type = "lite"
        self.prefix = ""
        self.arm_group_name = ""
        self.gripper_group_name = ""
        self.tcp_link = ""
        self.is_lite6 = True
        self.robot_ip = ""
        self.robot_cfg = None


class CuRoboMotionPlanner:
    def __init__(self, config=None, robot_ip=None):
        """
        Initialize CuRobo motion planner with xArm SDK integration
        
        Args:
            config: Optional configuration object
            robot_ip: IP address of the xArm robot
        """
        # Initialize configuration
        self.config = config or self.initialize_default_config()
        
        # Set robot IP if provided
        if robot_ip is not None:
            self.config.robot_ip = robot_ip
        
        # Initialize TensorDeviceType for cuRobo
        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        
        # Store collision objects
        self.collision_objects = []
        self.collision_object_lock = threading.Lock()
        
        # Current robot state
        self.current_joints = None
        self.joint_state_lock = threading.Lock()
        
        # Initialize xArm SDK
        self.arm = None
        self.arm_lock = threading.Lock()
        self.connect_robot()
        
        # Register callback for joint state updates
        if self.arm is not None:
            self.arm.register_report_location_callback(self.joint_state_callback)
        
        # Initialize cuRobo motion generator
        self.motion_gen = self.init_curobo()
        
        # Initialize IK solver
        self.ik_solver = self.init_ik_solver()
        
        print('CuRobo Motion Planner initialized with xArm SDK')
    
    def initialize_default_config(self) -> RobotConfig:
        """Initialize default configuration for the planner"""
        config = RobotConfig()
        
        # Set default parameters
        config.dof = 7
        config.robot_type = "xarm"
        config.prefix = ""
        config.robot_ip = "192.168.1.224"
        
        # Check if it's a Lite6 robot
        config.is_lite6 = (config.robot_type == "lite" and config.dof == 6)
        
        # Calculate planning group
        config.arm_group_name = f"{config.prefix}{config.robot_type}{config.dof}"
        
        # Similarly for gripper group
        config.gripper_group_name = f"{config.prefix}{config.robot_type}_gripper"
        config.tcp_link = f"{config.prefix}link_tcp"
        
        print(
            f"Configuration loaded: robot_type={config.robot_type}, dof={config.dof}, "
            f"is_lite6={config.is_lite6}, robot_ip={config.robot_ip}"
        )
        return config
    
    def connect_robot(self):
        """Connect to the physical robot using xArm SDK"""
        try:
            with self.arm_lock:
                if self.config.robot_ip:
                    print(f"Connecting to robot at {self.config.robot_ip}")
                    self.arm = XArmAPI(self.config.robot_ip)
                    
                    # Set default parameters
                    self.arm.motion_enable(enable=True)
                    self.arm.set_mode(0)  # Position control mode
                    code = self.arm.set_state(state=0)  # Start state
                    print(f"Set arm state: {code}")
                    # Get current joint states
                    code, angles = self.arm.get_servo_angle(is_radian=True)
                    if code == 0 and angles is not None:
                        self.set_current_joint_state(angles)
                        print(f"Current joint state: {angles}")
                    
                    print(f"Robot connected successfully. Mode: {self.arm.mode}, State: {self.arm.state}")
                else:
                    print("No robot IP configured, running in simulation mode")
        except Exception as e:
            print(f"Failed to connect to the robot: {str(e)}")
            self.arm = None
    
    def joint_state_callback(self, data):
        """Callback function for joint state updates from the robot
        
        Args:
            data: Robot state data from xArm SDK
        """
        if data and len(data) > 7:  # Make sure we have joint data
            # xArm SDK reports joint angles in the data
            joint_angles = data[:self.config.dof]  # xArm Lite6 has 6 joints
            with self.joint_state_lock:
                self.current_joints = np.array(joint_angles)
    
    def init_curobo(self):
        """Initialize cuRobo motion generator"""
        try:
            # Get the robot configs path from cuRobo
            robot_configs_path = get_robot_configs_path()
            print(f"Robot configs path: {robot_configs_path}")
            
            # Create path for XArm Lite6 config
            xarm_config_dir = os.path.join(robot_configs_path, f"xarm" + "_lite6" if self.config.is_lite6 else f"{self.config.dof}")
            os.makedirs(xarm_config_dir, exist_ok=True)
            filename = "xarm_lite6" if self.config.is_lite6 else f"{self.config.robot_type}{self.config.dof}"
            # Define robot configuration file path
            robot_config_path = os.path.join(robot_configs_path, filename + ".yml")
            
            # Basic world configuration with just a table below the robot
            world_config = {
                "cuboid": {
                    "table": {
                        "dims": [1.0, 1.0, 0.05],  # x, y, z
                        "pose": [0.0, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
                    }
                }
            }
            
            # Set up motion generator configuration with optimized parameters
            print(f"Loading motion generator config from: {robot_config_path}")
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_cfg=robot_config_path,
                world_model=world_config,
                tensor_args=self.tensor_args,
                # Planning parameters
                interpolation_dt=0.01,
                # trajopt_tsteps=32,
                interpolation_steps=5000,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                # Optimization parameters
                num_ik_seeds=32,
                num_graph_seeds=4,
                num_trajopt_seeds=6,
                grad_trajopt_iters=300,
                # Time parameters
                trajopt_dt=0.5,
                js_trajopt_dt=0.5,
                # Collision parameters
                collision_activation_distance=0.03,  # 3cm activation distance
                collision_max_outside_distance=0.5,
                # Quality parameters
                evaluate_interpolated_trajectory=True,
                position_threshold=0.005,  # 5mm position accuracy
                rotation_threshold=0.05,
                cspace_threshold=0.05,
                # Performance parameters
                use_cuda_graph=False,  # Set to True for faster performance once working
                store_debug_in_result=True,
                minimum_trajectory_dt=0.01,  # 1ms minimum timestep
                finetune_dt_scale=0.85,
                # Smoothness parameters
                minimize_jerk=True,
                filter_robot_command=True
            )
            
            # Store robot config for IK solver - use the imported RobotConfig, not our class
            # Load the robot config from the YAML file for IK solver
            from curobo.types.robot import RobotConfig as CuroboRobotConfig
            
            self.config.robot_cfg = CuroboRobotConfig.from_dict(
                load_yaml(robot_config_path)["robot_cfg"],
                self.tensor_args
            )
            
            # Create motion generator
            print("Initializing cuRobo motion generator...")
            motion_gen = MotionGen(motion_gen_config)
            
            # Warmup the motion generator
            motion_gen.warmup()
            
            print("cuRobo motion generator initialized successfully")
            return motion_gen
        except Exception as e:
            print(f"Failed to initialize cuRobo motion generator: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def init_ik_solver(self):
        """Initialize cuRobo IK solver"""
        try:
            if not hasattr(self.config, "robot_cfg") or self.config.robot_cfg is None:
                print("Robot configuration not available for IK solver")
                return None
                
            # Create IK solver configuration
            ik_config = IKSolverConfig.load_from_robot_config(
                self.config.robot_cfg,
                self.motion_gen.world_collision.world_model if self.motion_gen and self.motion_gen.world_collision else None,
                rotation_threshold=0.05,
                position_threshold=0.005,
                num_seeds=32,
                self_collision_check=True,
                self_collision_opt=True,
                tensor_args=self.tensor_args,
                use_cuda_graph=True  # For better performance
            )
            
            # Create IK solver
            ik_solver = IKSolver(ik_config)
            
            print("cuRobo IK solver initialized successfully")
            return ik_solver
        except Exception as e:
            print(f"Failed to initialize cuRobo IK solver: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    
    def set_current_joint_state(self, joint_positions):
        """Set current joint state for planning
        
        Args:
            joint_positions: Array of joint positions in radians
        """
        with self.joint_state_lock:
            self.current_joints = np.array(joint_positions[:self.config.dof])
    
    def get_robot_joint_state(self):
        """Get current joint state from the physical robot
        
        Returns:
            numpy.ndarray: Joint positions in radians or None if not available
        """
        if self.arm is None:
            print("Robot not connected")
            return None
            
        try:
            with self.arm_lock:
                code, angles = self.arm.get_servo_angle(is_radian=True)
                if code == 0 and angles is not None:
                    # Update our internal state
                    self.set_current_joint_state(angles)
                    return np.array(angles)
                else:
                    print(f"Failed to get joint state, error code: {code}")
                    return None
        except Exception as e:
            print(f"Error getting robot joint state: {str(e)}")
            return None
    
    def get_robot_tcp_pose(self):
        """Get current TCP position and orientation from the physical robot
        
        Returns:
            tuple: (position, orientation) or None if not available
        """
        if self.arm is None:
            print("Robot not connected")
            return None
            
        try:
            with self.arm_lock:
                code, pose = self.arm.get_position(is_radian=True)
                if code == 0 and pose is not None:
                    # XArm SDK returns TCP pose as [x, y, z, roll, pitch, yaw]
                    position = pose[:3]
                    # Convert Euler angles to quaternion
                    roll, pitch, yaw = pose[3:]
                    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                    orientation = [qw, qx, qy, qz]
                    return position, orientation
                else:
                    print(f"Failed to get TCP pose, error code: {code}")
                    return None
        except Exception as e:
            print(f"Error getting robot TCP pose: {str(e)}")
            return None
    
    def execute_trajectory(self, trajectory, dt, speed_factor=1.0):
        """Execute a trajectory on the physical robot with improved error handling
        
        Args:
            trajectory: Joint trajectory as list of JointState objects
            dt: Time step between trajectory points in seconds
            speed_factor: Factor to scale execution speed (>1 is faster)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        if not trajectory or len(trajectory) == 0:
            print("Empty trajectory provided")
            return False
            
        try:
                # Check robot status before execution
                print(f"Robot status before execution: Mode={self.arm.mode}, State={self.arm.state}")
                
                # Clear any errors first
                if self.arm.has_error or self.arm.has_warn:
                    print("Clearing robot errors before trajectory execution...")
                    self.clean_robot_error()
                    time.sleep(0.5)  # Give robot time to reset
                
                # Ensure robot is in position control mode
                self.arm.set_mode(0)  # Position control mode
                self.arm.set_state(0)  # Ready state
                time.sleep(0.1)
                
                # Verify robot is ready
                if self.arm.mode != 0:
                    print(f"Robot not in ready state (state: {self.arm.state}), aborting trajectory")
                    return False
                
                # Set trajectory parameters
                adjusted_dt = dt / speed_factor  # Adjust time step based on speed factor
                
                print(f"Executing trajectory with {len(trajectory)} points, dt={adjusted_dt:.4f}s")
                
                # Convert trajectory to list of joint positions
                joint_positions_list = []
                if type(trajectory) is not CuroboJointState:
                    trajectory = trajectory.tolist()
                
                for i, point in enumerate(trajectory.position):
                    try:
                        if hasattr(point, 'position'):
                            # Extract joint positions from JointState
                            if point.position.dim() > 1:
                                joint_pos = point.position[0].cpu().numpy()  # Take first batch element
                            else:
                                joint_pos = point.position.cpu().numpy()
                        else:
                            # Handle raw tensor
                            if point.dim() > 1:
                                joint_pos = point[0].cpu().numpy()
                            else:
                                joint_pos = point.cpu().numpy()
                        
                        # Ensure we have the right number of joints
                        if len(joint_pos) != self.config.dof:
                            print(f"Point {i}: Expected {self.config.dof} joints, got {len(joint_pos)}")
                            continue
                            
                        joint_positions_list.append(joint_pos.tolist())
                        
                    except Exception as e:
                        print(f"Error processing trajectory point {i}: {e}")
                        continue
                
                if len(joint_positions_list) == 0:
                    print("No valid trajectory points found")
                    return False
                    
                print(f"Processed {len(joint_positions_list)} valid trajectory points")
                
                # Execute each point in the trajectory
                start_time = time.time()
                print(joint_positions_list)
                for i, joint_pos in enumerate(joint_positions_list):
                    # Check if robot is still in running state
                    
                    # Check timing and wait if needed
                    elapsed = time.time() - start_time
                    target_time = i * adjusted_dt
                    
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
                    
                    # Send joint command to the robot
                    try:
                        # Use set_servo_angle for smoother motion
                        print(f"sending joint pose {joint_pos}, to joint {i}")
                        code = self.arm.set_servo_angle(angle=joint_pos, is_radian=True, wait=False)
                        if code != 0:
                            print(f"Failed to send joint command at point {i}, error code: {code}")
                            return False
                            
                    except Exception as e:
                        print(f"Error sending joint command at point {i}: {e}")
                        return False
                    
                    # Print progress every 10 points or at milestones
                    if i % max(1, len(joint_positions_list) // 10) == 0 or i == len(joint_positions_list) - 1:
                        print(f"Progress: {i+1}/{len(joint_positions_list)} ({100*(i+1)/len(joint_positions_list):.1f}%)")
                
                print(f"Trajectory execution completed in {time.time() - start_time} seconds")
                
                # Update our internal state to the final position
                if len(joint_positions_list) > 0:
                    self.set_current_joint_state(joint_positions_list[-1])
                
                return True
                    
        except Exception as e:
            print(f"Error executing trajectory: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Try to set the robot back to a safe mode/state
            try:
                self.arm.set_mode(0)  # Position control mode
                self.arm.set_state(0)  # Ready state
            except:
                pass
            return False
    
    def set_robot_mode(self, mode):
        """Set the robot control mode
        
        Args:
            mode: 0 for position control, 1 for servo mode, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                code = self.arm.set_mode(mode)
                if code == 0:
                    print(f"Robot mode set to {mode}")
                    return True
                else:
                    print(f"Failed to set robot mode, error code: {code}")
                    return False
        except Exception as e:
            print(f"Error setting robot mode: {str(e)}")
            return False
    
    def set_robot_state(self, state):
        """Set the robot state
        
        Args:
            state: 0 for ready state, 4 for emergency stop, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                code = self.arm.set_state(state)
                if code == 0:
                    print(f"Robot state set to {state}")
                    return True
                else:
                    print(f"Failed to set robot state, error code: {code}")
                    return False
        except Exception as e:
            print(f"Error setting robot state: {str(e)}")
            return False
    
    def emergency_stop(self):
        """Stop the robot immediately"""
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                self.arm.emergency_stop()
                print("Emergency stop triggered")
                return True
        except Exception as e:
            print(f"Error triggering emergency stop: {str(e)}")
            return False
    
    def set_robot_tcp_pose(self, position, orientation=None, wait=True, speed=100, mvacc=500):
        """Move the robot directly to a target TCP pose using the xArm SDK
        
        Args:
            position: [x, y, z] target position
            orientation: [roll, pitch, yaw] or None to keep current
            wait: Whether to wait for motion completion
            speed: Movement speed (mm/s)
            mvacc: Acceleration (mm/s²)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                # Set to position control mode
                self.arm.set_mode(0)
                self.arm.set_state(0)
                
                # Get current pose if orientation not provided
                if orientation is None:
                    code, current_pose = self.arm.get_position()
                    if code == 0:
                        orientation = current_pose[3:]
                    else:
                        print(f"Failed to get current pose, error code: {code}")
                        return False
                
                # Create target pose
                target_pose = position + orientation
                
                # Move to target pose
                print(f"Moving to TCP pose: {target_pose}")
                code = self.arm.set_position(*target_pose, speed=speed, mvacc=mvacc, wait=wait, is_radian=True)
                
                if code == 0:
                    print("TCP pose move completed successfully")
                    return True
                else:
                    print(f"Failed to move to TCP pose, error code: {code}")
                    return False
        except Exception as e:
            print(f"Error moving to TCP pose: {str(e)}")
            return False
    
    def set_robot_joint_angles(self, joint_angles, wait=True, speed=20, acc=500):
        """Move the robot directly to target joint angles using the xArm SDK
        
        Args:
            joint_angles: List of joint angles in radians
            wait: Whether to wait for motion completion
            speed: Joint speed (rad/s)
            acc: Joint acceleration (rad/s²)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                # Set to position control mode
                self.arm.set_mode(0)
                self.arm.set_state(0)
                
                # Move to target joint angles
                print(f"Moving to joint angles: {joint_angles}")
                code = self.arm.set_servo_angle(joint_angles, speed=speed, mvacc=acc, wait=wait, is_radian=True)
                
                if code == 0:
                    print("Joint move completed successfully")
                    # Update our internal state
                    self.set_current_joint_state(joint_angles)
                    return True
                else:
                    print(f"Failed to move to joint angles, error code: {code}")
                    return False
        except Exception as e:
            print(f"Error moving to joint angles: {str(e)}")
            return False
    
    def add_collision_object(self, name, dimensions, position, orientation=[1.0, 0.0, 0.0, 0.0]):
        """Add a collision object to the planning scene
        
        Args:
            name: Unique identifier for the collision object
            dimensions: [x, y, z] dimensions of the collision object
            position: [x, y, z] position of the collision object
            orientation: [w, x, y, z] quaternion orientation of the collision object
        """
        with self.collision_object_lock:
            # Create cuRobo cuboid
            cuboid = Cuboid(
                name=name,
                pose = position + orientation,
                dims=dimensions
            )
            
            self.collision_objects.append(cuboid)
            self.update_world_config()
            print(f'Added box collision object: {name}')
    
    def add_collision_mesh(self, name, file_path, position, orientation=[1.0, 0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]):
        """Add a collision mesh to the planning scene
        
        Args:
            name: Unique identifier for the collision object
            file_path: Path to the mesh file
            position: [x, y, z] position of the collision object
            orientation: [w, x, y, z] quaternion orientation of the collision object
            scale: [x, y, z] scale factors for the mesh
        """
        with self.collision_object_lock:
            # Create cuRobo mesh
            mesh = Mesh(
                name=name,
                file_path=file_path,
                position=position,
                orientation=orientation,
                scale=scale
            )
            
            self.collision_objects.append(mesh)
            self.update_world_config()
            print(f'Added mesh collision object: {name}')
    
    def update_world_config(self):
        """Create a world configuration with all collision objects and update both motion generator and IK solver"""
        if not self.motion_gen:
            print("Motion generator not initialized")
            return
            
        try:
            # Create a new world config with the current collision objects
            world_config = WorldConfig(self.collision_objects)
            
            # Update the motion generator with the new world configuration
            self.motion_gen.update_world(world_config)
            
            # Reinitialize IK solver with the new world configuration
            self.ik_solver = self.init_ik_solver()
            
            print('Updated world configuration with collision objects')
        except Exception as e:
            print(f"Failed to update world configuration: {str(e)}")
    
    def clear_collision_objects(self):
        """Clear all collision objects from the world"""
        with self.collision_object_lock:
            self.collision_objects = []
            if self.motion_gen and self.motion_gen.world_collision:
                self.motion_gen.world_collision.clear_cache()
                
                # Reinitialize IK solver after clearing
                self.ik_solver = self.init_ik_solver()
                
                print('Cleared world collision cache')
    
    def compute_collision_free_ik(self, position, orientation=None, num_seeds=32, max_attempts=5):
        """Compute collision-free inverse kinematics
        
        Args:
            position: [x, y, z] target position
            orientation: [w, x, y, z] quaternion orientation, or None for top-down
            num_seeds: Number of IK seeds to try
            max_attempts: Maximum number of attempts with random perturbations
            
        Returns:
            numpy.ndarray: Joint positions or None if no solution
        """
        try:
            if self.ik_solver is None:
                print("IK solver not initialized")
                return None
                
            # Use top-down orientation if not provided
            if orientation is None:
                orientation = self.create_top_down_orientation()
                print("Using top-down orientation for IK")
                
            # Create goal pose
            goal_pose = CuroboPose(
                position=self.tensor_args.to_device([position]),
                quaternion=self.tensor_args.to_device([orientation])
            )
            
            # Sample configs for seeds
            q_sample = self.ik_solver.sample_configs(num_seeds)
            
            # Solve IK first attempt
            result = self.ik_solver.solve_batch(goal_pose, seed_config=q_sample)
            
            if torch.any(result.success):
                # Get the first successful solution
                joint_positions = result.solution[result.success][0].cpu().numpy()
                print(f"Collision-free IK solution found with position error: "
                      f"{result.position_error[result.success][0].item():.6f}m")
                return joint_positions
                
            # If no collision-free solution found, try with small perturbations
            perturbation_scale = 0.01  # 1cm perturbation
            for attempt in range(max_attempts):
                # Increase perturbation scale with each attempt
                current_scale = perturbation_scale * (attempt + 1)
                
                # Generate random perturbation
                position_perturb = np.array(position) + np.random.uniform(-current_scale, current_scale, 3)
                
                # Try IK with perturbed position
                print(f"Trying IK with perturbed position (attempt {attempt+1}/{max_attempts})")
                
                # Create perturbed goal pose
                perturbed_goal_pose = CuroboPose(
                    position=self.tensor_args.to_device([position_perturb]),
                    quaternion=self.tensor_args.to_device([orientation])
                )
                
                # Sample new configs for seeds
                q_sample = self.ik_solver.sample_configs(num_seeds)
                
                # Solve IK with perturbed position
                result = self.ik_solver.solve_batch(perturbed_goal_pose, seed_config=q_sample)
                
                if torch.any(result.success):
                    # Get the first successful solution
                    joint_positions = result.solution[result.success][0].cpu().numpy()
                    print(f"Collision-free IK solution found with perturbed position, error: "
                          f"{result.position_error[result.success][0].item():.6f}m")
                    return joint_positions
                    
            # If we get here, no collision-free solution was found
            print("No collision-free IK solution found after all attempts")
            return None
                
        except Exception as e:
            print(f"Error in collision-free inverse kinematics: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
        
    
    
    def force_robot_ready_state(self):
        """Force robot into ready state with comprehensive recovery
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                print("=== Force Robot Recovery ===")
                
                # First, try emergency stop and recovery
                print("Triggering emergency stop for clean recovery...")
                try:
                    self.arm.emergency_stop()
                    time.sleep(2.0)
                except:
                    pass
                
                # Reset everything step by step
                print("Step 1: Clearing all errors and warnings...")
                try:
                    self.arm.clean_error()
                    self.arm.clean_warn()
                    time.sleep(1.0)
                except:
                    pass
                
                print("Step 2: Disabling and re-enabling motion...")
                try:
                    self.arm.motion_enable(enable=False)
                    time.sleep(1.0)
                    self.arm.motion_enable(enable=True)
                    time.sleep(1.0)
                except:
                    pass
                
                print("Step 3: Setting mode to manual then position control...")
                try:
                    self.arm.set_mode(2)  # Manual mode first
                    time.sleep(1.0)
                    self.arm.set_mode(0)  # Then position control
                    time.sleep(1.0)
                except:
                    pass
                
                print("Step 4: Setting state through pause to ready...")
                try:
                    self.arm.set_state(4)  # Pause state
                    time.sleep(2.0)
                    self.arm.set_state(0)  # Ready state
                    time.sleep(2.0)
                except:
                    pass
                
                # Final verification
                final_state = self.arm.state
                final_mode = self.arm.mode
                has_error = self.arm.has_error
                has_warn = self.arm.has_warn
                
                print(f"Final robot status: Mode={final_mode}, State={final_state}")
                print(f"Error status: has_error={has_error}, has_warn={has_warn}")
                
                if final_state == 0 and final_mode == 0 and not has_error:
                    print("Robot successfully recovered to ready state")
                    return True
                else:
                    print("Robot recovery failed")
                    return False
                    
        except Exception as e:
            print(f"Error in force robot recovery: {str(e)}")
            return False
    
    def prepare_robot_for_execution(self):
        """Prepare robot for trajectory execution with robust error recovery
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                print("=== Preparing Robot for Execution ===")
                
                # Check current status
                print(f"Initial status: Mode={self.arm.mode}, State={self.arm.state}")
                print(f"Error status: has_error={self.arm.has_error}, has_warn={self.arm.has_warn}")
                
                # If robot is in error state (state 2) or has errors, do comprehensive recovery
                if self.arm.state == 2 or self.arm.has_error or self.arm.has_warn:
                    print("Robot needs recovery - attempting comprehensive reset...")
                    if not self.force_robot_ready_state():
                        print("Comprehensive recovery failed")
                        return False
                
                # Standard preparation sequence
                elif self.arm.mode != 0:
                    print("Robot needs standard preparation...")
                    
                    # Clear any lingering errors
                    if self.arm.has_error or self.arm.has_warn:
                        print("Clearing robot errors...")
                        if not self.clean_robot_error():
                            print("Failed to clear errors")
                            return False
                    
                    # Enable motion
                    print("Enabling motion...")
                    code = self.arm.motion_enable(enable=True)
                    if code != 0:
                        print(f"Failed to enable motion, error code: {code}")
                        return False
                    time.sleep(0.5)
                    
                    # Set to position control mode
                    print("Setting position control mode...")
                    code = self.arm.set_mode(0)
                    if code != 0:
                        print(f"Failed to set position control mode, error code: {code}")
                        return False
                    time.sleep(0.5)
                    
                    # Set to ready state
                    print("Setting ready state...")
                    code = self.arm.set_state(0)
                    if code != 0:
                        print(f"Failed to set ready state, error code: {code}")
                        return False
                    time.sleep(1.0)
                
                # Final verification
                return self.check_robot_ready_for_execution()
                
        except Exception as e:
            print(f"Error preparing robot for execution: {e}")
            return False
    
    def check_robot_ready_for_execution(self):
        """Check if robot is ready for trajectory execution
        
        Returns:
            bool: True if ready, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                # Get current status
                current_mode = self.arm.mode
                current_state = self.arm.state
                has_error = self.arm.has_error
                has_warn = self.arm.has_warn
                
                print(f"Robot status check: Mode={current_mode}, State={current_state}")
                print(f"Error status: has_error={has_error}, has_warn={has_warn}")
                
                # Check for errors
                if has_error:
                    print(f"Robot has error: {self.arm.error_code}")
                    return False
                    
                if has_warn:
                    print(f"Robot has warning: {self.arm.warn_code}")
                    # Warnings might be acceptable, continue
                    
                # Check mode and state
                if current_mode != 0:
                    print(f"Robot not in position control mode (mode: {current_mode})")
                    return False
                    
                # if current_state != 0:
                #     print(f"Robot not in ready state (state: {current_state})")
                #     return False
                    
                # Check if motors are enabled
                try:
                    motor_states = self.arm.motor_enable_states
                    if motor_states and not all(motor_states):
                        print("Not all motors are enabled")
                        return False
                except:
                    pass  # Skip motor check if not available
                    
                print("Robot is ready for execution")
                return True
                
        except Exception as e:
            print(f"Error checking robot readiness: {e}")
            return False
    
    
    def check_start_state_validity(self, joint_positions):
        """Check if the given joint state is valid (collision-free and within limits)"""
        if self.motion_gen is None:
            print("Motion generator not initialized")
            return False
            
        try:
            # Create a joint state from the given joint positions
            js = CuroboJointState.from_position(
                self.tensor_args.to_device([joint_positions]),
                joint_names=[
                    f"{self.config.prefix}joint1",
                    f"{self.config.prefix}joint2",
                    f"{self.config.prefix}joint3",
                    f"{self.config.prefix}joint4",
                    f"{self.config.prefix}joint5",
                    f"{self.config.prefix}joint6",
                ],
            )
            
            # Check if the state is valid using motion_gen's built-in method
            valid, status = self.motion_gen.check_start_state(js)

            if not valid:
                print(f"Start state is invalid: {status}")
            
            return valid
        except Exception as e:
            print(f"Error checking start state validity: {str(e)}")
            return False
    
    def compute_ik_goalset(self, positions, orientations=None, num_seeds=32):
        """Compute IK to find a solution that reaches one pose in a set of poses
        
        Args:
            positions: List of [x, y, z] target positions
            orientations: List of [w, x, y, z] orientations, or None to use top-down for all
            num_seeds: Number of IK seeds to try
            
        Returns:
            tuple: (joint_positions, reached_index) or (None, None) if failed
        """
        try:
            if self.ik_solver is None:
                print("IK solver not initialized")
                return None, None
                
            # Create a list of positions and orientations
            if not isinstance(positions, list):
                positions = [positions]
                
            if orientations is None:
                # Use top-down orientation for all positions
                default_orientation = self.create_top_down_orientation()
                orientations = [default_orientation] * len(positions)
            elif not isinstance(orientations, list):
                orientations = [orientations]
                
            if len(positions) != len(orientations):
                print(f"Error: Number of positions ({len(positions)}) does not match orientations ({len(orientations)})")
                return None, None
                
            # Convert to torch tensors
            position_tensors = self.tensor_args.to_device(positions)
            orientation_tensors = self.tensor_args.to_device(orientations)
            
            # Create cuRobo Pose
            goal_pose = CuroboPose(
                position=position_tensors,
                quaternion=orientation_tensors
            )
            
            # Sample configs for seeds
            q_sample = self.ik_solver.sample_configs(num_seeds)
            
            # Solve goalset IK
            result = self.ik_solver.solve_goalset(
                goal_pose, 
                seed_config=q_sample, 
                return_seeds=1
            )
            
            if result.success.item():
                joint_positions = result.solution[0].cpu().numpy()
                reached_index = result.goalset_index.item() if result.goalset_index is not None else 0
                print(f"IK succeeded for goal {reached_index} with position error: {result.position_error.item():.6f}m")
                return joint_positions, reached_index
            else:
                print("No valid IK solution found for any of the goal poses")
                return None, None
        except Exception as e:
            print(f"Error in goalset inverse kinematics: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, None
    
    def create_top_down_orientation(self):
        """Create a quaternion for top-down orientation"""
        # This quaternion represents Z-axis pointing downward in the base frame
        orientation = [0.0, 1.0, 0.0, 0.0]  # w, x, y, z (180 degrees around Y axis)
        
        print(
            f"Created top-down orientation: [w={orientation[0]}, x={orientation[1]}, "
            f"y={orientation[2]}, z={orientation[3]}]"
        )
        
        return orientation
    
    
    def _create_joint_state_with_batch(self, joint_positions):
        """Helper function to create a CuroboJointState with proper batch dimension
        
        Args:
            joint_positions: Joint positions as tensor, numpy array, or list
            
        Returns:
            CuroboJointState: Joint state with proper batch dimension
        """
        try:
            # Convert to tensor if needed
            if isinstance(joint_positions, (list, np.ndarray)):
                position_tensor = self.tensor_args.to_device([joint_positions])
            else:
                # Already a tensor
                position_tensor = joint_positions
                
            # Ensure proper batch dimension
            if position_tensor.dim() == 1:
                position_tensor = position_tensor.unsqueeze(0)  # Add batch dimension (1, dof)
            elif position_tensor.dim() > 2:
                position_tensor = position_tensor.reshape(1, -1)  # Reshape to (1, dof)
                
            # Ensure we have exactly the right number of joints
            if position_tensor.shape[-1] != self.config.dof:
                print(f"Warning: Expected {self.config.dof} joints, got {position_tensor.shape[-1]}")
                position_tensor = position_tensor[..., :self.config.dof]  # Truncate if needed
                
            # Create joint state
            joint_state = CuroboJointState.from_position(
                position_tensor,
                joint_names=[f"{self.config.prefix}joint{i}" for i in range(1, self.config.dof + 1)]
            )
            
            return joint_state
            
        except Exception as e:
            print(f"Error creating joint state with batch: {e}")
            raise

    
    def move_to_pose(
        self,
        target_position,
        target_orientation=None,
        force_top_down=False,
        unconstrained_orientation=False,
        planning_timeout=10.0,
        execute=False,
        speed_factor=1.0
    ):
        """Plan movement to a target pose with proper batch dimension handling"""
        
        # Handle orientation (same as before)
        if force_top_down:
            target_orientation = self.create_top_down_orientation()
            print("Using top-down orientation for target")
            unconstrained_orientation = False
        elif target_orientation is None:
            target_orientation = self.create_top_down_orientation()
            print("No orientation provided, using top-down orientation")
            unconstrained_orientation = False
        else:
            # Validate the provided orientation
            quat_norm = np.sqrt(sum(x**2 for x in target_orientation))
            if abs(quat_norm - 1.0) > 0.01 or quat_norm < 0.01:
                print("Invalid quaternion detected. Using top-down orientation instead.")
                target_orientation = self.create_top_down_orientation()
                unconstrained_orientation = False
            else:
                print("Using provided orientation")
                
        # Get current joint state from the robot if connected
        if self.arm is not None:
            robot_joints = self.get_robot_joint_state()
            if robot_joints is not None:
                self.set_current_joint_state(robot_joints)
                
        # Check if current joints are available
        with self.joint_state_lock:
            if self.current_joints is None:
                print("No current joint state available")
                return False, None, None
            current_joints = self.current_joints
            
            # Verify joint state size
            if len(current_joints) != self.config.dof:
                print(f"Joint state size mismatch: {len(current_joints)} != {self.config.dof}")
                return False, None, None
            
        try:
            # Convert to CuroboJointState using helper function
            start_state = self._create_joint_state_with_batch(current_joints)
            
            # Convert to torch tensors for cuRobo
            target_position_tensor = self.tensor_args.to_device([target_position])
            target_orientation_tensor = self.tensor_args.to_device([target_orientation])
            
            # Create cuRobo Goal Pose
            goal_pose = CuroboPose(
                position=target_position_tensor,
                quaternion=target_orientation_tensor
            )
            
            # Create planning configuration with optimized parameters
            plan_config = MotionGenPlanConfig(
                max_attempts=5,
                timeout=planning_timeout * 0.9,
                enable_opt=True,
                enable_graph=True,
                enable_graph_attempt=2,
                enable_finetune_trajopt=True,
                parallel_finetune=True,
                time_dilation_factor=0.99
            )
            
            # Plan the motion
            print("Planning motion to target pose...")
            result = self.motion_gen.plan_single(
                start_state=start_state,
                goal_pose=goal_pose,
                plan_config=plan_config
            )
            
            if result.success.item():
                print(f"Motion planning successful in {result.solve_time} seconds, "
                    f"motion time: {result.motion_time} seconds")
                
                # Get the interpolated trajectory
                trajectory = result.get_interpolated_plan()
                
                # Execute on the robot if requested
                if execute and self.arm is not None:
                    print("Executing planned trajectory...")
                    execution_success = self.execute_trajectory(trajectory, result.interpolation_dt, speed_factor)
                    
                    if execution_success:
                        print("Trajectory execution completed successfully")
                    else:
                        print("Trajectory execution failed")
                
                return True, trajectory, result.interpolation_dt
            else:
                print(f"Motion planning failed: {result.status}, "
                    f"after {result.solve_time} seconds")
                return False, None, None
        except Exception as e:
            print(f"Error in motion planning: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, None, None
        
        
    def move_to_pose_with_preparation(
        self,
        target_position,
        target_orientation=None,
        force_top_down=False,
        unconstrained_orientation=False,
        planning_timeout=10.0,
        execute=False,
        speed_factor=1.0
    ):
        """Plan movement to a target pose with improved robot preparation"""
        
        # Plan the motion (same as before)
        success, trajectory, dt = self.move_to_pose(
            target_position, target_orientation, force_top_down, 
            unconstrained_orientation, planning_timeout, execute=False, speed_factor=speed_factor
        )
        
        if not success:
            return False, None, None
            
        # Execute with proper preparation if requested
        if execute and self.arm is not None:
            print("Preparing robot for execution...")
            if not self.prepare_robot_for_execution():
                print("Failed to prepare robot for execution")
                return success, trajectory, dt  # Return planning success but execution failure
                
            print("Executing planned trajectory...")
            execution_success = self.execute_trajectory(trajectory, dt, speed_factor)
            
            if execution_success:
                print("Trajectory execution completed successfully")
            else:
                print("Trajectory execution failed")
                
        return success, trajectory, dt
        
    def plan_cartesian_path(
        self,
        start_position,
        start_orientation,
        target_position,
        target_orientation=None,
        planning_timeout=10.0,
        time_scaling=0.99,
        execute=False,
        speed_factor=1.0
    ):
        """Plan a Cartesian (straight-line) path motion
        
        Args:
            start_position: [x, y, z] starting position
            start_orientation: [w, x, y, z] quaternion starting orientation
            target_position: [x, y, z] target position
            target_orientation: [w, x, y, z] quaternion target orientation, or None to use start_orientation
            planning_timeout: Timeout for planning in seconds
            time_scaling: Time scaling factor (slower for values > 1.0)
            execute: Whether to execute the planned trajectory on the physical robot
            speed_factor: Speed factor for execution (>1 is faster)
            
        Returns:
            tuple: (success, trajectory, dt)
        """
        print(f"Planning Cartesian path from {start_position} to {target_position}")
        
        # Use current orientation for target if not provided
        if target_orientation is None:
            target_orientation = start_orientation
            print("No target orientation provided, using start orientation")
            
        # Get current joint state from the robot if connected
        if self.arm is not None:
            robot_joints = self.get_robot_joint_state()
            if robot_joints is not None:
                self.set_current_joint_state(robot_joints)
            
        # Check if current joints are available
        with self.joint_state_lock:
            if self.current_joints is None:
                print("No current joint state available")
                return False, None, None
            current_joints = self.current_joints
            
            # Verify joint state size
            if len(current_joints) != self.config.dof:
                print(f"Joint state size mismatch: {len(current_joints)} != {self.config.dof}")
                return False, None, None
            
        try:
            # Convert to CuroboJointState
            start_state = CuroboJointState.from_position(
                self.tensor_args.to_device([current_joints[:self.config.dof]]),
                joint_names=[
                    f"{self.config.prefix}joint{i}" for i in range(1, self.config.dof + 1)
                ],
            )
            
            # Convert to torch tensors for cuRobo
            target_position_tensor = self.tensor_args.to_device([target_position])
            target_orientation_tensor = self.tensor_args.to_device([target_orientation])
            
            # Create cuRobo Goal Pose
            goal_pose = CuroboPose(
                position=target_position_tensor,
                quaternion=target_orientation_tensor
            )
            
            # Method 1: Try Cartesian planning with optimized trajectory parameters
            print("Attempting Cartesian planning with trajectory optimization...")
            cartesian_plan_config = MotionGenPlanConfig(
                max_attempts=3,
                timeout=planning_timeout * 0.4,  # Use 40% of time for first attempt
                enable_opt=True,
                enable_graph=False,  # Disable graph for pure trajectory optimization
                enable_finetune_trajopt=True,
                parallel_finetune=True,
                # Trajectory optimization settings for straighter paths
                # trajopt_tsteps=40,  # More timesteps for smoother paths
                # optimize_dt=True,   # Allow dt optimization
                time_dilation_factor=time_scaling,
                # Don't use problematic pose cost metric
                pose_cost_metric=None
            )
            
            # Plan the motion
            result = self.motion_gen.plan_single(
                start_state=start_state,
                goal_pose=goal_pose,
                plan_config=cartesian_plan_config
            )
            
            if result.success.item():
                print(f"Cartesian planning successful in {result.solve_time} seconds, "
                    f"motion time: {result.motion_time} seconds")
                
                # Get the interpolated trajectory
                trajectory = result.get_interpolated_plan()
                
                # Verify trajectory is reasonably straight
                if self._verify_cartesian_trajectory(trajectory, start_position, target_position):
                    print("Trajectory verified as sufficiently Cartesian")
                    
                    # Execute on the robot if requested
                    if execute and self.arm is not None:
                        print("Executing planned Cartesian trajectory on the robot...")
                        execution_success = self.execute_trajectory(
                            trajectory=trajectory,
                            dt=result.interpolation_dt,
                            speed_factor=speed_factor
                        )
                        
                        if execution_success:
                            print("Cartesian trajectory execution completed successfully")
                        else:
                            print("Cartesian trajectory execution failed")
                    
                    return True, trajectory, result.interpolation_dt
                else:
                    print("Trajectory not sufficiently Cartesian, trying alternative approach...")
            
            # Method 2: Try with interpolated waypoints
            print("Attempting Cartesian planning with interpolated waypoints...")
            waypoint_success, trajectory, dt = self._plan_with_cartesian_waypoints(
                start_state, start_position, target_position, target_orientation, 
                planning_timeout * 0.4, time_scaling, execute, speed_factor
            )
            
            if waypoint_success:
                return True, trajectory, dt
            
            # Method 3: Fall back to standard planning
            print("Cartesian methods failed, trying standard planning as fallback...")
            standard_plan_config = MotionGenPlanConfig(
                max_attempts=3,
                timeout=planning_timeout * 0.2,  # Use remaining time
                enable_opt=True,
                enable_graph=True,  # Enable graph for standard planning
                enable_finetune_trajopt=True,
                time_dilation_factor=time_scaling
            )
            
            # Plan with standard (non-Cartesian) planning
            result = self.motion_gen.plan_single(
                start_state=start_state,
                goal_pose=goal_pose,
                plan_config=standard_plan_config
            )
            
            if result.success.item():
                print(f"Standard planning successful in {result.solve_time} seconds")
                
                # Get the interpolated trajectory
                trajectory = result.get_interpolated_plan()
                
                # Execute on the robot if requested
                if execute and self.arm is not None:
                    print("Executing planned trajectory on the robot...")
                    execution_success = self.execute_trajectory(
                        trajectory=trajectory,
                        dt=result.interpolation_dt,
                        speed_factor=speed_factor
                    )
                    
                    if execution_success:
                        print("Trajectory execution completed successfully")
                    else:
                        print("Trajectory execution failed")
                
                return True, trajectory, result.interpolation_dt
            else:
                print(f"All planning methods failed: {result.status}")
                return False, None, None
                
        except Exception as e:
            print(f"Error in Cartesian planning: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, None, None

    def _verify_cartesian_trajectory(self, trajectory, start_pos, target_pos):
        """Verify that a trajectory is reasonably straight (Cartesian)"""
        try:
            if trajectory is None or len(trajectory) == 0:
                return False
                
            print(f"Verifying trajectory with {len(trajectory)} points")
            
            # Get end effector positions from the trajectory
            ee_positions = []
            
            # Sample points from trajectory to avoid processing too many points
            sample_indices = np.linspace(0, len(trajectory) - 1, min(10, len(trajectory)), dtype=int)
            
            for idx in sample_indices:
                try:
                    joint_state = trajectory[idx]
                    
                    # Ensure we have the right shape for forward kinematics
                    if hasattr(joint_state, 'position'):
                        joint_positions = joint_state.position
                        if joint_positions.dim() == 1:
                            joint_positions = joint_positions.unsqueeze(0)
                    else:
                        # Handle case where trajectory might be raw tensor
                        joint_positions = joint_state
                        if joint_positions.dim() == 1:
                            joint_positions = joint_positions.unsqueeze(0)
                    
                    # Get forward kinematics for this joint configuration
                    fk_result = self.ik_solver.fk(joint_positions)
                    ee_pos = fk_result.ee_position[0].cpu().numpy()
                    ee_positions.append(ee_pos)
                    
                except Exception as e:
                    print(f"Error processing trajectory point {idx}: {e}")
                    continue
            
            if len(ee_positions) < 2:
                print("Could not process enough trajectory points for verification")
                return True  # Accept if we can't verify
                
            ee_positions = np.array(ee_positions)
            
            # Calculate the straight-line distance
            straight_line_distance = np.linalg.norm(np.array(target_pos) - np.array(start_pos))
            
            # Calculate the actual path length
            path_distances = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)
            actual_path_length = np.sum(path_distances)
            
            # Calculate deviation ratio (should be close to 1.0 for straight paths)
            if straight_line_distance > 0.001:  # Avoid division by zero
                deviation_ratio = actual_path_length / straight_line_distance
                print(f"Path deviation ratio: {deviation_ratio} (closer to 1.0 is better)")
                
                # Accept if path is within 30% of straight line (more lenient)
                return deviation_ratio < 1.3
            else:
                # Very short movement, accept it
                return True
                
        except Exception as e:
            print(f"Error verifying Cartesian trajectory: {e}")
            return True  # Accept if we can't verify

    def _plan_with_cartesian_waypoints(self, start_state, start_pos, target_pos, target_orientation, 
                                timeout, time_scaling, execute, speed_factor):
        """Plan Cartesian path by interpolating waypoints"""
        try:
            print("Planning with interpolated Cartesian waypoints...")
            
            # Generate waypoints along the straight line
            num_waypoints = 3  # Reduce number of waypoints to avoid complexity
            waypoints = []
            
            for i in range(num_waypoints + 1):
                t = i / num_waypoints
                waypoint_pos = (1 - t) * np.array(start_pos) + t * np.array(target_pos)
                waypoints.append(waypoint_pos.tolist())
            
            print(f"Generated {len(waypoints)} waypoints")
            
            # Plan through each waypoint
            current_state = start_state
            full_trajectory = []
            total_dt = 0
            
            for i, waypoint in enumerate(waypoints[1:], 1):  # Skip first waypoint (start position)
                print(f"Planning to waypoint {i}/{len(waypoints)-1}: {waypoint}")
                
                # Convert waypoint to goal pose
                waypoint_tensor = self.tensor_args.to_device([waypoint])
                orientation_tensor = self.tensor_args.to_device([target_orientation])
                
                goal_pose = CuroboPose(
                    position=waypoint_tensor,
                    quaternion=orientation_tensor
                )
                
                # Plan to this waypoint
                waypoint_config = MotionGenPlanConfig(
                    max_attempts=2,
                    timeout=timeout / (len(waypoints) - 1),  # Distribute time across waypoints
                    enable_opt=True,
                    enable_graph=False,
                    enable_finetune_trajopt=True,
                    time_dilation_factor=time_scaling
                )
                
                result = self.motion_gen.plan_single(
                    start_state=current_state,
                    goal_pose=goal_pose,
                    plan_config=waypoint_config
                )
                
                if not result.success.item():
                    print(f"Failed to plan to waypoint {i}")
                    return False, None, None
                
                # Get trajectory segment and convert to proper format
                segment = result.get_interpolated_plan()
                print(f"Waypoint {i} segment has {len(segment)} points")
                
                # Convert CuRobo trajectory to list of individual joint states
                segment_list = []
                try:
                    # Extract each point individually from the trajectory
                    for j in range(len(segment)):
                        try:
                            # Use CuRobo's indexing to get individual states
                            point = segment[j]
                            segment_list.append(point)
                        except (IndexError, ValueError) as e:
                            print(f"Error accessing segment point {j}: {e}")
                            # Try alternative extraction method
                            if hasattr(segment, 'state_seq'):
                                point = segment.state_seq[j] if j < len(segment.state_seq) else None
                                if point is not None:
                                    segment_list.append(point)
                            break
                            
                except Exception as e:
                    print(f"Error converting segment to list: {e}")
                    # Fallback: try to extract the underlying state sequence
                    try:
                        if hasattr(segment, 'state_seq'):
                            # Extract joint states from the state sequence
                            for j in range(min(len(segment), segment.state_seq.shape[0])):
                                joint_pos = segment.state_seq.position[j:j+1]  # Keep batch dimension
                                joint_state = CuroboJointState.from_position(
                                    joint_pos,
                                    joint_names=[f"{self.config.prefix}joint{k}" for k in range(1, self.config.dof + 1)]
                                )
                                segment_list.append(joint_state)
                        else:
                            print("Cannot extract trajectory points, skipping this segment")
                            return False, None, None
                    except Exception as inner_e:
                        print(f"Fallback extraction failed: {inner_e}")
                        return False, None, None
                
                if len(segment_list) == 0:
                    print(f"No valid points extracted from segment {i}")
                    return False, None, None
                    
                print(f"Successfully extracted {len(segment_list)} points from segment {i}")
                
                # Append to full trajectory (skip first point to avoid duplication)
                if i == 1:
                    full_trajectory.extend(segment_list)
                else:
                    full_trajectory.extend(segment_list[1:])
                
                total_dt = result.interpolation_dt
                
                # FIX: Update current state for next segment with proper batch dimensions
                if len(segment_list) > 0:
                    last_point = segment_list[-1]  # Get last point safely
                    
                    # Extract joint positions and recreate with proper batch dimension
                    try:
                        if hasattr(last_point, 'position'):
                            # Extract joint positions from the joint state
                            joint_positions = last_point.position
                            if joint_positions.dim() > 1:
                                joint_positions = joint_positions[0]  # Remove batch dimension to get raw positions
                            
                            # Recreate joint state with proper batch dimension using helper function
                            current_state = self._create_joint_state_with_batch(joint_positions.cpu().numpy())
                            
                        else:
                            # Handle case where last_point might be raw tensor
                            if last_point.dim() > 1:
                                joint_positions = last_point[0]
                            else:
                                joint_positions = last_point
                            
                            # Recreate joint state with proper batch dimension
                            current_state = self._create_joint_state_with_batch(joint_positions.cpu().numpy())
                            
                    except Exception as state_error:
                        print(f"Error updating current state: {state_error}")
                        # Fallback: use the segment result's final state
                        if hasattr(result, 'get_final_state'):
                            current_state = result.get_final_state()
                        else:
                            print("Cannot update current state, trajectory may be incomplete")
                            return False, None, None
                else:
                    print(f"Warning: Empty segment for waypoint {i}")
                    return False, None, None
            
            print(f"Successfully planned waypoint-based Cartesian path with {len(full_trajectory)} points")
            
            # Execute if requested
            if execute and self.arm is not None:
                print("Executing waypoint-based Cartesian trajectory...")
                execution_success = self.execute_trajectory(
                    trajectory=full_trajectory,
                    dt=total_dt,
                    speed_factor=speed_factor
                )
                
                if execution_success:
                    print("Waypoint-based trajectory execution completed successfully")
                else:
                    print("Waypoint-based trajectory execution failed")
            
            return True, full_trajectory, total_dt
            
        except Exception as e:
            print(f"Error in waypoint-based Cartesian planning: {e}")
            import traceback
            print(traceback.format_exc())
            return False, None, None
    
    def plan_joint_motion(
        self,
        target_joints,
        planning_timeout=10.0,
        speed_scaling=0.99,
        execute=False,
        speed_factor=1.0
    ):
        """Plan motion in joint space and optionally execute on the robot
        
        Args:
            target_joints: Target joint positions
            planning_timeout: Timeout for planning in seconds
            speed_scaling: Speed scaling factor (faster for values > 1.0)
            execute: Whether to execute the planned trajectory on the physical robot
            speed_factor: Speed factor for execution (>1 is faster)
            
        Returns:
            tuple: (success, trajectory, dt)
        """
        print(f"Planning joint motion to {target_joints}")
        
        # Get current joint state from the robot if connected
        if self.arm is not None:
            robot_joints = self.get_robot_joint_state()
            if robot_joints is not None:
                self.set_current_joint_state(robot_joints)
        
        # Check if current joints are available
        with self.joint_state_lock:
            if self.current_joints is None:
                print("No current joint state available")
                return False, None, None
            current_joints = self.current_joints
        
        # Check if the start state is valid
        if not self.check_start_state_validity(current_joints):
            print("Start state is invalid (in collision or outside joint limits)")
            return False, None, None
            
        try:
            # Create start and goal states for planning
            start_state = CuroboJointState.from_position(
                self.tensor_args.to_device([current_joints]),
                joint_names=[
                    f"{self.config.prefix}joint{i}" for i in range(1, self.config.dof + 1)
                ],
            )
            
            goal_state = CuroboJointState.from_position(
                self.tensor_args.to_device([target_joints]),
                joint_names=[
                    f"{self.config.prefix}joint{i}" for i in range(1, self.config.dof + 1)
                ],
            )
            
            # Create planning configuration
            plan_config = MotionGenPlanConfig(
                max_attempts=3,
                timeout=planning_timeout * 0.9,
                enable_opt=True,
                enable_graph=True,
                enable_finetune_trajopt=True,
                time_dilation_factor= speed_scaling  # Scale by inverse of speed
            )
            
            # Plan the motion in joint space
            print("Planning joint space motion...")
            result = self.motion_gen.plan_single_js(
                start_state=start_state,
                goal_state=goal_state,
                plan_config=plan_config
            )
            
            if result.success.item():
                print(
                    f"Joint planning successful in {result.solve_time} seconds, "
                    f"motion time: {result.motion_time} seconds"
                )
                
                # Get the interpolated trajectory
                trajectory = result.get_interpolated_plan()
                
                # Execute on the robot if requested
                if execute and self.arm is not None:
                    print("Executing planned joint trajectory on the robot...")
                    execution_success = self.execute_trajectory(
                        trajectory=trajectory,
                        dt=result.interpolation_dt,
                        speed_factor=speed_factor
                    )
                    
                    if execution_success:
                        print("Joint trajectory execution completed successfully")
                    else:
                        print("Joint trajectory execution failed")
                
                # Return the trajectory and timestep for execution
                return True, trajectory, result.interpolation_dt
            else:
                print(f"Joint planning failed: {result.status}")
                return False, None, None
        except Exception as e:
            print(f"Error in joint planning: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, None, None
    
    def move_to_home(self, speed=0.3, execute=False, speed_factor=1.0):
        """Plan movement to the home position and optionally execute on the robot
        
        Args:
            speed: Speed scaling factor (faster for values > 1.0)
            execute: Whether to execute the planned trajectory on the physical robot
            speed_factor: Speed factor for execution (>1 is faster)
            
        Returns:
            tuple: (success, trajectory, dt)
        """
        print(f"Planning motion to home position with speed: {speed}")
        
        try:
            # Get the retract config from motion generator
            retract_cfg = self.motion_gen.get_retract_config()
            home_joints = retract_cfg.cpu().numpy()
            
            # Plan joint motion to home position
            return self.plan_joint_motion(
                target_joints=home_joints,
                planning_timeout=10.0,
                speed_scaling=speed,
                execute=execute,
                speed_factor=speed_factor
            )
        except Exception as e:
            print(f"Error in move_to_home: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, None, None
    
    def get_forward_kinematics(self, joint_positions=None):
        """Compute forward kinematics for given joint positions
        
        Args:
            joint_positions: Optional joint positions, or None to use current
            
        Returns:
            tuple: (position, orientation) or None if error
        """
        try:
            # Get current joint state from the robot if connected and no positions provided
            if joint_positions is None and self.arm is not None:
                robot_joints = self.get_robot_joint_state()
                if robot_joints is not None:
                    joint_positions = robot_joints
            
            # Use current joints if still not provided
            if joint_positions is None:
                with self.joint_state_lock:
                    if self.current_joints is None:
                        print("No current joint state available")
                        return None
                    joint_positions = self.current_joints
            
            # Check if we can use the robot's FK for better accuracy
            if self.arm is not None:
                try:
                    with self.arm_lock:
                        code, pose = self.arm.get_position(is_radian=True)
                        if code == 0 and pose is not None:
                            # XArm SDK returns TCP pose as [x, y, z, roll, pitch, yaw]
                            position = pose[:3]
                            # Convert Euler angles to quaternion
                            roll, pitch, yaw = pose[3:]
                            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                            orientation = [qw, qx, qy, qz]
                            return position, orientation
                except Exception as e:
                    print(f"Could not get FK from robot: {str(e)}, using CuRobo FK")
            
            # Use CuRobo's FK if we can't use the robot's
            # Create CuroboJointState from joint values
            joint_state = CuroboJointState.from_position(
                self.tensor_args.to_device([joint_positions]),
                joint_names=[
                    f"{self.config.prefix}joint1",
                    f"{self.config.prefix}joint2",
                    f"{self.config.prefix}joint3",
                    f"{self.config.prefix}joint4",
                    f"{self.config.prefix}joint5",
                    f"{self.config.prefix}joint6",
                ],
            )
            
            # Compute forward kinematics
            fk_result = self.ik_solver.fk(self.tensor_args.to_device([joint_positions]))
            position = fk_result.ee_position[0].cpu().numpy()
            orientation = fk_result.ee_quaternion[0].cpu().numpy()
            
            return position, orientation
            
        except Exception as e:
            print(f"Error in forward kinematics: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def set_gripper(self, position, wait=True, timeout=5.0):
        """Control the gripper position
        
        Args:
            position: Gripper position (0 for open, 1 for closed, or value in between)
            wait: Whether to wait for the gripper operation to complete
            timeout: Timeout in seconds for the gripper operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
        code = 0
        try:
            with self.arm_lock:
                # Enable the gripper
                if not self.config.is_lite6:
                    code = self.arm.set_gripper_enable(True)
                if code != 0:
                    print(f"Failed to enable gripper, error code: {code}")
                    return False
                
                # Set gripper position
                # Note: xArm SDK takes position as 0 for open and 100 for closed
                scaled_position = position * 100
                print(f"Setting gripper position to {scaled_position}")
                code = 0
                if self.config.is_lite6:
                    if scaled_position < 50:
                        code = self.arm.open_lite6_gripper()
                    else:
                        code = self.arm.close_lite6_gripper()
                else:
                    code = self.arm.set_gripper_position(scaled_position, wait=wait, timeout=timeout)
                
                if code == 0:
                    print("Gripper operation completed successfully")
                    return True
                else:
                    print(f"Failed to set gripper position, error code: {code}")
                    return False
        except Exception as e:
            print(f"Error controlling gripper: {str(e)}")
            return False
    
    def close_gripper(self, wait=True, timeout=5.0):
        """Close the gripper
        
        Args:
            wait: Whether to wait for the gripper operation to complete
            timeout: Timeout in seconds for the gripper operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.set_gripper(1.0, wait, timeout)
    
    def open_gripper(self, wait=True, timeout=5.0):
        """Open the gripper
        
        Args:
            wait: Whether to wait for the gripper operation to complete
            timeout: Timeout in seconds for the gripper operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.set_gripper(0.0, wait, timeout)
    
    def execute_pick_and_place(
        self, 
        pick_position, 
        pick_orientation=None,
        place_position=None,
        place_orientation=None,
        approach_distance=0.1,
        approach_direction=[0, 0, 1],  # Default approaching from above
        planning_timeout=10.0,
        speed_factor=1.0
    ):
        """Execute a pick and place operation
        
        Args:
            pick_position: [x, y, z] picking position
            pick_orientation: [w, x, y, z] quaternion orientation for picking
            place_position: [x, y, z] placing position (or None to just pick)
            place_orientation: [w, x, y, z] quaternion orientation for placing
            approach_distance: Distance to approach from before picking/placing
            approach_direction: Direction vector for approach
            planning_timeout: Timeout for each planning phase
            speed_factor: Speed factor for execution
            
        Returns:
            bool: True if all operations completed successfully
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            # Normalize approach direction
            approach_dir = np.array(approach_direction)
            approach_dir = approach_dir / np.linalg.norm(approach_dir)
            
            # Calculate approach positions
            pick_approach_pos = np.array(pick_position) - approach_dir * approach_distance
            
            # Use top-down orientation if not provided
            if pick_orientation is None:
                pick_orientation = self.create_top_down_orientation()
                
            # Use same orientation for place if not provided
            if place_position is not None and place_orientation is None:
                place_orientation = pick_orientation
                
            # Execute picking sequence
            print(f"Executing pick operation at {pick_position}")
            
            # 1. Open the gripper
            if not self.open_gripper():
                print("Failed to open gripper")
                return False
                
            # 2. Move to approach position
            success, _, _ = self.move_to_pose(
                pick_approach_pos,
                pick_orientation,
                planning_timeout=planning_timeout,
                execute=True,
                speed_factor=speed_factor
            )
            
            if not success:
                print("Failed to move to pick approach position")
                return False
                
            # 3. Move to pick position
            success, _, _ = self.plan_cartesian_path(
                start_position=pick_approach_pos,
                start_orientation=pick_orientation,
                target_position=pick_position,
                target_orientation=pick_orientation,
                planning_timeout=planning_timeout,
                execute=True,
                speed_factor=speed_factor * 0.5  # Slower for precise positioning
            )
            
            if not success:
                print("Failed to move to pick position")
                return False
                
            # 4. Close the gripper
            if not self.close_gripper():
                print("Failed to close gripper")
                return False
                
            # 5. Move back to approach position
            success, _, _ = self.plan_cartesian_path(
                start_position=pick_position,
                start_orientation=pick_orientation,
                target_position=pick_approach_pos,
                target_orientation=pick_orientation,
                planning_timeout=planning_timeout,
                execute=True,
                speed_factor=speed_factor
            )
            
            if not success:
                print("Failed to retreat from pick position")
                return False
                
            # If no place position specified, we're done
            if place_position is None:
                print("Pick operation completed successfully")
                return True
                
            # Calculate place approach position
            place_approach_pos = np.array(place_position) - approach_dir * approach_distance
            
            # Execute placing sequence
            print(f"Executing place operation at {place_position}")
            
            # 6. Move to place approach position
            success, _, _ = self.move_to_pose(
                place_approach_pos,
                place_orientation,
                planning_timeout=planning_timeout,
                execute=True,
                speed_factor=speed_factor
            )
            
            if not success:
                print("Failed to move to place approach position")
                return False
                
            # 7. Move to place position
            success, _, _ = self.plan_cartesian_path(
                start_position=place_approach_pos,
                start_orientation=place_orientation,
                target_position=place_position,
                target_orientation=place_orientation,
                planning_timeout=planning_timeout,
                execute=True,
                speed_factor=speed_factor * 0.5  # Slower for precise positioning
            )
            
            if not success:
                print("Failed to move to place position")
                return False
                
            # 8. Open the gripper
            if not self.open_gripper():
                print("Failed to open gripper")
                return False
                
            # 9. Move back to approach position
            success, _, _ = self.plan_cartesian_path(
                start_position=place_position,
                start_orientation=place_orientation,
                target_position=place_approach_pos,
                target_orientation=place_orientation,
                planning_timeout=planning_timeout,
                execute=True,
                speed_factor=speed_factor
            )
            
            if not success:
                print("Failed to retreat from place position")
                return False
                
            print("Pick and place operation completed successfully")
            return True
            
        except Exception as e:
            print(f"Error in pick and place operation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
            
    def plan_push_pull(
        self, 
        distance, 
        is_push=True, 
        custom_normal=None, 
        move_parallel=False, 
        planning_timeout=10.0,
        current_position=None,
        current_orientation=None,
        execute=False,
        speed_factor=1.0
    ):
        """Plan a push or pull movement along a direction vector
        
        Args:
            distance: Distance to push/pull in meters
            is_push: True for push, False for pull
            custom_normal: Optional custom normal vector [x, y, z] for movement direction
            move_parallel: Whether to move parallel to the surface (perpendicular to normal)
            planning_timeout: Timeout for planning in seconds
            current_position: Optional current position [x, y, z], if None use forward kinematics 
            current_orientation: Optional current orientation [w, x, y, z], if None use forward kinematics
            execute: Whether to execute the planned trajectory on the physical robot
            speed_factor: Speed factor for execution (>1 is faster)
            
        Returns:
            tuple: (success, trajectory, dt)
        """
        direction_type = "push" if is_push else "pull"
        movement_type = "parallel" if move_parallel else "perpendicular"
        print(f"Planning {direction_type} movement with distance: {distance} {movement_type}")
        
        try:
            # Get current joint state from the robot if connected
            if self.arm is not None:
                robot_joints = self.get_robot_joint_state()
                if robot_joints is not None:
                    self.set_current_joint_state(robot_joints)
                    
            # Check if current joints are available
            with self.joint_state_lock:
                if self.current_joints is None:
                    print("No current joint state available")
                    return False, None, None
                current_joints = self.current_joints
            
            # Create CuroboJointState from current joint values
            start_state = CuroboJointState.from_position(
                self.tensor_args.to_device([current_joints]),
                joint_names=[
                    f"{self.config.prefix}joint1",
                    f"{self.config.prefix}joint2",
                    f"{self.config.prefix}joint3",
                    f"{self.config.prefix}joint4",
                    f"{self.config.prefix}joint5",
                    f"{self.config.prefix}joint6",
                ],
            )
            
            # Get current TCP pose through forward kinematics if not provided
            if current_position is None or current_orientation is None:
                # Get the pose from the robot if connected
                if self.arm is not None:
                    pose = self.get_robot_tcp_pose()
                    if pose:
                        current_position, current_orientation = pose
                        print(f"Using robot pose: position={current_position}, orientation={current_orientation}")
                    else:
                        # Fallback to FK if robot pose not available
                        pose = self.get_forward_kinematics()
                        if pose:
                            current_position, current_orientation = pose
                            print(f"Using FK for current pose: position={current_position}, orientation={current_orientation}")
                        else:
                            print("Failed to get current pose")
                            return False, None, None
                else:
                    # Compute forward kinematics to get current pose
                    pose = self.get_forward_kinematics()
                    if pose:
                        current_position, current_orientation = pose
                        print(f"Using FK for current pose: position={current_position}, orientation={current_orientation}")
                    else:
                        print("Failed to get current pose")
                        return False, None, None
            
            # Calculate the movement direction based on custom normal or TCP orientation
            if custom_normal is not None:
                # Use the provided custom normal
                print(f"Using custom surface normal: {custom_normal}")
                surface_normal = np.array(custom_normal)
                surface_normal = surface_normal / np.linalg.norm(surface_normal)
            else:
                # Fallback to using the TCP z-axis as the normal
                print("No custom normal provided, using TCP z-axis as normal")
                
                # Convert quaternion to rotation matrix to extract z-axis
                w, x, y, z = current_orientation
                rotation_matrix = np.array([
                    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
                ])
                
                # Extract z-axis (3rd column) from rotation matrix
                surface_normal = rotation_matrix[:, 2]
                surface_normal = surface_normal / np.linalg.norm(surface_normal)
            
            # Determine the movement direction based on parallel flag
            movement_dir = np.zeros(3)
            
            if move_parallel:
                # Movement is perpendicular to the surface normal (parallel to surface)
                # We need a perpendicular vector - we'll use the cross product with the global Z-axis
                global_z = np.array([0, 0, 1])
                perpendicular_dir = np.cross(surface_normal, global_z)
                
                # If the perpendicular direction is too close to zero (normal is parallel to Z),
                # use the X-axis instead
                if np.linalg.norm(perpendicular_dir) < 0.01:
                    global_x = np.array([1, 0, 0])
                    perpendicular_dir = np.cross(surface_normal, global_x)
                
                # Normalize the perpendicular direction
                if np.linalg.norm(perpendicular_dir) > 0.01:
                    perpendicular_dir = perpendicular_dir / np.linalg.norm(perpendicular_dir)
                    movement_dir = perpendicular_dir
                    print(f"Using movement parallel to surface: {movement_dir}")
                else:
                    # Fallback if we couldn't generate a valid perpendicular vector
                    print("Could not determine parallel direction, using surface normal")
                    movement_dir = surface_normal
            else:
                # Movement is along the surface normal (perpendicular to surface)
                movement_dir = surface_normal
                print(f"Using movement perpendicular to surface: {movement_dir}")
            
            # Calculate the movement vector, adjusting direction based on push/pull
            direction_factor = 1.0 if is_push else -1.0
            movement = distance * direction_factor * movement_dir
            
            # Calculate target position
            target_position = current_position + movement
            
            # Keep the current orientation for the target
            target_orientation = current_orientation
            
            # Convert to torch tensors for cuRobo
            target_position_tensor = self.tensor_args.to_device([target_position])
            target_orientation_tensor = self.tensor_args.to_device([target_orientation])
            
            # Create a pose cost metric that enforces straight-line motion
            pose_cost_metric = PoseCostMetric(
                # Keep all position components constrained for linear motion
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device([0.05, 0.05, 0.05, 1.0, 1.0, 1.0]),
                project_to_goal_frame=False,
            )
            
            # Create goal pose for planning
            goal_pose = CuroboPose(
                position=target_position_tensor,
                quaternion=target_orientation_tensor
            )
            
            # Create planning config with Cartesian constraints
            plan_config = MotionGenPlanConfig(
                max_attempts=3,
                timeout=planning_timeout * 0.9,
                enable_opt=True,
                enable_graph=False,  # Don't use graph for Cartesian paths
                enable_finetune_trajopt=True,
                pose_cost_metric=pose_cost_metric,
            )
            
            # Plan the motion
            print("Planning push/pull Cartesian path...")
            result = self.motion_gen.plan_single(
                start_state=start_state,
                goal_pose=goal_pose,
                plan_config=plan_config
            )
            
            if result.success.item():
                print(
                    f"Push/pull planning successful in {result.solve_time} seconds, "
                    f"motion time: {result.motion_time} seconds"
                )
                
                # Get the interpolated trajectory
                trajectory = result.get_interpolated_plan()
                
                # Execute on the robot if requested
                if execute and self.arm is not None:
                    print("Executing planned push/pull trajectory on the robot...")
                    execution_success = self.execute_trajectory(
                        trajectory=trajectory,
                        dt=result.interpolation_dt,
                        speed_factor=speed_factor
                    )
                    
                    if execution_success:
                        print("Push/pull trajectory execution completed successfully")
                    else:
                        print("Push/pull trajectory execution failed")
                
                # Return the trajectory and timestep for execution
                return True, trajectory, result.interpolation_dt
            else:
                print(f"Push/pull planning failed: {result.status}")
                
                # Try standard planning as a fallback
                print("Trying standard planning as fallback...")
                standard_plan_config = MotionGenPlanConfig(
                    max_attempts=3,
                    timeout=planning_timeout * 0.5,  # Use remaining time
                    enable_opt=True,
                    enable_graph=True,  # Enable graph for standard planning
                    enable_finetune_trajopt=True,
                )
                
                # Plan with standard (non-Cartesian) planning
                result = self.motion_gen.plan_single(
                    start_state=start_state,
                    goal_pose=goal_pose,
                    plan_config=standard_plan_config
                )
                
                if result.success.item():
                    print(f"Standard planning successful in {result.solve_time} seconds")
                    
                    # Get the interpolated trajectory
                    trajectory = result.get_interpolated_plan()
                    
                    # Execute on the robot if requested
                    if execute and self.arm is not None:
                        print("Executing planned trajectory on the robot...")
                        execution_success = self.execute_trajectory(
                            trajectory=trajectory,
                            dt=result.interpolation_dt,
                            speed_factor=speed_factor
                        )
                        
                        if execution_success:
                            print("Trajectory execution completed successfully")
                        else:
                            print("Trajectory execution failed")
                    
                    # Return the trajectory and timestep for execution
                    return True, trajectory, result.interpolation_dt
                else:
                    print(f"All planning methods failed: {result.status}")
                    return False, None, None
        except Exception as e:
            print(f"Error in push/pull planning: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, None, None
            
    def get_robot_status(self):
        """Get the current status of the robot
        
        Returns:
            dict: Dictionary with robot status information
        """
        if self.arm is None:
            print("Robot not connected")
            return None
            
        try:
            with self.arm_lock:
                status = {
                    "mode": self.arm.mode,
                    "state": self.arm.state,
                    "error_code": self.arm.error_code,
                    "warn_code": self.arm.warn_code,
                    "has_error": self.arm.has_error,
                    "has_warn": self.arm.has_warn,
                    "connected": self.arm.connected,
                    "tcp_load": self.arm.tcp_load,
                    "collision_sensitivity": self.arm.collision_sensitivity,
                    "teach_sensitivity": self.arm.teach_sensitivity,
                    "motor_brake_states": self.arm.motor_brake_states,
                    "motor_enable_states": self.arm.motor_enable_states,
                    "cmd_num": self.arm.cmd_num,
                    "temperatures": self.arm.temperatures
                }
                
                # Add current joint positions
                code, angles = self.arm.get_servo_angle(is_radian=True)
                if code == 0 and angles is not None:
                    status["joint_positions"] = angles
                
                # Add current TCP pose
                code, pose = self.arm.get_position(is_radian=True)
                if code == 0 and pose is not None:
                    status["tcp_position"] = pose[:3]
                    status["tcp_orientation_euler"] = pose[3:]
                
                return status
        except Exception as e:
            print(f"Error getting robot status: {str(e)}")
            return None
    
    def clean_robot_error(self):
        """Clear any error or warning on the robot with improved error handling
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                print(f"Current robot state before cleaning: Mode={self.arm.mode}, State={self.arm.state}")
                print(f"Error status: has_error={self.arm.has_error}, error_code={self.arm.error_code}")
                print(f"Warning status: has_warn={self.arm.has_warn}, warn_code={self.arm.warn_code}")
                
                # Step 1: Clear error
                if self.arm.has_error:
                    print("Clearing robot error...")
                    code = self.arm.clean_error()
                    if code != 0:
                        print(f"Failed to clear error, error code: {code}")
                    else:
                        print("Error cleared successfully")
                    time.sleep(0.5)
                
                # Step 2: Clear warning
                if self.arm.has_warn:
                    print("Clearing robot warning...")
                    code = self.arm.clean_warn()
                    if code != 0:
                        print(f"Failed to clear warning, error code: {code}")
                    else:
                        print("Warning cleared successfully")
                    time.sleep(0.5)
                
                # Step 3: Re-enable motion
                print("Re-enabling robot motion...")
                code = self.arm.motion_enable(enable=True)
                if code != 0:
                    print(f"Failed to re-enable robot, error code: {code}")
                    return False
                time.sleep(0.5)
                
                # Step 4: Set mode to position control
                print("Setting robot to position control mode...")
                code = self.arm.set_mode(0)
                if code != 0:
                    print(f"Failed to set mode, error code: {code}")
                    return False
                time.sleep(0.2)
                
                # Step 5: Set to ready state - try multiple times if needed
                for attempt in range(3):
                    print(f"Setting robot to ready state (attempt {attempt + 1})...")
                    code = self.arm.set_state(0)
                    if code == 0:
                        time.sleep(0.5)
                        # Check if state actually changed
                        if self.arm.state == 0:
                            print("Robot successfully set to ready state")
                            return True
                        else:
                            print(f"State command succeeded but robot still in state {self.arm.state}")
                            time.sleep(1.0)  # Wait longer before next attempt
                    else:
                        print(f"Failed to set state (attempt {attempt + 1}), error code: {code}")
                        time.sleep(1.0)
                
                print(f"Failed to set robot to ready state after 3 attempts. Final state: {self.arm.state}")
                return False
                
        except Exception as e:
            print(f"Error clearing robot error: {str(e)}")
            return False
    
    def disconnect_robot(self):
        """Disconnect from the physical robot
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is None:
            print("Robot not connected")
            return False
            
        try:
            with self.arm_lock:
                # Remove callback
                self.arm.release_report_location_callback(self.joint_state_callback)
                
                # Disconnect
                self.arm.disconnect()
                self.arm = None
                
                print("Robot disconnected successfully")
                return True
        except Exception as e:
            print(f"Error disconnecting from robot: {str(e)}")
            return False
    
    def reconnect_robot(self):
        """Reconnect to the physical robot
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.arm is not None:
            print("Robot already connected")
            return True
            
        try:
            # Call the connect method
            self.connect_robot()
            
            if self.arm is not None:
                print("Robot reconnected successfully")
                return True
            else:
                print("Failed to reconnect to robot")
                return False
        except Exception as e:
            print(f"Error reconnecting to robot: {str(e)}")
            return False


def example_usage_with_robot():
    """Example usage of the CuRobo Motion Planner with a real xArm robot"""
    # Create the planner with robot IP
    robot_ip = "192.168.1.224"  # Replace with your robot's IP
    planner = CuRoboMotionPlanner(robot_ip=robot_ip)
    
    # Check robot status
    status = planner.get_robot_status()
    if status:
        print(f"Robot status: Mode={status['mode']}, State={status['state']}")
        
        # Clear any errors
        if status['has_error'] or status['has_warn']:
            print("Clearing robot errors...")
            planner.clean_robot_error()
    
    # Add a table below the robot for collision avoidance
    # planner.add_collision_object(
    #     name="table",
    #     dimensions=[0.8, 0.8, 0.05],
    #     position=[0.0, 0.0, -0.1]
    # )
    
    # Get current pose from the robot
    pose = planner.get_robot_tcp_pose()
    if pose:
        current_position, current_orientation = pose
        print(f"Current TCP pose: position={current_position}, orientation={current_orientation}")
    
    # Move to home position
    print("Moving to home position...")
    success, _, _ = planner.move_to_home(execute=True, speed_factor=0.5)
    if not success:
        print("Failed to move to home position")
    
    # Plan and execute motion to a target pose
    target_position = [0.3, 0.0, 0.3]  # x, y, z
    print(f"Moving to position {target_position}...")
    success, _, _ = planner.move_to_pose(
        target_position=target_position,
        execute=True,
        speed_factor=0.5
    )
    
    if success:
        print("Motion completed successfully")
        
        # Execute a pick and place operation
        pick_pos = [0.3, 0.1, 0.1]  # x, y, z position to pick
        place_pos = [0.3, -0.1, 0.1]  # x, y, z position to place
        
        print(f"Executing pick and place from {pick_pos} to {place_pos}...")
        pick_success = planner.execute_pick_and_place(
            pick_position=pick_pos,
            place_position=place_pos,
            speed_factor=0.5
        )
        
        if pick_success:
            print("Pick and place completed successfully")
        else:
            print("Pick and place failed")
    else:
        print("Motion failed")
    
    # Return to home position
    print("Returning to home position...")
    planner.move_to_home(execute=True, speed_factor=0.5)
    
    # Disconnect from the robot
    planner.disconnect_robot()
    print("Example completed")


if __name__ == "__main__":
    # Use simulation mode if no robot is available
    example_usage_with_robot()