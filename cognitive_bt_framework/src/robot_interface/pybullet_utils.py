import pybullet as p
import pybullet_data
import numpy as np
import time
from xarm_curobo_interface import RobotConfig

# Initialize PyBullet - use GUI mode for visualization or DIRECT for headless operation
physics_client = p.connect(p.GUI)  # or p.DIRECT for no visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity and other simulation properties
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(1)  # 1 for real-time, 0 for stepped simulation


def initialize_default_config(self) -> RobotConfig:
        """Initialize default configuration for the planner"""
        config = RobotConfig()
        
        # Set default parameters
        config.dof = 6
        config.robot_type = "lite"
        config.prefix = ""
        config.robot_ip = "192.168.1.153"
        
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
    


def setup_robot_and_camera():
    # Load robot - replace with your URDF
    robot_id = p.loadURDF("your_robot.urdf", useFixedBase=True)
    
    # Get robot base link index
    base_link_index = -1
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[12].decode('utf-8') == 'L_link_base':
            base_link_index = i
            break
    
    # Setup camera - this creates a virtual camera in the simulation
    camera_position = [0.5, 0, 0.5]
    camera_target = [0, 0, 0]
    camera_up_vector = [0, 0, 1]
    camera_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=base_link_index,  # Attach to the appropriate link
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=camera_position,
        childFramePosition=[0, 0, 0],
        childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )
    
    return robot_id, camera_id, base_link_index


def transform_pose_to_base(robot_id, position, orientation, source_link_index, target_link_index):
    """Transform pose from source link frame to target link frame"""
    
    # Get the current state of the source link
    source_state = p.getLinkState(robot_id, source_link_index)
    source_pos, source_orient = source_state[0], source_state[1]
    
    # Get the current state of the target link
    target_state = p.getLinkState(robot_id, target_link_index)
    target_pos, target_orient = target_state[0], target_state[1]
    
    # Convert input orientation to quaternion if it's in Euler format
    if len(orientation) == 3:
        quat = p.getQuaternionFromEuler(orientation)
    else:
        quat = orientation
        
    # Create transform matrices
    source_transform = p.invertTransform(source_pos, source_orient)
    target_transform = p.invertTransform(target_pos, target_orient)
    
    # Apply transformations
    local_pos, local_orient = p.multiplyTransforms(
        source_transform[0], source_transform[1], 
        position, quat
    )
    
    world_pos, world_orient = p.multiplyTransforms(
        target_pos, target_orient,
        local_pos, local_orient
    )
    
    # If original orientation was Euler, convert back to Euler
    if len(orientation) == 3:
        world_orient_euler = p.getEulerFromQuaternion(world_orient)
        return np.array(world_pos), np.array(world_orient_euler)
    
    return np.array(world_pos), np.array(world_orient)

def transform_points_between_frames(points, robot_id, source_link_index, target_link_index):
    """Transform point cloud from source link frame to target link frame"""
    
    # Get transforms for source and target links
    source_state = p.getLinkState(robot_id, source_link_index)
    source_pos, source_orient = source_state[0], source_state[1]
    
    target_state = p.getLinkState(robot_id, target_link_index)
    target_pos, target_orient = target_state[0], target_state[1]
    
    # Inverse source transform (source to world)
    inv_source_pos, inv_source_orient = p.invertTransform(source_pos, source_orient)
    
    # Process each point
    transformed_points = []
    for point in points:
        # Extract XYZ (and RGB if available)
        if len(point) > 3:
            pt_xyz, pt_rgb = point[:3], point[3:]
        else:
            pt_xyz, pt_rgb = point, None
            
        # Source frame to world frame
        world_pos, _ = p.multiplyTransforms(
            inv_source_pos, inv_source_orient,
            pt_xyz, [0, 0, 0, 1]
        )
        
        # World frame to target frame
        target_pos, _ = p.multiplyTransforms(
            target_pos, target_orient,
            world_pos, [0, 0, 0, 1]
        )
        
        # Add RGB data if available
        if pt_rgb is not None:
            transformed_points.append(np.concatenate([target_pos, pt_rgb]))
        else:
            transformed_points.append(target_pos)
            
    return np.array(transformed_points)

def transform_surface_normal(normal, robot_id, source_link_index, target_link_index):
    """Transform a surface normal from source link frame to target link frame"""
    
    # Get the rotation matrices from link states
    source_state = p.getLinkState(robot_id, source_link_index)
    source_orient = source_state[1]
    
    target_state = p.getLinkState(robot_id, target_link_index)
    target_orient = target_state[1]
    
    # Convert orientations to rotation matrices
    source_rot = np.array(p.getMatrixFromQuaternion(source_orient)).reshape(3, 3)
    target_rot = np.array(p.getMatrixFromQuaternion(target_orient)).reshape(3, 3)
    
    # Calculate the rotation from source to target
    rotation = np.dot(target_rot.T, source_rot)
    
    # Apply rotation to normal
    transformed_normal = np.dot(rotation, normal)
    
    # Normalize
    norm = np.linalg.norm(transformed_normal)
    if norm > 1e-6:
        transformed_normal = transformed_normal / norm
        
    return transformed_normal


