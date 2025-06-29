#!/usr/bin/env python3
"""
URDF Coordinate Frame Visualizer

This script loads a URDF file, sets joint positions, and visualizes coordinate frames
with position and orientation. Perfect for debugging robot transformations.

Usage:
    python urdf_visualizer.py --urdf path/to/robot.urdf --joints 0,0,0,0,0,0 --pos 0.3,0.1,0.3 --quat 0,1,0,0

Requirements:
    pip install pybullet numpy scipy argparse
"""

import pybullet as p
import pybullet_data
import numpy as np
import argparse
import time
import sys
from scipy.spatial.transform import Rotation

class URDFVisualizer:
    def __init__(self, urdf_path, gui=True):
        """Initialize the URDF visualizer"""
        self.urdf_path = urdf_path
        
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            # Better camera angle
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.ground = p.loadURDF("plane.urdf")
        
        # Load robot
        try:
            self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0], useFixedBase=True)
            print(f"✅ Successfully loaded URDF: {self.urdf_path}")
        except Exception as e:
            print(f"❌ Failed to load URDF: {e}")
            sys.exit(1)
        
        # Get robot info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_info = {}
        self.joint_names = []
        
        print(f"\n📊 Robot has {self.num_joints} joints:")
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            self.joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower_limit': joint_info[8],
                'upper_limit': joint_info[9]
            }
            self.joint_names.append(joint_name)
            
            type_name = {
                0: "REVOLUTE",
                1: "PRISMATIC", 
                4: "FIXED"
            }.get(joint_type, f"TYPE_{joint_type}")
            
            print(f"  Joint {i}: {joint_name} ({type_name})")
    
    def set_joint_positions(self, joint_positions):
        """Set joint positions for the robot"""
        if len(joint_positions) != self.num_joints:
            print(f"⚠️  Warning: Expected {self.num_joints} joint positions, got {len(joint_positions)}")
            # Pad with zeros or truncate
            joint_positions = (list(joint_positions) + [0] * self.num_joints)[:self.num_joints]
        
        print(f"\n🔧 Setting joint positions: {joint_positions}")
        
        for i, pos in enumerate(joint_positions):
            joint_info = self.joint_info[i]
            if joint_info['type'] in [0, 1]:  # REVOLUTE or PRISMATIC
                p.resetJointState(self.robot_id, i, pos)
                print(f"  Joint {i} ({joint_info['name']}): {pos:.3f}")
    
    def get_link_pose(self, link_name):
        """Get pose of a specific link by name"""
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode('utf-8') == link_name:  # Link name
                link_state = p.getLinkState(self.robot_id, i)
                return link_state[0], link_state[1]  # Position, orientation
        
        print(f"⚠️  Link '{link_name}' not found")
        return None, None
    
    def draw_coordinate_frame(self, position, orientation, scale=0.1, duration=0):
        """Draw coordinate frame axes at given position and orientation"""
        if isinstance(orientation, (list, tuple)):
            orientation = np.array(orientation)
        
        # Convert quaternion to rotation matrix
        if len(orientation) == 4:
            # Assume quaternion [x, y, z, w]
            rot = Rotation.from_quat(orientation)
            rotation_matrix = rot.as_matrix()
        else:
            raise ValueError("Orientation must be a quaternion [x, y, z, w]")
        
        # Define axis vectors
        x_axis = rotation_matrix[:, 0] * scale  # Red - X axis
        y_axis = rotation_matrix[:, 1] * scale  # Green - Y axis  
        z_axis = rotation_matrix[:, 2] * scale  # Blue - Z axis
        
        # Draw axes
        line_ids = []
        
        # X axis (Red)
        line_ids.append(p.addUserDebugLine(
            position, 
            position + x_axis,
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            lifeTime=duration
        ))
        
        # Y axis (Green)
        line_ids.append(p.addUserDebugLine(
            position,
            position + y_axis, 
            lineColorRGB=[0, 1, 0],
            lineWidth=3,
            lifeTime=duration
        ))
        
        # Z axis (Blue)
        line_ids.append(p.addUserDebugLine(
            position,
            position + z_axis,
            lineColorRGB=[0, 0, 1],
            lineWidth=3,
            lifeTime=duration
        ))
        
        return line_ids
    
    def add_coordinate_frame_with_label(self, position, orientation, label="Frame", scale=0.1):
        """Add coordinate frame with text label"""
        # Draw the frame
        line_ids = self.draw_coordinate_frame(position, orientation, scale, duration=0)
        
        # Add text label
        text_id = p.addUserDebugText(
            label,
            [position[0], position[1], position[2] + scale + 0.05],
            textColorRGB=[1, 1, 1],
            textSize=1.0,
            lifeTime=0
        )
        
        return line_ids, text_id
    
    def visualize_robot_with_frames(self, joint_positions, frames_to_show):
        """
        Visualize robot with coordinate frames
        
        Args:
            joint_positions: List of joint positions
            frames_to_show: List of dicts with keys: 'position', 'orientation', 'label'
        """
        # Set joint positions
        self.set_joint_positions(joint_positions)
        
        # Allow physics to settle
        for _ in range(10):
            p.stepSimulation()
        
        # Draw coordinate frames
        frame_objects = []
        for frame in frames_to_show:
            position = frame['position']
            orientation = frame['orientation'] 
            label = frame.get('label', 'Frame')
            scale = frame.get('scale', 0.1)
            
            print(f"\n📍 Adding coordinate frame: {label}")
            print(f"   Position: {position}")
            print(f"   Orientation (quat): {orientation}")
            
            lines, text = self.add_coordinate_frame_with_label(
                position, orientation, label, scale
            )
            frame_objects.append((lines, text))
        
        return frame_objects
    
    def run_interactive(self, joint_positions, frames_to_show):
        """Run interactive visualization"""
        print(f"\n🎮 Interactive mode started")
        print(f"   - Use mouse to rotate/zoom the view")
        print(f"   - Press 'r' to reset joint positions")
        print(f"   - Press 'q' or close window to quit")
        
        frame_objects = self.visualize_robot_with_frames(joint_positions, frames_to_show)
        
        try:
            while True:
                # Handle keyboard input
                keys = p.getKeyboardEvents()
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    break
                elif ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    print("🔄 Resetting joint positions...")
                    self.set_joint_positions([0] * self.num_joints)
                
                # p.stepSimulation()
                # time.sleep(1/240)  # 240 Hz
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up PyBullet connection"""
        p.disconnect(self.client)
        print("✅ Cleaned up PyBullet connection")


def main():
    parser = argparse.ArgumentParser(description="Visualize URDF with coordinate frames")
    parser.add_argument("--urdf", help="Path to URDF file", default="/home/liam/installs/curobo/src/curobo/content/assets/robot/xarm7/xarm7.urdf")
    parser.add_argument("--joints", default="0.0021, -1.2677, -0.0341,  0.8922,  0.0197,  1.5883, -0.0050", help="Joint positions (comma-separated)")
    parser.add_argument("--pos", default="0.16382727, 0.0065152114, 0.6358539", help="Frame position (x,y,z)")
    parser.add_argument("--quat", default="-0.67583245, 0.68075216, -0.18272126, 0.21549949", help="Frame orientation quaternion (x,y,z,w)")
    parser.add_argument("--label", default="Target Frame", help="Label for the coordinate frame")
    parser.add_argument("--scale", type=float, default=0.1, help="Scale for coordinate axes")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    
    args = parser.parse_args()
    
    # Parse joint positions
    try:
        joint_positions = [float(x.strip()) for x in args.joints.split(',')]
    except ValueError:
        print("❌ Invalid joint positions format. Use comma-separated numbers.")
        sys.exit(1)
    
    # Parse position
    try:
        position = [float(x.strip()) for x in args.pos.split(',')]
        if len(position) != 3:
            raise ValueError("Position must have 3 values")
    except ValueError:
        print("❌ Invalid position format. Use x,y,z")
        sys.exit(1)
    
    # Parse quaternion
    try:
        quaternion = [float(x.strip()) for x in args.quat.split(',')]
        if len(quaternion) != 4:
            raise ValueError("Quaternion must have 4 values")
    except ValueError:
        print("❌ Invalid quaternion format. Use x,y,z,w")
        sys.exit(1)
    
    # Create visualizer
    visualizer = URDFVisualizer(args.urdf, gui=not args.no_gui)
    
    # Define frames to show
    frames_to_show = [{
        'position': position,
        'orientation': quaternion,
        'label': args.label,
        'scale': args.scale
    }]
    
    # Run visualization
    if args.no_gui:
        visualizer.visualize_robot_with_frames(joint_positions, frames_to_show)
        print("✅ Visualization complete (no GUI mode)")
    else:
        visualizer.run_interactive(joint_positions, frames_to_show)


if __name__ == "__main__":
    main()


