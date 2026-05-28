#!/usr/bin/env python3
"""
ArUco Coordinate Frame Visualizer - Updated for Simplified Interface

This script visualizes all the coordinate frames calculated in the SimplifiedStereoArucoDetector
using PyBullet for 3D visualization and debugging.

Usage:
    python aruco_frame_visualizer.py --urdf path/to/robot.urdf --live
    python aruco_frame_visualizer.py --static  # Use static example transforms

Requirements:
    pip install pybullet numpy scipy
"""

import os
import pybullet as p
import pybullet_data
import numpy as np
import argparse
import time
import sys
from scipy.spatial.transform import Rotation
from typing import Dict, Optional, Tuple, List
import json


class ArucoFrameVisualizer:
    def __init__(self, urdf_path=None, gui=True):
        """Initialize the ArUco frame visualizer"""
        self.urdf_path = urdf_path
        self.robot_id = None
        
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            # Better camera angle for robotics
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=[0.5, 0, 0.5]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.ground = p.loadURDF("plane.urdf")
        
        # Load robot if URDF provided
        if urdf_path:
            try:
                self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
                print(f"✅ Successfully loaded robot URDF: {urdf_path}")
                
                # Get robot info
                self.num_joints = p.getNumJoints(self.robot_id)
                print(f"📊 Robot has {self.num_joints} joints")
                
            except Exception as e:
                print(f"⚠️  Failed to load URDF: {e}")
                print("🔄 Continuing without robot model...")
                self.robot_id = None
        
        # Storage for frame visualization objects
        self.frame_objects = {}
        self.text_objects = {}
        
        # Frame colors
        self.frame_colors = {
            'world_frame': [1.0, 1.0, 1.0],     # White - World frame
            'robot_base': [0.8, 0.8, 0.8],      # Light Gray - Robot base_link
            'camera_link': [1.0, 0.5, 0.0],     # Orange - Camera
            'aruco_frame': [1.0, 0.0, 1.0],     # Magenta
            'aruco_marker': [1.0, 1.0, 0.0],    # Yellow - ArUco markers
            'zed_camera': [0.0, 1.0, 1.0],      # Cyan - ZED camera
            'zed_frame': [0.5, 0.0, 1.0],       # Purple
        }

    def matrix_to_pose(self, transform_matrix):
        """Convert 4x4 transformation matrix to position and quaternion"""
        position = transform_matrix[:3, 3]
        rotation_matrix = transform_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        return position.tolist(), quaternion.tolist()

    def draw_coordinate_frame(self, position, orientation, scale=0.1, color_multiplier=1.0, frame_id="", duration=0):
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
        
        # Draw axes with color multiplier for dimming
        line_ids = []
        
        # X axis (Red)
        line_ids.append(p.addUserDebugLine(
            position, 
            position + x_axis,
            lineColorRGB=[1.0 * color_multiplier, 0, 0],
            lineWidth=4,
            lifeTime=duration
        ))
        
        # Y axis (Green)
        line_ids.append(p.addUserDebugLine(
            position,
            position + y_axis, 
            lineColorRGB=[0, 1.0 * color_multiplier, 0],
            lineWidth=4,
            lifeTime=duration
        ))
        
        # Z axis (Blue)
        line_ids.append(p.addUserDebugLine(
            position,
            position + z_axis,
            lineColorRGB=[0, 0, 1.0 * color_multiplier],
            lineWidth=4,
            lifeTime=duration
        ))
        
        return line_ids

    def add_coordinate_frame_with_label(self, position, orientation, label="Frame", 
                                      scale=0.1, color_multiplier=1.0, frame_id=""):
        """Add coordinate frame with text label"""
        # Draw the frame
        line_ids = self.draw_coordinate_frame(position, orientation, scale, color_multiplier, frame_id)
        
        # Get frame color for text
        text_color = self.frame_colors.get(frame_id, [1, 1, 1])
        
        # Add text label with offset
        text_id = p.addUserDebugText(
            label,
            [position[0], position[1], position[2] + scale + 0.05],
            textColorRGB=text_color,
            textSize=1.2,
            lifeTime=0
        )
        
        return line_ids, text_id

    def clear_frame(self, frame_name):
        """Clear a specific frame from visualization"""
        if frame_name in self.frame_objects:
            for line_id in self.frame_objects[frame_name]:
                p.removeUserDebugItem(line_id)
            del self.frame_objects[frame_name]
        
        if frame_name in self.text_objects:
            p.removeUserDebugItem(self.text_objects[frame_name])
            del self.text_objects[frame_name]

    def clear_all_frames(self):
        """Clear all coordinate frames and debug items"""
        # Clear stored frame objects
        for frame_name in list(self.frame_objects.keys()):
            self.clear_frame(frame_name)
        
        # Clear all debug lines and text (this removes ALL debug items)
        p.removeAllUserDebugItems()
        
        # Reset storage
        self.frame_objects = {}
        self.text_objects = {}
    
    def clear_all_marker_frames(self):
        """Clear all ArUco marker frames specifically"""
        # Clear any remaining marker-specific frames
        marker_frame_keys = [key for key in self.frame_objects.keys() 
                           if 'aruco_marker' in key]
        for key in marker_frame_keys:
            self.clear_frame(key)
        
        marker_text_keys = [key for key in self.text_objects.keys() 
                          if 'aruco_marker' in key]
        for key in marker_text_keys:
            if key in self.text_objects:
                try:
                    p.removeUserDebugItem(self.text_objects[key])
                except:
                    pass
                del self.text_objects[key]

    def visualize_aruco_transforms(self, detector, joint_positions=None):
        """
        Visualize coordinate frames from the StereoArucoDetector
        
        Args:
            detector: StereoArucoDetector instance
            joint_positions: Current robot joint positions in radians
        """
        print(f"\n🎯 Visualizing ArUco coordinate frames...")
        
        # Set robot joint positions if robot is loaded and positions provided
        if self.robot_id and joint_positions:
            try:
                self.set_joint_positions(joint_positions)
                print(f"📍 Updated robot pose: [{', '.join([f'{angle:.3f}' for angle in joint_positions[:7]])}]")
                
                # Allow physics to settle
                for _ in range(5):
                    p.stepSimulation()
            except Exception as e:
                print(f"⚠️  Failed to update robot pose: {e}")
        
        # Clear existing frames and ALL debug items
        self.clear_all_frames()
        
        # Extra safety: Clear any remaining marker frames
        self.clear_all_marker_frames()
        
        # 1. WORLD FRAME - The static global reference frame at origin
        world_frame_pos = [0.0, 0.0, 0.0]
        world_frame_quat = [0.0, 0.0, 0.0, 1.0]  # Identity orientation
        
        lines, text = self.add_coordinate_frame_with_label(
            world_frame_pos, world_frame_quat, 
            "World Frame (Global Origin)", scale=0.25, frame_id="world_frame"
        )
        self.frame_objects["world_frame"] = lines
        self.text_objects["world_frame"] = text
        print(f"📍 World Frame (Static Global Reference): [0.000, 0.000, 0.000]")
        
        # Add prominent visual indicators for world frame
        p.addUserDebugLine(
            [-0.15, 0, 0], [0.4, 0, 0],  # Extended X-axis (forward in world)
            lineColorRGB=[1.0, 0.0, 0.0], lineWidth=6, lifeTime=0
        )
        p.addUserDebugLine(
            [0, -0.15, 0], [0, 0.15, 0],  # Y-axis (left in world)
            lineColorRGB=[0.0, 1.0, 0.0], lineWidth=6, lifeTime=0
        )
        p.addUserDebugLine(
            [0, 0, 0], [0, 0, 0.3],  # Z-axis (up in world)
            lineColorRGB=[0.0, 0.0, 1.0], lineWidth=6, lifeTime=0
        )
        
        # 2. ROBOT BASE_LINK FRAME
        if self.robot_id:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            lines, text = self.add_coordinate_frame_with_label(
                base_pos, base_orn,
                "Robot Base Link", scale=0.18, frame_id="robot_base"
            )
            self.frame_objects["robot_base"] = lines
            self.text_objects["robot_base"] = text
            print(f"📍 Robot Base Link: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
        else:
            base_pos = [0.0, 0.0, 0.0]
            base_orn = [0.0, 0.0, 0.0, 1.0]
        
        # 3. CAMERA LINK FRAME
        if detector.T_robot_base_to_camera_link is not None:
            T_camera_link = detector.T_robot_base_to_camera_link
            
            # Transform camera position to world coordinates
            if self.robot_id:
                T_world_to_base = np.eye(4)
                T_world_to_base[:3, 3] = base_pos
                base_rotation = Rotation.from_quat(base_orn)
                T_world_to_base[:3, :3] = base_rotation.as_matrix()
                
                T_world_to_camera = T_world_to_base @ T_camera_link
                camera_pos_world, camera_quat_world = self.matrix_to_pose(T_world_to_camera)
            else:
                camera_pos_world, camera_quat_world = self.matrix_to_pose(T_camera_link)
            
            lines, text = self.add_coordinate_frame_with_label(
                camera_pos_world, camera_quat_world,
                "Robot Camera Link", scale=0.12, frame_id="camera_link"
            )
            self.frame_objects["camera_link"] = lines
            self.text_objects["camera_link"] = text
            print(f"📍 Camera Link (world coords): [{camera_pos_world[0]:.3f}, {camera_pos_world[1]:.3f}, {camera_pos_world[2]:.3f}]")
            
            # Draw connection from robot base to camera
            p.addUserDebugLine(
                base_pos, camera_pos_world,
                lineColorRGB=[1.0, 0.5, 0.0], lineWidth=3, lifeTime=0
            )
        else:
            print("❌ Camera link transform not available")
            camera_pos_world = None
        
        # 4. GET CURRENT DETECTION DATA FOR MARKER VISUALIZATION
        robot_detection, zed_detection = detector.get_detection_data()
        
        # 5. VISUALIZE ARUCO MARKERS (ROBOT SIDE)
        if (robot_detection is not None and 
            len(robot_detection['marker_poses']) > 0 and 
            camera_pos_world is not None):
            
            print(f"📍 Found {len(robot_detection['marker_poses'])} markers from robot camera")
            
            for i, (marker_id, T_aruco_frame_to_marker) in enumerate(robot_detection['marker_poses'].items()):
                # Transform marker from robot camera to world coordinates
                # Chain: T_world_to_marker = T_world_to_base @ T_base_to_camera @ T_camera_to_aruco_frame @ T_aruco_frame_to_marker
                
                if self.robot_id:
                    T_world_to_base = np.eye(4)
                    T_world_to_base[:3, 3] = base_pos
                    base_rotation = Rotation.from_quat(base_orn)
                    T_world_to_base[:3, :3] = base_rotation.as_matrix()
                    
                    # Full chain for robot side marker
                    T_world_to_marker_robot = (T_world_to_base @ 
                                             detector.T_robot_base_to_camera_link @ 
                                             detector.T_camera_link_to_aruco_frame @ 
                                             T_aruco_frame_to_marker)
                else:
                    T_world_to_marker_robot = (detector.T_robot_base_to_camera_link @ 
                                             detector.T_camera_link_to_aruco_frame @ 
                                             T_aruco_frame_to_marker)
                
                marker_pos_robot_world, marker_quat_robot_world = self.matrix_to_pose(T_world_to_marker_robot)
                
                lines, text = self.add_coordinate_frame_with_label(
                    marker_pos_robot_world, marker_quat_robot_world,
                    f"ArUco Marker {marker_id} (Robot)", scale=0.08, frame_id="aruco_marker"
                )
                self.frame_objects[f"aruco_marker_robot_{marker_id}"] = lines
                self.text_objects[f"aruco_marker_robot_{marker_id}"] = text
                print(f"📍 ArUco Marker {marker_id} (Robot, world coords): [{marker_pos_robot_world[0]:.3f}, {marker_pos_robot_world[1]:.3f}, {marker_pos_robot_world[2]:.3f}]")
                
                # Draw connection from camera to marker
                p.addUserDebugLine(
                    camera_pos_world, marker_pos_robot_world,
                    lineColorRGB=[1.0, 0.5, 0.0], lineWidth=2, lifeTime=0
                )
        else:
            print("❌ No ArUco markers detected from robot camera")
        
        # 6. ZED CAMERA FRAME
        zed_transform = detector.get_transformation_matrix()
        if zed_transform is not None:
            # T_zed_to_robot_base exists, so compute T_robot_base_to_zed
            T_zed_to_base = zed_transform
            T_base_to_zed = np.linalg.inv(T_zed_to_base)
            
            # Transform ZED to world coordinates
            if self.robot_id:
                T_world_to_base = np.eye(4)
                T_world_to_base[:3, 3] = base_pos
                base_rotation = Rotation.from_quat(base_orn)
                T_world_to_base[:3, :3] = base_rotation.as_matrix()
                
                T_world_to_zed = T_world_to_base @ T_base_to_zed
                zed_pos_world, zed_quat_world = self.matrix_to_pose(T_world_to_zed)
            else:
                zed_pos_world, zed_quat_world = self.matrix_to_pose(T_base_to_zed)
            
            lines, text = self.add_coordinate_frame_with_label(
                zed_pos_world, zed_quat_world,
                "ZED Camera", scale=0.12, frame_id="zed_camera"
            )
            self.frame_objects["zed_camera"] = lines
            self.text_objects["zed_camera"] = text
            print(f"📍 ZED Camera (world coords): [{zed_pos_world[0]:.3f}, {zed_pos_world[1]:.3f}, {zed_pos_world[2]:.3f}]")
            
            # Draw connection from world origin to ZED
            p.addUserDebugLine(
                world_frame_pos, zed_pos_world,
                lineColorRGB=[0.0, 1.0, 1.0], lineWidth=2, lifeTime=0
            )
            
            # 7. VISUALIZE ARUCO MARKERS (ZED SIDE)
            if (zed_detection is not None and 
                len(zed_detection['marker_poses']) > 0):
                
                print(f"📍 Found {len(zed_detection['marker_poses'])} markers from ZED camera")
                
                for i, (marker_id, T_aruco_frame_to_marker) in enumerate(zed_detection['marker_poses'].items()):
                    # Transform marker from ZED to world coordinates
                    # Chain: T_world_to_marker = T_world_to_zed @ T_zed_to_aruco_frame @ T_aruco_frame_to_marker
                    
                    T_world_to_marker_zed = (T_world_to_zed @ 
                                           detector.T_zed_to_aruco_frame @ 
                                           T_aruco_frame_to_marker)
                    
                    marker_pos_zed_world, marker_quat_zed_world = self.matrix_to_pose(T_world_to_marker_zed)
                    
                    lines, text = self.add_coordinate_frame_with_label(
                        marker_pos_zed_world, marker_quat_zed_world,
                        f"ArUco Marker {marker_id} (ZED)", scale=0.06, frame_id="aruco_marker", 
                        color_multiplier=0.7
                    )
                    self.frame_objects[f"aruco_marker_zed_{marker_id}"] = lines
                    self.text_objects[f"aruco_marker_zed_{marker_id}"] = text
                    print(f"📍 ArUco Marker {marker_id} (ZED, world coords): [{marker_pos_zed_world[0]:.3f}, {marker_pos_zed_world[1]:.3f}, {marker_pos_zed_world[2]:.3f}]")
                    
                    # Draw connection from ZED to marker
                    p.addUserDebugLine(
                        zed_pos_world, marker_pos_zed_world,
                        lineColorRGB=[0.0, 1.0, 1.0], lineWidth=2, lifeTime=0
                    )
                    
                    # Check alignment between robot and ZED marker detections if same marker ID
                    if (robot_detection is not None and 
                        marker_id in robot_detection['marker_poses']):
                        
                        # Get robot marker position
                        T_aruco_frame_to_marker_robot = robot_detection['marker_poses'][marker_id]
                        if self.robot_id:
                            T_world_to_base = np.eye(4)
                            T_world_to_base[:3, 3] = base_pos
                            base_rotation = Rotation.from_quat(base_orn)
                            T_world_to_base[:3, :3] = base_rotation.as_matrix()
                            
                            T_world_to_marker_robot = (T_world_to_base @ 
                                                     detector.T_robot_base_to_camera_link @ 
                                                     detector.T_camera_link_to_aruco_frame @ 
                                                     T_aruco_frame_to_marker_robot)
                        else:
                            T_world_to_marker_robot = (detector.T_robot_base_to_camera_link @ 
                                                     detector.T_camera_link_to_aruco_frame @ 
                                                     T_aruco_frame_to_marker_robot)
                        
                        marker_pos_robot_world_check, _ = self.matrix_to_pose(T_world_to_marker_robot)
                        
                        # Calculate alignment error
                        robot_marker_pos = np.array(marker_pos_robot_world_check)
                        zed_marker_pos = np.array(marker_pos_zed_world)
                        alignment_error = np.linalg.norm(robot_marker_pos - zed_marker_pos)
                        
                        print(f"🎯 Marker {marker_id} alignment check:")
                        print(f"   Robot position: [{robot_marker_pos[0]:.4f}, {robot_marker_pos[1]:.4f}, {robot_marker_pos[2]:.4f}]")
                        print(f"   ZED position:   [{zed_marker_pos[0]:.4f}, {zed_marker_pos[1]:.4f}, {zed_marker_pos[2]:.4f}]")
                        print(f"   Alignment error: {alignment_error:.4f}m")
                        
                        # Draw error visualization if significant
                        if alignment_error > 0.01:  # 1cm threshold
                            p.addUserDebugLine(
                                marker_pos_robot_world_check, marker_pos_zed_world,
                                lineColorRGB=[1.0, 0.0, 0.0], lineWidth=3, lifeTime=0
                            )
                            # Add error text
                            mid_point = [(robot_marker_pos[i] + zed_marker_pos[i])/2 for i in range(3)]
                            p.addUserDebugText(
                                f"Fresh Error: {alignment_error:.3f}m",
                                [mid_point[0], mid_point[1], mid_point[2] + 0.1],
                                textColorRGB=[1, 0, 0], textSize=1.0, lifeTime=0
                            )
                        else:
                            print(f"   ✅ Good alignment (< 1cm threshold)")
            else:
                print("❌ No ArUco markers detected from ZED camera")
        else:
            print("❌ ZED to robot base transform not available")
        
        # Add status information panel
        self.add_detection_status_panel(detector, robot_detection, zed_detection)
        
        print(f"🎯 Coordinate frame visualization complete")

    def add_detection_status_panel(self, detector, robot_detection, zed_detection):
        """Add a status panel showing detection information"""
        info_lines = []
        info_lines.append("=== DETECTION STATUS ===")
        info_lines.append("")
        
        # Robot camera status
        if robot_detection is not None:
            robot_markers = len(robot_detection['marker_poses'])
            info_lines.append(f"🤖 Robot Camera: {robot_markers} markers")
            if robot_markers > 0:
                marker_ids = list(robot_detection['marker_poses'].keys())
                info_lines.append(f"   IDs: {marker_ids}")
        else:
            info_lines.append("🤖 Robot Camera: No data")
        
        # ZED camera status  
        if zed_detection is not None:
            zed_markers = len(zed_detection['marker_poses'])
            info_lines.append(f"📷 ZED Camera: {zed_markers} markers")
            if zed_markers > 0:
                marker_ids = list(zed_detection['marker_poses'].keys())
                info_lines.append(f"   IDs: {marker_ids}")
        else:
            info_lines.append("📷 ZED Camera: No data")
        
        # Common markers
        if robot_detection is not None and zed_detection is not None:
            common_markers = set(robot_detection['marker_poses'].keys()).intersection(
                set(zed_detection['marker_poses'].keys()))
            info_lines.append(f"🎯 Common markers: {list(common_markers)}")
        else:
            info_lines.append("🎯 Common markers: None")
        
        # Transform status
        transform_available = detector.get_transformation_matrix() is not None
        info_lines.append(f"🔄 Transform: {'✅ Available' if transform_available else '❌ None'}")
        
        info_lines.append("")
        info_lines.append("FRAMES SHOWN:")
        info_lines.append("• World (thick RGB axes)")
        info_lines.append("• Robot Base (gray)")
        info_lines.append("• Camera Link (orange)")
        info_lines.append("• ZED Camera (cyan)")
        info_lines.append("• ArUco Markers (yellow/red)")
        
        # Display the panel
        panel_text = "\n".join(info_lines)
        p.addUserDebugText(
            panel_text,
            [-1.2, 0.3, 1.5],
            textColorRGB=[1, 1, 1],
            textSize=0.7,
            lifeTime=0
        )

    def set_joint_positions(self, joint_positions):
        """Set joint positions for the robot"""
        if not self.robot_id:
            return
            
        num_joints = p.getNumJoints(self.robot_id)
        
        # Only set the first N joints that correspond to active DOF
        # Assume the first 7 joints are the active arm joints
        num_active_joints = min(len(joint_positions), 7)
        
        for i in range(num_active_joints):
            try:
                joint_info = p.getJointInfo(self.robot_id, i+1)
                if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    p.resetJointState(self.robot_id, i+1, joint_positions[i])
            except Exception as e:
                # Skip joints that can't be set
                continue

    def run_interactive_with_aruco_detector(self, detector, joint_positions=None):
        """
        Run interactive visualization with ArUco detector
        
        Args:
            detector: StereoArucoDetector instance
            joint_positions: Optional joint positions for robot
        """
        print(f"\n🎮 Interactive ArUco frame visualization started")
        print(f"   - Mouse: rotate/zoom view")
        print(f"   - 'r': reset view")
        print(f"   - 'u': update frames from detector")
        print(f"   - 'c': compute transformation")
        print(f"   - 's': save current transform to JSON")
        print(f"   - 'q': quit")
        
        last_update_time = 0
        update_interval = 0.5  # Update every 500ms
        
        try:
            while True:
                current_time = time.time()
                
                # Auto-update frames from detector
                if current_time - last_update_time > update_interval:
                    self.visualize_aruco_transforms(detector, joint_positions)
                    last_update_time = current_time
                
                # Handle keyboard input
                keys = p.getKeyboardEvents()
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    break
                elif ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    print("🔄 Resetting camera view...")
                    p.resetDebugVisualizerCamera(
                        cameraDistance=2.0, cameraYaw=45, cameraPitch=-20,
                        cameraTargetPosition=[0.5, 0, 0.5]
                    )
                elif ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
                    print("🔄 Updating frames from detector...")
                    self.visualize_aruco_transforms(detector, joint_positions)
                elif ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                    print("🔄 Computing transformation and updating visualization...")
                    transform = detector.compute_transformation()
                    if transform is not None:
                        print("✅ Transformation computed successfully")
                        # Force immediate visualization update with fresh data
                        self.visualize_aruco_transforms(detector, joint_positions)
                    else:
                        print("❌ Failed to compute transformation")
                elif ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                    transform = detector.get_transformation_matrix()
                    if transform is not None:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"simplified_aruco_transform_{timestamp}.json"
                        
                        # Save transformation data
                        transform_data = {
                            'timestamp': timestamp,
                            'T_zed_to_robot_base': transform.tolist(),
                            'T_robot_base_to_camera_link': detector.T_robot_base_to_camera_link.tolist() if detector.T_robot_base_to_camera_link is not None else None
                        }
                        
                        with open(filename, 'w') as f:
                            json.dump(transform_data, f, indent=2)
                        print(f"💾 Saved transform to {filename}")
                    else:
                        print("❌ No transformation available to save")
                
                time.sleep(1/60)  # 60 Hz update rate
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        
        self.cleanup()

    def run_static_demo(self):
        """Run demonstration with static example transforms"""
        print(f"\n🎯 Running static ArUco frame demonstration...")
        
        # Create a mock detector for demonstration
        class MockDetector:
            def __init__(self):
                self.T_robot_base_to_camera_link = np.array([
                    [0.0, -1.0, 0.0, 0.3],
                    [0.0, 0.0, -1.0, 0.1],    
                    [1.0, 0.0, 0.0, 0.8],     
                    [0.0, 0.0, 0.0, 1.0]      
                ])
                self.T_zed_to_robot_base = np.array([
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 1.0, 0.0, -0.5],     
                    [0.0, 0.0, 1.0, 0.3],     
                    [0.0, 0.0, 0.0, 1.0]      
                ])
            
            def get_transformation_matrix(self):
                return self.T_zed_to_robot_base
        
        mock_detector = MockDetector()
        
        # Visualize the frames
        self.visualize_aruco_transforms(mock_detector)
        
        print(f"\n🎮 Static demo controls:")
        print(f"   - Mouse: rotate/zoom view")
        print(f"   - 'r': reset view") 
        print(f"   - 'q': quit")
        
        try:
            while True:
                keys = p.getKeyboardEvents()
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    break
                elif ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    p.resetDebugVisualizerCamera(
                        cameraDistance=2.0, cameraYaw=45, cameraPitch=-20,
                        cameraTargetPosition=[0.5, 0, 0.5]
                    )
                
                time.sleep(1/60)
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        
        self.cleanup()

    def cleanup(self):
        """Clean up PyBullet connection"""
        self.clear_all_frames()
        p.disconnect(self.client)
        print("✅ Cleaned up PyBullet connection")


def main():
    parser = argparse.ArgumentParser(description="Visualize ArUco coordinate frames")
    parser.add_argument("--urdf", help="Path to robot URDF file",
                       default=os.environ.get("CUROBO_XARM7_URDF", ""))
    parser.add_argument("--joints", default="0.0021,-1.2677,-0.0341,0.8922,0.0197,1.5883,-0.0050", 
                       help="Joint positions (comma-separated)")
    parser.add_argument("--static", action="store_true", help="Run static demo with example transforms")
    parser.add_argument("--live", action="store_true", help="Connect to live ArUco detector")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    
    args = parser.parse_args()
    
    # Parse joint positions
    joint_positions = None
    if args.joints:
        try:
            joint_positions = [float(x.strip()) for x in args.joints.split(',')]
        except ValueError:
            print("❌ Invalid joint positions format. Use comma-separated numbers.")
            sys.exit(1)
    
    # Create visualizer
    urdf_path = args.urdf if args.urdf else None
    visualizer = ArucoFrameVisualizer(urdf_path, gui=not args.no_gui)
    
    if args.static:
        # Run static demonstration
        visualizer.run_static_demo()
    elif args.live:
        # Try to import and connect to live ArUco detector
        try:
            # Import the simplified ArUco detector
            sys.path.append('.')
            from cognitive_bt_framework.src.vision.aruco_tf_v3 import StereoArucoDetector
            
            print("🔄 Setting up live simplified ArUco detector connection...")
            print("⚠️  Note: This requires the ArUco detector to be running with cameras")
            
            # You would initialize your stereo detector here
            # detector = StereoArucoDetector(...)
            # visualizer.run_interactive_with_aruco_detector(detector, joint_positions)
            
            print("❌ Live mode not fully implemented - please integrate with your ArUco detector")
            visualizer.run_static_demo()
            
        except ImportError as e:
            print(f"❌ Could not import ArUco detector: {e}")
            print("🔄 Running static demo instead...")
            visualizer.run_static_demo()
    else:
        # Default to static demo
        visualizer.run_static_demo()


if __name__ == "__main__":
    main()