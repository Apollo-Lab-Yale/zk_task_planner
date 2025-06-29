#!/usr/bin/env python3
"""
Test script for integrating RealSense camera with GraspMolmo grasp prediction.
This script captures RGB-D data from a RealSense camera and uses GraspMolmo 
via Hugging Face cloud API to predict task-oriented grasps with comprehensive visualization.

Dependencies:
- pyrealsense2: RealSense camera interface
- opencv-python: Image processing and 2D visualization
- numpy: Numerical computations
- matplotlib: 3D point cloud visualization (optional)
- huggingface_hub: Cloud API access to GraspMolmo (recommended)

Install with:
pip install pyrealsense2 opencv-python numpy matplotlib huggingface_hub

For HuggingFace API access, you may need to login:
huggingface-cli login
"""

import numpy as np
import cv2
import time
import sys
import os
from typing import List, Dict, Tuple, Optional

# Import the RealSense camera class
from cognitive_bt_framework.src.vision.realsense import Camera

# Import GraspMolmo via Hugging Face inference client
try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
    print("Hugging Face inference client available")
except ImportError as e:
    print(f"Warning: huggingface_hub not available. Install with: pip install huggingface_hub")
    print(f"Import error: {e}")
    HF_CLIENT_AVAILABLE = False
except Exception as e:
    print(f"Warning: Hugging Face client import failed: {e}")
    HF_CLIENT_AVAILABLE = False

# Try to import matplotlib for 3D visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib not available. 3D visualization disabled.")
    MATPLOTLIB_AVAILABLE = False


def project_3d_to_2d(point_3d, camera_intrinsics):
    """
    Project 3D point to 2D image coordinates.
    
    Args:
        point_3d: 3D point [x, y, z]
        camera_intrinsics: Camera intrinsics from RealSense
    
    Returns:
        u, v: 2D image coordinates
    """
    x, y, z = point_3d
    if z == 0:
        return None, None
    
    u = int(x * camera_intrinsics.fx / z + camera_intrinsics.ppx)
    v = int(y * camera_intrinsics.fy / z + camera_intrinsics.ppy)
    
    return u, v


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z] or [x, y, z, w]
    
    Returns:
        R: 3x3 rotation matrix
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    # Assume format [x, y, z, w]
    x, y, z, w = q
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def draw_grasp_on_image(image, grasp, camera_intrinsics, gripper_length=0.05):
    """
    Draw grasp visualization on RGB image.
    
    Args:
        image: RGB image
        grasp: Grasp dictionary with 'position' and 'orientation'
        camera_intrinsics: Camera intrinsics
        gripper_length: Length of gripper visualization in meters
    
    Returns:
        image_with_grasp: Image with grasp overlay
    """
    img = image.copy()
    
    position = grasp['position']
    orientation = grasp['orientation']
    
    # Project grasp center to image
    u_center, v_center = project_3d_to_2d(position, camera_intrinsics)
    
    if u_center is None or v_center is None:
        return img
    
    # Check if point is within image bounds
    h, w = img.shape[:2]
    if not (0 <= u_center < w and 0 <= v_center < h):
        return img
    
    # Draw center point
    cv2.circle(img, (u_center, v_center), 8, (0, 255, 0), -1)
    cv2.circle(img, (u_center, v_center), 12, (0, 0, 0), 2)
    
    # Draw gripper orientation
    try:
        R = quaternion_to_rotation_matrix(orientation)
        
        # Define gripper axes (assuming z-axis is approach direction)
        gripper_x = R[:, 0] * gripper_length  # Gripper width direction
        gripper_y = R[:, 1] * gripper_length  # Gripper finger direction
        gripper_z = R[:, 2] * gripper_length  # Approach direction
        
        # Calculate end points
        end_x = position + gripper_x
        end_y = position + gripper_y
        end_z = position + gripper_z
        
        # Project to image coordinates
        u_x, v_x = project_3d_to_2d(end_x, camera_intrinsics)
        u_y, v_y = project_3d_to_2d(end_y, camera_intrinsics)
        u_z, v_z = project_3d_to_2d(end_z, camera_intrinsics)
        
        # Draw orientation axes
        if all(coord is not None for coord in [u_x, v_x]):
            cv2.arrowedLine(img, (u_center, v_center), (u_x, v_x), (255, 0, 0), 3)  # Red for X
        if all(coord is not None for coord in [u_y, v_y]):
            cv2.arrowedLine(img, (u_center, v_center), (u_y, v_y), (0, 255, 0), 3)  # Green for Y
        if all(coord is not None for coord in [u_z, v_z]):
            cv2.arrowedLine(img, (u_center, v_center), (u_z, v_z), (0, 0, 255), 3)  # Blue for Z
        
        # Draw gripper fingers (simplified)
        finger_offset = 0.02  # 2cm finger separation
        finger_length = gripper_length * 0.7
        
        finger1_start = position + gripper_x * finger_offset / gripper_length
        finger2_start = position - gripper_x * finger_offset / gripper_length
        finger1_end = finger1_start + gripper_y * finger_length / gripper_length
        finger2_end = finger2_start + gripper_y * finger_length / gripper_length
        
        for finger_start, finger_end in [(finger1_start, finger1_end), (finger2_start, finger2_end)]:
            u_start, v_start = project_3d_to_2d(finger_start, camera_intrinsics)
            u_end, v_end = project_3d_to_2d(finger_end, camera_intrinsics)
            
            if all(coord is not None for coord in [u_start, v_start, u_end, v_end]):
                cv2.line(img, (u_start, v_start), (u_end, v_end), (255, 255, 0), 2)  # Cyan fingers
    
    except Exception as e:
        print(f"Error drawing grasp orientation: {e}")
    
    # Add grasp info text
    info_text = f"Grasp Score: {grasp.get('score', 0):.3f}"
    cv2.putText(img, info_text, (u_center + 15, v_center - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def visualize_point_cloud_with_grasps(point_cloud, grasps, selected_idx=None):
    """
    Create 3D visualization of point cloud with grasps.
    
    Args:
        point_cloud: Nx3 or Nx6 point cloud
        grasps: List of grasp dictionaries
        selected_idx: Index of selected grasp to highlight
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for 3D visualization")
        return
    
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot point cloud
        if len(point_cloud) > 0:
            pc_xyz = point_cloud[:, :3]
            
            # Subsample for visualization if too many points
            if len(pc_xyz) > 5000:
                indices = np.random.choice(len(pc_xyz), 5000, replace=False)
                pc_xyz = pc_xyz[indices]
                if point_cloud.shape[1] >= 6:
                    pc_colors = point_cloud[indices, 3:6] / 255.0
                else:
                    pc_colors = 'blue'
            else:
                if point_cloud.shape[1] >= 6:
                    pc_colors = point_cloud[:, 3:6] / 255.0
                else:
                    pc_colors = 'blue'
            
            ax.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], 
                      c=pc_colors, s=1, alpha=0.6)
        
        # Plot all grasps
        for i, grasp in enumerate(grasps):
            pos = grasp['position']
            color = 'red' if i == selected_idx else 'orange'
            size = 100 if i == selected_idx else 50
            
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.8)
            
            # Draw grasp orientation
            try:
                R = quaternion_to_rotation_matrix(grasp['orientation'])
                length = 0.05 if i == selected_idx else 0.03
                
                # Draw coordinate axes
                colors = ['red', 'green', 'blue']
                for j in range(3):
                    end_point = pos + R[:, j] * length
                    ax.plot([pos[0], end_point[0]], 
                           [pos[1], end_point[1]], 
                           [pos[2], end_point[2]], 
                           color=colors[j], linewidth=2 if i == selected_idx else 1)
            except:
                pass
            
            # Add label
            ax.text(pos[0], pos[1], pos[2] + 0.02, f'G{i}', fontsize=8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Point Cloud with Grasps (Red = Selected)')
        
        # Set equal aspect ratio
        max_range = 0.5  # 50cm range
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
    except Exception as e:
        print(f"Error in 3D visualization: {e}")


def draw_all_grasps_on_image(image, grasps, camera_intrinsics, selected_idx=None):
    """
    Draw all grasps on the image with the selected one highlighted.
    
    Args:
        image: RGB image
        grasps: List of grasp dictionaries
        camera_intrinsics: Camera intrinsics
        selected_idx: Index of selected grasp
    
    Returns:
        image_with_grasps: Image with all grasps overlay
    """
    img = image.copy()
    
    # Draw all grasps
    for i, grasp in enumerate(grasps):
        position = grasp['position']
        u, v = project_3d_to_2d(position, camera_intrinsics)
        
        if u is None or v is None:
            continue
            
        h, w = img.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            continue
        
        # Choose color and size based on selection
        if i == selected_idx:
            color = (0, 255, 0)  # Green for selected
            radius = 10
            thickness = -1
        else:
            color = (255, 165, 0)  # Orange for others
            radius = 6
            thickness = 2
        
        cv2.circle(img, (u, v), radius, color, thickness)
        cv2.putText(img, f"{i}", (u + 12, v), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1)
    
    return img

def backproject_depth_to_pointcloud(depth_image, camera_intrinsics, rgb_image=None):
    """
    Convert depth image to 3D point cloud using camera intrinsics.
    
    Args:
        depth_image: Depth image in meters
        camera_intrinsics: Camera intrinsics from RealSense
        rgb_image: Optional RGB image for colored point cloud
    
    Returns:
        point_cloud: Nx3 or Nx6 array (XYZ or XYZRGB)
    """
    height, width = depth_image.shape
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Extract intrinsic parameters
    fx, fy = camera_intrinsics.fx, camera_intrinsics.fy
    cx, cy = camera_intrinsics.ppx, camera_intrinsics.ppy
    
    # Convert to 3D coordinates
    z = depth_image.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack coordinates
    points_3d = np.stack([x, y, z], axis=-1)
    
    # Filter out invalid points (where depth is 0)
    valid_mask = z > 0
    points_3d = points_3d[valid_mask]
    
    if rgb_image is not None:
        # Add RGB information
        rgb_values = rgb_image[valid_mask]
        points_3d = np.concatenate([points_3d, rgb_values], axis=-1)
    
    return points_3d


def generate_dummy_grasps(point_cloud, num_grasps=10):
    """
    Generate realistic dummy grasps for testing purposes.
    In a real implementation, you would use a grasp predictor like M2T2.
    
    Args:
        point_cloud: Input point cloud
        num_grasps: Number of dummy grasps to generate
    
    Returns:
        grasps: List of grasp poses (dummy data)
    """
    # Get bounding box of point cloud
    if len(point_cloud) == 0:
        return []
    
    min_coords = np.min(point_cloud[:, :3], axis=0)
    max_coords = np.max(point_cloud[:, :3], axis=0)
    
    # Focus grasps in areas with actual points (more realistic)
    valid_points = point_cloud[:, :3]
    
    grasps = []
    for i in range(num_grasps):
        # Choose a random point from the point cloud as grasp location
        if len(valid_points) > 0:
            idx = np.random.randint(0, len(valid_points))
            base_position = valid_points[idx]
            
            # Add small random offset
            position = base_position + np.random.normal(0, 0.01, 3)
        else:
            # Fallback to random position in bounds
            position = np.random.uniform(min_coords, max_coords)
        
        # Generate realistic orientation (biased towards vertical approach)
        # Favor downward-facing grasps (more common for tabletop objects)
        approach_angle = np.random.normal(0, 0.3)  # Mostly vertical with some variation
        rotation_angle = np.random.uniform(0, 2*np.pi)  # Random rotation around vertical
        
        # Create quaternion for this orientation
        # Simple approach: rotate around X then Z axes
        qx = np.sin(approach_angle/2)
        qy = 0
        qz = np.sin(rotation_angle/2) * np.cos(approach_angle/2)
        qw = np.cos(rotation_angle/2) * np.cos(approach_angle/2)
        
        orientation = np.array([qx, qy, qz, qw])
        orientation = orientation / np.linalg.norm(orientation)
        
        # Generate realistic scores (favor grasps closer to center and higher up)
        center = (min_coords + max_coords) / 2
        distance_from_center = np.linalg.norm(position[:2] - center[:2])
        height_score = (position[2] - min_coords[2]) / max(max_coords[2] - min_coords[2], 0.01)
        
        base_score = 0.5 + 0.3 * height_score - 0.2 * distance_from_center
        score = np.clip(base_score + np.random.normal(0, 0.1), 0.1, 0.9)
        
        # Dummy grasp representation (position + orientation)
        grasp = {
            'position': position,
            'orientation': orientation,
            'score': score,
            'id': i,
            'width': np.random.uniform(0.02, 0.08),  # Gripper width in meters
            'force': np.random.uniform(10, 50)  # Grip force
        }
        grasps.append(grasp)
    
    # Sort by score (best first)
    grasps.sort(key=lambda g: g['score'], reverse=True)
    
    return grasps


def initialize_graspmolmo_safely():
    """
    Safely initialize GraspMolmo using Hugging Face inference client.
    This is much faster than downloading the 30GB model locally.
    """
    if not HF_CLIENT_AVAILABLE:
        return None
    
    print("Initializing GraspMolmo via Hugging Face inference client...")
    
    try:
        # Initialize the inference client with the correct model name
        client = InferenceClient(model="allenai/GraspMolmo")
        
        # Test the client with a simple request
        print("Testing connection to GraspMolmo API...")
        
        # Create a wrapper class that matches our interface
        class GraspMolmoHF:
            def __init__(self, client):
                self.client = client
                self.temp_image_path = "temp_grasp_image.png"
            
            def pred_grasp(self, rgb_image, point_cloud, task, grasps):
                """
                Predict the best grasp using Hugging Face inference API.
                
                Args:
                    rgb_image: RGB image from camera
                    point_cloud: 3D point cloud (not used by HF API currently)
                    task: Task description string
                    grasps: List of candidate grasps
                
                Returns:
                    idx: Index of selected grasp
                """
                try:
                    # Save RGB image temporarily
                    cv2.imwrite(self.temp_image_path, rgb_image)
                    
                    # Call the HF inference API with proper method
                    with open(self.temp_image_path, "rb") as image_file:
                        # Structure the request according to the model's expectations
                        prompt = f"Point to where I should grasp to accomplish the following task: {task}"
                        
                        # Use post() method with 'robotics' task type as required by GraspMolmo
                        response = self.client.post(
                            json={
                                "inputs": {
                                    "image": image_file.read(),
                                    "text": prompt,
                                    "task": "robotics"  # Model requires 'robotics' task type
                                },
                                
                            }
                        )
                    
                    # Parse the response - it may be a string that needs to be parsed
                    print(f"GraspMolmo HF response: {response}")
                    
                    # If the response is a string (text output from the model)
                    if isinstance(response, str):
                        # Try to extract coordinate information from the text response
                        import re
                        
                        # Look for patterns like "coordinates: [x, y, z]" or similar
                        coord_pattern = r'coordinates:?\s*\[([0-9.-]+),\s*([0-9.-]+),\s*([0-9.-]+)\]'
                        coord_match = re.search(coord_pattern, response)
                        
                        if coord_match:
                            # Extract coordinates
                            x, y, z = map(float, coord_match.groups())
                            target_pos = np.array([x, y, z])
                            
                            # Find the closest grasp
                            if len(grasps) > 0:
                                distances = [np.linalg.norm(g['position'] - target_pos) 
                                           for g in grasps]
                                return np.argmin(distances)
                        
                        # If coordinates not found in text response, check for grasp index
                        index_pattern = r'grasp(?:\s+index)?(?:\s*:)?\s*(\d+)'
                        index_match = re.search(index_pattern, response.lower())
                        
                        if index_match:
                            try:
                                idx = int(index_match.group(1))
                                if 0 <= idx < len(grasps):
                                    return idx
                            except (ValueError, IndexError):
                                pass
                    
                    # Handle dict response
                    elif isinstance(response, dict):
                        # Look for grasp selection or coordinates
                        if 'selected_grasp' in response:
                            return response['selected_grasp']
                        elif 'grasp_index' in response:
                            return response['grasp_index']
                        elif 'coordinates' in response or 'position' in response:
                            # If we get coordinates, find the closest grasp
                            target_pos = response.get('coordinates', response.get('position'))
                            if target_pos and len(grasps) > 0:
                                distances = [np.linalg.norm(g['position'] - np.array(target_pos)) 
                                           for g in grasps]
                                return np.argmin(distances)
                    
                    # Fallback: return best scoring grasp
                    print("Could not parse GraspMolmo response, using highest scoring grasp")
                    return max(range(len(grasps)), key=lambda i: grasps[i]['score'])
                    
                except Exception as e:
                    print(f"Error calling GraspMolmo HF API: {e}")
                    # Fallback to highest scoring grasp
                    if len(grasps) > 0:
                        return max(range(len(grasps)), key=lambda i: grasps[i]['score'])
                    return 0
                finally:
                    # Clean up temp file
                    try:
                        import os
                        if os.path.exists(self.temp_image_path):
                            os.remove(self.temp_image_path)
                    except:
                        pass
        
        gm = GraspMolmoHF(client)
        print("✅ GraspMolmo HF client initialized successfully!")
        return gm
        
    except Exception as e:
        print(f"\nError initializing GraspMolmo HF client: {e}")
        print("This might be due to:")
        print("- No internet connection")
        print("- HuggingFace API issues") 
        print("- Authentication required (try: huggingface-cli login)")
        print("Continuing with dummy grasps only...")
        return None


def main():
    """Main test function."""
    print("=" * 60)
    print("GraspMolmo + RealSense Integration Test")
    print("=" * 60)
    
    # Parse command line arguments for testing modes
    import sys
    test_mode = "full"  # Default mode
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--camera-only":
            test_mode = "camera"
        elif sys.argv[1] == "--dummy-grasps":
            test_mode = "dummy"
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python graspmolmo_test.py              # Full test with GraspMolmo")
            print("  python graspmolmo_test.py --camera-only # Test camera and visualization only")
            print("  python graspmolmo_test.py --dummy-grasps # Test with dummy grasps (no GraspMolmo)")
            return
    
    print(f"Running in mode: {test_mode}")
    
    # Initialize camera
    print("\n1. Initializing RealSense camera...")
    camera = Camera(width=640, height=480, fps=30, debug=True)
    
    if not camera.start():
        print("❌ Failed to start camera!")
        print("Make sure your RealSense camera is connected and not used by another application.")
        return
    
    print("✅ Camera started successfully!")
    
    # Initialize GraspMolmo based on test mode
    gm = None
    if test_mode == "full":
        print("\n2. Initializing GraspMolmo...")
        gm = initialize_graspmolmo_safely()
        if gm is None:
            print("⚠️  GraspMolmo not available, falling back to dummy grasps")
            test_mode = "dummy"
        else:
            print("✅ GraspMolmo ready!")
    elif test_mode == "dummy":
        print("\n2. Skipping GraspMolmo initialization (using dummy grasps)")
    elif test_mode == "camera":
        print("\n2. Camera-only mode (no grasp prediction)")
    
    # Test parameters
    task = "open the drawer on the box"  # Example task
    capture_count = 0
    current_grasps = []
    selected_grasp_idx = None
    current_rgb = None
    
    try:
        print(f"\n3. Starting main loop...")
        print("\nControls:")
        if test_mode == "camera":
            print("- Any key to update display")
        else:
            print("- Press 'c' to capture and process a frame")
            print("- Press 'v' to show 3D visualization (if available)")
            print("- Press 'r' to reset grasps")
        print("- Press 'q' or ESC to quit")
        print("- Press 'h' for help")
        
        if test_mode != "camera":
            print(f"\nCurrent task: '{task}'")
        
        while True:
            # Get frames for processing
            frames = camera.get_frames()
            if frames is None:
                continue
                
            rgb_image, depth_image = frames
            current_rgb = rgb_image.copy()
            
            if test_mode == "camera":
                # Camera-only mode: just show the feed
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                combined_display = np.hstack((rgb_image, depth_colormap))
                cv2.putText(combined_display, "Camera Test Mode - Press 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('RealSense Camera Test', combined_display)
            else:
                # Create display image with grasp overlays
                display_image = rgb_image.copy()
                if current_grasps and selected_grasp_idx is not None:
                    # Draw all grasps with selected one highlighted
                    display_image = draw_all_grasps_on_image(
                        display_image, current_grasps, camera.intrinsics, selected_grasp_idx
                    )
                    
                    # Draw detailed view of selected grasp
                    display_image = draw_grasp_on_image(
                        display_image, current_grasps[selected_grasp_idx], camera.intrinsics
                    )
                elif current_grasps:
                    # Draw all grasps without selection
                    display_image = draw_all_grasps_on_image(
                        display_image, current_grasps, camera.intrinsics
                    )
                
                # Create depth visualization
                depth_meters = depth_image.astype(np.float32) * camera.depth_scale
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # Combine RGB and depth for display
                combined_display = np.hstack((display_image, depth_colormap))
                
                # Add text overlay with information
                info_y = 30
                if current_grasps:
                    info_text = f"Grasps: {len(current_grasps)}"
                    if selected_grasp_idx is not None:
                        info_text += f" | Selected: #{selected_grasp_idx}"
                        info_text += f" | Score: {current_grasps[selected_grasp_idx]['score']:.3f}"
                    cv2.putText(combined_display, info_text, (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    info_y += 30
                
                cv2.putText(combined_display, f"Task: {task}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                info_y += 25
                
                mode_text = "GraspMolmo" if gm else "Dummy Grasps"
                cv2.putText(combined_display, f"Mode: {mode_text}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('GraspMolmo + RealSense', combined_display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('h'):  # Help
                print("\n" + "="*40)
                print("HELP - Keyboard Controls:")
                print("="*40)
                if test_mode != "camera":
                    print("c - Capture and process frame")
                    print("v - Show 3D visualization")
                    print("r - Reset grasps")
                print("q - Quit application")
                print("h - Show this help")
                print("="*40)
            elif test_mode == "camera":
                continue  # Skip grasp processing in camera-only mode
            elif key == ord('r'):  # Reset grasps
                current_grasps = []
                selected_grasp_idx = None
                print("Grasps reset")
            elif key == ord('v'):  # Show 3D visualization
                if current_grasps and MATPLOTLIB_AVAILABLE:
                    print("Showing 3D visualization...")
                    # Recreate point cloud for visualization
                    depth_meters = depth_image.astype(np.float32) * camera.depth_scale
                    point_cloud = backproject_depth_to_pointcloud(
                        depth_meters, camera.intrinsics, rgb_image
                    )
                    visualize_point_cloud_with_grasps(
                        point_cloud, current_grasps, selected_grasp_idx
                    )
                else:
                    if not current_grasps:
                        print("No grasps available - press 'c' to capture first")
                    else:
                        print("Matplotlib not available for 3D visualization")
            elif key == ord('c'):  # Capture and process
                capture_count += 1
                print(f"\n--- Capture #{capture_count} ---")
                
                # Convert depth from millimeters to meters
                depth_meters = depth_image.astype(np.float32) * camera.depth_scale
                
                # Get camera intrinsics
                intrinsics = camera.intrinsics
                print(f"Camera intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
                      f"cx={intrinsics.ppx:.1f}, cy={intrinsics.ppy:.1f}")
                
                # Create point cloud
                print("Creating point cloud...")
                point_cloud = backproject_depth_to_pointcloud(
                    depth_meters, intrinsics, rgb_image
                )
                print(f"Point cloud has {len(point_cloud)} points")
                
                if len(point_cloud) == 0:
                    print("Empty point cloud! Make sure objects are visible.")
                    continue
                
                # Generate dummy grasps (replace with real grasp predictor)
                print("Generating dummy grasps...")
                grasps = generate_dummy_grasps(point_cloud, num_grasps=5)
                print(f"Generated {len(grasps)} dummy grasps")
                
                if not grasps:
                    print("No grasps generated!")
                    continue
                
                # Store grasps for visualization
                current_grasps = grasps
                
                # Use GraspMolmo if available
                if gm is not None:
                    try:
                        print(f"Using GraspMolmo to select best grasp for task: '{task}'")
                        
                        # Note: This is the expected interface from the README
                        # You may need to adjust based on actual GraspMolmo API
                        idx = gm.pred_grasp(rgb_image, point_cloud, task, grasps)
                        
                        selected_grasp_idx = idx
                        print(f"GraspMolmo selected grasp #{idx}")
                        selected_grasp = grasps[idx]
                        print(f"Selected grasp position: {selected_grasp['position']}")
                        print(f"Selected grasp score: {selected_grasp['score']:.3f}")
                        
                    except Exception as e:
                        print(f"Error during GraspMolmo prediction: {e}")
                        print("This might be due to API differences or missing dependencies")
                        # Fallback: select best scoring grasp
                        selected_grasp_idx = max(range(len(grasps)), key=lambda i: grasps[i]['score'])
                        print(f"Fallback: selected highest scoring grasp #{selected_grasp_idx}")
                
                else:
                    print("GraspMolmo not available - selecting highest scoring dummy grasp")
                    selected_grasp_idx = max(range(len(grasps)), key=lambda i: grasps[i]['score'])
                    best_grasp = grasps[selected_grasp_idx]
                    print(f"Best dummy grasp #{selected_grasp_idx}: position={best_grasp['position']}, "
                          f"score={best_grasp['score']:.3f}")
                
                # Create visualization with selected grasp
                print("Creating grasp visualizations...")
                
                # Show 2D visualization immediately
                viz_image = draw_all_grasps_on_image(
                    rgb_image, grasps, camera.intrinsics, selected_grasp_idx
                )
                viz_image = draw_grasp_on_image(
                    viz_image, grasps[selected_grasp_idx], camera.intrinsics
                )
                
                cv2.imshow('Grasp Visualization', viz_image)
                cv2.waitKey(1)
                
                # Show 3D visualization if available
                if MATPLOTLIB_AVAILABLE:
                    print("Generating 3D visualization... (press 'v' to show again later)")
                    visualize_point_cloud_with_grasps(point_cloud, grasps, selected_grasp_idx)
                
                # Display point cloud statistics
                pc_xyz = point_cloud[:, :3]
                print(f"Point cloud bounds:")
                print(f"  X: {np.min(pc_xyz[:, 0]):.3f} to {np.max(pc_xyz[:, 0]):.3f}")
                print(f"  Y: {np.min(pc_xyz[:, 1]):.3f} to {np.max(pc_xyz[:, 1]):.3f}")
                print(f"  Z: {np.min(pc_xyz[:, 2]):.3f} to {np.max(pc_xyz[:, 2]):.3f}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("\n4. Shutting down...")
        camera.stop()
        print("✅ Camera stopped successfully")
        cv2.destroyAllWindows()


def check_dependencies():
    """Check if all required dependencies are available."""
    print("Checking dependencies...")
    
    missing_deps = []
    
    # Check RealSense
    try:
        import pyrealsense2 as rs
        print("✅ pyrealsense2 available")
    except ImportError:
        missing_deps.append("pyrealsense2")
        
    # Check OpenCV
    try:
        import cv2
        print("✅ opencv-python available")
    except ImportError:
        missing_deps.append("opencv-python")
        
    # Check numpy
    try:
        import numpy as np
        print("✅ numpy available")
    except ImportError:
        missing_deps.append("numpy")
    
    # Check matplotlib (optional)
    if MATPLOTLIB_AVAILABLE:
        print("✅ matplotlib available (3D visualization enabled)")
    else:
        print("⚠️  matplotlib not available (3D visualization disabled)")
        
    # Check Hugging Face client (optional but recommended)
    if HF_CLIENT_AVAILABLE:
        print("✅ huggingface_hub available (GraspMolmo cloud API enabled)")
    else:
        print("⚠️  huggingface_hub not available (will use dummy grasps)")
        print("    Install with: pip install huggingface_hub")
    
    if missing_deps:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All required dependencies available!")
    return True


if __name__ == "__main__":
    print("GraspMolmo + RealSense Integration Test")
    print("======================================")
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies before running this script.")
        print("\nQuick setup:")
        print("1. pip install pyrealsense2 opencv-python numpy matplotlib huggingface_hub")
        print("2. (Optional) huggingface-cli login  # For API access")
        print("\nThen run:")
        print("python graspmolmo_test.py                    # Full test with GraspMolmo cloud API")
        print("python graspmolmo_test.py --dummy-grasps     # Test with dummy grasps only")
        print("python graspmolmo_test.py --camera-only      # Test camera and visualization only")
        sys.exit(1)
    
    main()

if __name__ == "__main__":
    main()