#!/usr/bin/env python3
"""
Text-based visualization of collision objects generated from RealSense point cloud data
This provides a simple ASCII representation of the collision objects
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger
from cognitive_bt_framework.src.vision.realsense import Camera


def create_ascii_workspace(collision_objects, workspace_bounds, resolution=20):
    """
    Create ASCII representation of workspace with collision objects
    
    Args:
        collision_objects: List of collision objects
        workspace_bounds: Dictionary with 'x', 'y', 'z' bounds
        resolution: Number of characters for workspace representation
        
    Returns:
        ASCII representation as string
    """
    # Create grid
    x_range = workspace_bounds['x']
    y_range = workspace_bounds['y']
    
    # Create coordinate mapping
    x_coords = np.linspace(x_range[0], x_range[1], resolution)
    y_coords = np.linspace(y_range[0], y_range[1], resolution)
    
    # Initialize grid
    grid = [[' ' for _ in range(resolution)] for _ in range(resolution)]
    
    # Mark robot base
    robot_x = 0
    robot_y = 0
    robot_x_idx = int((robot_x - x_range[0]) / (x_range[1] - x_range[0]) * (resolution - 1))
    robot_y_idx = int((robot_y - y_range[0]) / (y_range[1] - y_range[0]) * (resolution - 1))
    if 0 <= robot_x_idx < resolution and 0 <= robot_y_idx < resolution:
        grid[robot_y_idx][robot_x_idx] = 'R'
    
    # Mark collision objects
    for i, obj in enumerate(collision_objects):
        if obj['type'] == 'Cuboid':
            # Calculate object bounds
            x_min = obj['position'][0] - obj['dimensions'][0]/2
            x_max = obj['position'][0] + obj['dimensions'][0]/2
            y_min = obj['position'][1] - obj['dimensions'][1]/2
            y_max = obj['position'][1] + obj['dimensions'][1]/2
            
            # Convert to grid coordinates
            x_min_idx = int((x_min - x_range[0]) / (x_range[1] - x_range[0]) * (resolution - 1))
            x_max_idx = int((x_max - x_range[0]) / (x_range[1] - x_range[0]) * (resolution - 1))
            y_min_idx = int((y_min - y_range[0]) / (y_range[1] - y_range[0]) * (resolution - 1))
            y_max_idx = int((y_max - y_range[0]) / (y_range[1] - y_range[0]) * (resolution - 1))
            
            # Clamp to grid bounds
            x_min_idx = max(0, min(resolution-1, x_min_idx))
            x_max_idx = max(0, min(resolution-1, x_max_idx))
            y_min_idx = max(0, min(resolution-1, y_min_idx))
            y_max_idx = max(0, min(resolution-1, y_max_idx))
            
            # Mark object in grid
            for y in range(y_min_idx, y_max_idx + 1):
                for x in range(x_min_idx, x_max_idx + 1):
                    if grid[y][x] == ' ':
                        grid[y][x] = str(i)
                    elif grid[y][x] == 'R':
                        grid[y][x] = 'R'  # Robot takes precedence
    
    # Create ASCII representation
    ascii_art = []
    ascii_art.append("=" * (resolution + 2))
    ascii_art.append("COLLISION OBJECTS WORKSPACE (TOP-DOWN VIEW)")
    ascii_art.append("=" * (resolution + 2))
    ascii_art.append(f"X: {x_range[0]:.2f}m to {x_range[1]:.2f}m")
    ascii_art.append(f"Y: {y_range[0]:.2f}m to {y_range[1]:.2f}m")
    ascii_art.append("")
    
    # Add Y-axis labels
    for i in range(resolution):
        y_val = y_coords[resolution-1-i]  # Flip Y axis for display
        ascii_art.append(f"{y_val:6.2f} |{''.join(grid[resolution-1-i])}|")
    
    # Add X-axis labels
    x_labels = "      "
    for i in range(0, resolution, 2):
        x_labels += f"{x_coords[i]:6.2f}"
    ascii_art.append("       " + "=" * resolution)
    ascii_art.append(x_labels)
    ascii_art.append("")
    
    # Add legend
    ascii_art.append("LEGEND:")
    ascii_art.append("R = Robot Base")
    for i, obj in enumerate(collision_objects):
        if obj['type'] == 'Cuboid':
            ascii_art.append(f"{i} = {obj['name']} ({obj['dimensions'][0]:.2f}m × {obj['dimensions'][1]:.2f}m)")
    
    return "\n".join(ascii_art)


def visualize_collision_objects_text():
    """
    Text-based visualization of collision objects from RealSense
    """
    print("="*60)
    print("TEXT-BASED COLLISION OBJECTS VISUALIZATION")
    print("="*60)
    
    # Initialize motion planner
    print("1. Initializing motion planner...")
    motion_planner = CuRoboMotionPlanner(robot_ip="192.168.1.224")
    motion_planner.init_curobo()
    print("✅ Motion planner initialized")
    
    # Initialize RealSense camera
    print("\n2. Initializing RealSense camera...")
    try:
        camera = Camera(
            width=640,
            height=480,
            fps=30,
            use_viewer_defaults=True,
            debug=False
        )
        
        if not camera.start():
            print("❌ Failed to start RealSense camera")
            return False
        print("✅ RealSense camera started")
        
    except Exception as e:
        print(f"❌ Failed to initialize RealSense camera: {e}")
        return False
    
    # Create collision debugger
    print("\n3. Creating collision debugger...")
    debugger = CollisionDebugger(motion_planner)
    
    # Capture point cloud and generate collision objects
    print("\n4. Capturing point cloud and generating collision objects...")
    pcd = camera.get_point_cloud(use_averaging=False, manual_calculation=True)
    if pcd is not None and len(pcd) > 0:
        print(f"✅ Captured point cloud with {len(pcd)} points")
        
        # Update collision objects
        motion_planner.update_dynamic_collision_objects(pcd)
        
        # Get collision objects
        collision_objects = debugger.list_collision_objects()
        print(f"✅ Generated {len(collision_objects)} collision objects")
        
        # Create workspace bounds
        workspace_bounds = {
            'x': [-0.5, 1.0],
            'y': [-0.8, 0.8],
            'z': [0.0, 1.2]
        }
        
        # Create ASCII visualization
        print("\n5. Creating ASCII visualization...")
        ascii_workspace = create_ascii_workspace(collision_objects, workspace_bounds)
        print(ascii_workspace)
        
        # Print detailed information
        print("\n" + "="*50)
        print("DETAILED COLLISION OBJECTS INFORMATION")
        print("="*50)
        
        total_volume = 0
        for i, obj in enumerate(collision_objects):
            if obj['type'] == 'Cuboid':
                volume = obj['dimensions'][0] * obj['dimensions'][1] * obj['dimensions'][2]
                total_volume += volume
                
                print(f"\n{obj['name']}:")
                print(f"  Position: ({obj['position'][0]:.3f}, {obj['position'][1]:.3f}, {obj['position'][2]:.3f})")
                print(f"  Dimensions: {obj['dimensions'][0]:.3f}m × {obj['dimensions'][1]:.3f}m × {obj['dimensions'][2]:.3f}m")
                print(f"  Volume: {volume:.3f} m³")
                print(f"  Orientation: ({obj['orientation'][0]:.3f}, {obj['orientation'][1]:.3f}, {obj['orientation'][2]:.3f}, {obj['orientation'][3]:.3f})")
        
        print(f"\nTotal volume of collision objects: {total_volume:.3f} m³")
        
        # Print point cloud statistics
        print("\n" + "="*50)
        print("POINT CLOUD STATISTICS")
        print("="*50)
        print(f"Total points: {len(pcd)}")
        print(f"X range: {pcd[:, 0].min():.3f} to {pcd[:, 0].max():.3f} m")
        print(f"Y range: {pcd[:, 1].min():.3f} to {pcd[:, 1].max():.3f} m")
        print(f"Z range: {pcd[:, 2].min():.3f} to {pcd[:, 2].max():.3f} m")
        print(f"Mean position: ({pcd[:, 0].mean():.3f}, {pcd[:, 1].mean():.3f}, {pcd[:, 2].mean():.3f})")
        
        # Test collision detection
        print("\n" + "="*50)
        print("COLLISION DETECTION TEST")
        print("="*50)
        
        test_positions = [
            [0.0, 0.0, 0.3],   # Above table
            [0.2, 0.0, 0.3],   # In front of robot
            [0.4, 0.0, 0.3],   # Further in front
            [0.0, 0.2, 0.3],   # To the right
            [0.0, -0.2, 0.3],  # To the left
        ]
        
        collision_results = debugger.test_collision_detection(test_positions)
        
        print("\nCollision detection results:")
        for i, pos in enumerate(test_positions):
            status = "COLLISION" if collision_results[i] else "FREE"
            print(f"  Position {i}: {pos} -> {status}")
        
        print(f"\nSummary: {sum(collision_results)}/{len(test_positions)} positions have collisions")
        
    else:
        print("❌ Failed to capture point cloud")
    
    # Cleanup
    camera.stop()
    motion_planner.disconnect_robot()
    return True


if __name__ == "__main__":
    print("Text-Based Collision Objects Visualization")
    print("="*60)
    
    visualize_collision_objects_text()
    
    print("\n✅ Text-based visualization completed!") 