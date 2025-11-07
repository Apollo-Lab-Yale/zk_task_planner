#!/usr/bin/env python3
"""
Visualize collision objects generated from RealSense point cloud data
This script shows the collision objects in 3D space with the robot workspace
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger
from cognitive_bt_framework.src.vision.realsense import Camera


def create_cuboid_vertices(position, dimensions, orientation):
    """
    Create vertices for a cuboid given position, dimensions, and orientation
    
    Args:
        position: [x, y, z] center position
        dimensions: [width, height, depth] dimensions
        orientation: [qx, qy, qz, qw] quaternion orientation
        
    Returns:
        vertices: 8x3 array of vertex positions
    """
    # Create unit cube vertices
    w, h, d = dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2
    vertices = np.array([
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
        [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]
    ])
    
    # Convert quaternion to rotation matrix
    qx, qy, qz, qw = orientation
    rotation_matrix = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*qx*qx - 2*qz*qz, 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*qx*qx - 2*qy*qy]
    ])
    
    # Apply rotation and translation
    vertices = vertices @ rotation_matrix.T + position
    
    return vertices


def create_cuboid_faces(vertices):
    """
    Create faces for a cuboid from vertices
    
    Args:
        vertices: 8x3 array of vertex positions
        
    Returns:
        faces: list of face vertices
    """
    # Define faces by vertex indices
    face_indices = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5]   # right
    ]
    
    faces = [vertices[indices] for indices in face_indices]
    return faces


def visualize_collision_objects_realtime():
    """
    Real-time visualization of collision objects from RealSense
    """
    print("="*60)
    print("REALTIME COLLISION OBJECTS VISUALIZATION")
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
            debug=False  # Reduce debug output
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
    
    # Setup visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('RealSense Collision Objects Visualization')
    
    # Define robot workspace bounds
    workspace_bounds = {
        'x': [-0.5, 1.0],
        'y': [-0.8, 0.8],
        'z': [0.0, 1.2]
    }
    
    # Set axis limits
    ax.set_xlim(workspace_bounds['x'])
    ax.set_ylim(workspace_bounds['y'])
    ax.set_zlim(workspace_bounds['z'])
    
    # Add robot base position
    robot_base = np.array([0, 0, 0])
    ax.scatter([robot_base[0]], [robot_base[1]], [robot_base[2]], 
               c='red', s=100, marker='o', label='Robot Base')
    
    # Add workspace grid
    x_grid = np.linspace(workspace_bounds['x'][0], workspace_bounds['x'][1], 5)
    y_grid = np.linspace(workspace_bounds['y'][0], workspace_bounds['y'][1], 5)
    z_grid = np.linspace(workspace_bounds['z'][0], workspace_bounds['z'][1], 3)
    
    # Draw grid lines
    for x in x_grid:
        ax.plot([x, x], [workspace_bounds['y'][0], workspace_bounds['y'][1]], [0, 0], 'k:', alpha=0.3)
    for y in y_grid:
        ax.plot([workspace_bounds['x'][0], workspace_bounds['x'][1]], [y, y], [0, 0], 'k:', alpha=0.3)
    
    # Add table surface
    table_x = np.array([workspace_bounds['x'][0], workspace_bounds['x'][1]])
    table_y = np.array([workspace_bounds['y'][0], workspace_bounds['y'][1]])
    table_z = np.zeros((2, 2))
    ax.plot_surface(table_x[:, None], table_y[None, :], table_z, alpha=0.2, color='gray')
    
    print("\n4. Starting real-time visualization...")
    print("Press 'q' to quit, 'r' to refresh collision objects")
    
    try:
        while True:
            # Capture point cloud and update collision objects
            pcd = camera.get_point_cloud(use_averaging=False, manual_calculation=True)
            if pcd is not None and len(pcd) > 0:
                print(f"\rCaptured {len(pcd)} points, updating collision objects...", end='')
                
                # Update collision objects
                motion_planner.update_dynamic_collision_objects(pcd)
                
                # Get collision objects
                collision_objects = debugger.list_collision_objects()
                
                # Clear previous objects
                ax.clear()
                
                # Reset plot
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'RealSense Collision Objects ({len(collision_objects)} objects)')
                ax.set_xlim(workspace_bounds['x'])
                ax.set_ylim(workspace_bounds['y'])
                ax.set_zlim(workspace_bounds['z'])
                
                # Add robot base
                ax.scatter([robot_base[0]], [robot_base[1]], [robot_base[2]], 
                           c='red', s=100, marker='o', label='Robot Base')
                
                # Add grid
                for x in x_grid:
                    ax.plot([x, x], [workspace_bounds['y'][0], workspace_bounds['y'][1]], [0, 0], 'k:', alpha=0.3)
                for y in y_grid:
                    ax.plot([workspace_bounds['x'][0], workspace_bounds['x'][1]], [y, y], [0, 0], 'k:', alpha=0.3)
                
                # Add table surface
                ax.plot_surface(table_x[:, None], table_y[None, :], table_z, alpha=0.2, color='gray')
                
                # Visualize collision objects
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                for i, obj in enumerate(collision_objects):
                    if obj['type'] == 'Cuboid':
                        # Create cuboid visualization
                        vertices = create_cuboid_vertices(
                            obj['position'], 
                            obj['dimensions'], 
                            obj['orientation']
                        )
                        faces = create_cuboid_faces(vertices)
                        
                        # Create 3D collection
                        poly3d = Poly3DCollection(faces, alpha=0.6, 
                                                 facecolor=colors[i % len(colors)])
                        ax.add_collection3d(poly3d)
                        
                        # Add center point
                        ax.scatter([obj['position'][0]], [obj['position'][1]], [obj['position'][2]], 
                                   c=colors[i % len(colors)], s=50, marker='x')
                        
                        # Add text label
                        ax.text(obj['position'][0], obj['position'][1], obj['position'][2], 
                               obj['name'], fontsize=8)
                
                # Add legend
                ax.legend()
                
                plt.draw()
                plt.pause(0.1)
                
            else:
                print("\rNo point cloud data available...", end='')
                plt.pause(0.5)
                
    except KeyboardInterrupt:
        print("\n\nStopping visualization...")
    except Exception as e:
        print(f"\nError during visualization: {e}")
    finally:
        # Cleanup
        camera.stop()
        motion_planner.disconnect_robot()
        plt.close()
        print("✅ Visualization stopped")


def visualize_collision_objects_static():
    """
    Static visualization of collision objects from a single capture
    """
    print("="*60)
    print("STATIC COLLISION OBJECTS VISUALIZATION")
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
        
        # Create visualization
        print("\n5. Creating visualization...")
        fig = plt.figure(figsize=(15, 10))
        
        # Main 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Collision Objects in 3D Space')
        
        # Set workspace bounds
        workspace_bounds = {
            'x': [-0.5, 1.0],
            'y': [-0.8, 0.8],
            'z': [0.0, 1.2]
        }
        ax1.set_xlim(workspace_bounds['x'])
        ax1.set_ylim(workspace_bounds['y'])
        ax1.set_zlim(workspace_bounds['z'])
        
        # Add robot base
        robot_base = np.array([0, 0, 0])
        ax1.scatter([robot_base[0]], [robot_base[1]], [robot_base[2]], 
                   c='red', s=100, marker='o', label='Robot Base')
        
        # Add workspace grid
        x_grid = np.linspace(workspace_bounds['x'][0], workspace_bounds['x'][1], 5)
        y_grid = np.linspace(workspace_bounds['y'][0], workspace_bounds['y'][1], 5)
        for x in x_grid:
            ax1.plot([x, x], [workspace_bounds['y'][0], workspace_bounds['y'][1]], [0, 0], 'k:', alpha=0.3)
        for y in y_grid:
            ax1.plot([workspace_bounds['x'][0], workspace_bounds['x'][1]], [y, y], [0, 0], 'k:', alpha=0.3)
        
        # Add table surface
        table_x = np.array([workspace_bounds['x'][0], workspace_bounds['x'][1]])
        table_y = np.array([workspace_bounds['y'][0], workspace_bounds['y'][1]])
        table_z = np.zeros((2, 2))
        ax1.plot_surface(table_x[:, None], table_y[None, :], table_z, alpha=0.2, color='gray')
        
        # Visualize collision objects
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, obj in enumerate(collision_objects):
            if obj['type'] == 'Cuboid':
                # Create cuboid visualization
                vertices = create_cuboid_vertices(
                    obj['position'], 
                    obj['dimensions'], 
                    obj['orientation']
                )
                faces = create_cuboid_faces(vertices)
                
                # Create 3D collection
                poly3d = Poly3DCollection(faces, alpha=0.6, 
                                         facecolor=colors[i % len(colors)])
                ax1.add_collection3d(poly3d)
                
                # Add center point
                ax1.scatter([obj['position'][0]], [obj['position'][1]], [obj['position'][2]], 
                           c=colors[i % len(colors)], s=50, marker='x')
                
                # Add text label
                ax1.text(obj['position'][0], obj['position'][1], obj['position'][2], 
                       obj['name'], fontsize=8)
        
        # Top-down view
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-Down View (X-Y Plane)')
        ax2.set_xlim(workspace_bounds['x'])
        ax2.set_ylim(workspace_bounds['y'])
        ax2.grid(True, alpha=0.3)
        
        # Add robot base
        ax2.scatter([robot_base[0]], [robot_base[1]], c='red', s=100, marker='o', label='Robot Base')
        
        # Add collision objects as rectangles
        for i, obj in enumerate(collision_objects):
            if obj['type'] == 'Cuboid':
                # Create rectangle for top-down view
                rect = patches.Rectangle(
                    (obj['position'][0] - obj['dimensions'][0]/2, 
                     obj['position'][1] - obj['dimensions'][1]/2),
                    obj['dimensions'][0], obj['dimensions'][1],
                    alpha=0.6, facecolor=colors[i % len(colors)],
                    edgecolor='black', linewidth=1
                )
                ax2.add_patch(rect)
                
                # Add center point
                ax2.scatter([obj['position'][0]], [obj['position'][1]], 
                           c=colors[i % len(colors)], s=50, marker='x')
                
                # Add text label
                ax2.text(obj['position'][0], obj['position'][1], obj['name'], 
                       fontsize=8, ha='center', va='bottom')
        
        ax2.legend()
        
        # Add statistics
        fig.suptitle(f'RealSense Collision Objects Visualization\n{len(collision_objects)} objects detected from {len(pcd)} points', 
                    fontsize=14)
        
        # Add object details
        details_text = "Object Details:\n"
        for i, obj in enumerate(collision_objects):
            details_text += f"{obj['name']}: {obj['dimensions'][0]:.2f}m × {obj['dimensions'][1]:.2f}m × {obj['dimensions'][2]:.2f}m\n"
        
        fig.text(0.02, 0.02, details_text, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualization completed")
        
    else:
        print("❌ Failed to capture point cloud")
    
    # Cleanup
    camera.stop()
    motion_planner.disconnect_robot()
    return True


if __name__ == "__main__":
    print("Collision Objects Visualization")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description="Visualize collision objects from RealSense")
    parser.add_argument("--mode", choices=["static", "realtime"], default="static",
                       help="Visualization mode: static (single capture) or realtime (continuous)")
    
    args = parser.parse_args()
    
    if args.mode == "realtime":
        visualize_collision_objects_realtime()
    else:
        visualize_collision_objects_static()
    
    print("\n✅ Visualization script completed!") 