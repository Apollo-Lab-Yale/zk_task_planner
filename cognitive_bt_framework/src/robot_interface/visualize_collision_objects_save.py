#!/usr/bin/env python3
"""
Visualize collision objects and save plots to files
This avoids Qt backend issues by saving plots instead of displaying them
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger
from cognitive_bt_framework.src.vision.realsense import Camera


def visualize_collision_objects_save():
    """
    Visualize collision objects and save plots to files
    """
    print("="*60)
    print("COLLISION OBJECTS VISUALIZATION (SAVE TO FILES)")
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
        
        # Create output directory
        output_dir = "collision_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        print("\n5. Creating and saving visualizations...")
        
        # Set workspace bounds
        workspace_bounds = {
            'x': [-0.5, 1.0],
            'y': [-0.8, 0.8],
            'z': [0.0, 1.2]
        }
        
        # Colors for objects
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # 1. Top-down view (X-Y plane)
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Collision Objects - Top-Down View (X-Y Plane)')
        ax1.set_xlim(workspace_bounds['x'])
        ax1.set_ylim(workspace_bounds['y'])
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Add robot base
        robot_base = np.array([0, 0])
        ax1.scatter([robot_base[0]], [robot_base[1]], c='red', s=100, marker='o', label='Robot Base')
        
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
                ax1.add_patch(rect)
                
                # Add center point
                ax1.scatter([obj['position'][0]], [obj['position'][1]], 
                           c=colors[i % len(colors)], s=50, marker='x')
                
                # Add text label
                ax1.text(obj['position'][0], obj['position'][1], obj['name'], 
                       fontsize=8, ha='center', va='bottom')
        
        ax1.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/collision_objects_top_down.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved top-down view to {output_dir}/collision_objects_top_down.png")
        plt.close()
        
        # 2. Side view (X-Z plane)
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Collision Objects - Side View (X-Z Plane)')
        ax2.set_xlim(workspace_bounds['x'])
        ax2.set_ylim(workspace_bounds['z'])
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Add robot base
        ax2.scatter([robot_base[0]], [0], c='red', s=100, marker='o', label='Robot Base')
        
        # Add collision objects as rectangles
        for i, obj in enumerate(collision_objects):
            if obj['type'] == 'Cuboid':
                # Create rectangle for side view
                rect = patches.Rectangle(
                    (obj['position'][0] - obj['dimensions'][0]/2, 
                     obj['position'][2] - obj['dimensions'][2]/2),
                    obj['dimensions'][0], obj['dimensions'][2],
                    alpha=0.6, facecolor=colors[i % len(colors)],
                    edgecolor='black', linewidth=1
                )
                ax2.add_patch(rect)
                
                # Add center point
                ax2.scatter([obj['position'][0]], [obj['position'][2]], 
                           c=colors[i % len(colors)], s=50, marker='x')
                
                # Add text label
                ax2.text(obj['position'][0], obj['position'][2], obj['name'], 
                       fontsize=8, ha='center', va='bottom')
        
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/collision_objects_side_view.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved side view to {output_dir}/collision_objects_side_view.png")
        plt.close()
        
        # 3. Comprehensive view with all projections
        fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top-down view
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Top-Down View (X-Y Plane)')
        ax1.set_xlim(workspace_bounds['x'])
        ax1.set_ylim(workspace_bounds['y'])
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Add robot base
        ax1.scatter([robot_base[0]], [robot_base[1]], c='red', s=100, marker='o', label='Robot Base')
        
        # Add collision objects
        for i, obj in enumerate(collision_objects):
            if obj['type'] == 'Cuboid':
                rect = patches.Rectangle(
                    (obj['position'][0] - obj['dimensions'][0]/2, 
                     obj['position'][1] - obj['dimensions'][1]/2),
                    obj['dimensions'][0], obj['dimensions'][1],
                    alpha=0.6, facecolor=colors[i % len(colors)],
                    edgecolor='black', linewidth=1
                )
                ax1.add_patch(rect)
                ax1.scatter([obj['position'][0]], [obj['position'][1]], 
                           c=colors[i % len(colors)], s=50, marker='x')
                ax1.text(obj['position'][0], obj['position'][1], obj['name'], 
                       fontsize=8, ha='center', va='bottom')
        
        ax1.legend()
        
        # Side view
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Side View (X-Z Plane)')
        ax2.set_xlim(workspace_bounds['x'])
        ax2.set_ylim(workspace_bounds['z'])
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        ax2.scatter([robot_base[0]], [0], c='red', s=100, marker='o', label='Robot Base')
        
        for i, obj in enumerate(collision_objects):
            if obj['type'] == 'Cuboid':
                rect = patches.Rectangle(
                    (obj['position'][0] - obj['dimensions'][0]/2, 
                     obj['position'][2] - obj['dimensions'][2]/2),
                    obj['dimensions'][0], obj['dimensions'][2],
                    alpha=0.6, facecolor=colors[i % len(colors)],
                    edgecolor='black', linewidth=1
                )
                ax2.add_patch(rect)
                ax2.scatter([obj['position'][0]], [obj['position'][2]], 
                           c=colors[i % len(colors)], s=50, marker='x')
                ax2.text(obj['position'][0], obj['position'][2], obj['name'], 
                       fontsize=8, ha='center', va='bottom')
        
        ax2.legend()
        
        # Front view (Y-Z plane)
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Front View (Y-Z Plane)')
        ax3.set_xlim(workspace_bounds['y'])
        ax3.set_ylim(workspace_bounds['z'])
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        ax3.scatter([robot_base[1]], [0], c='red', s=100, marker='o', label='Robot Base')
        
        for i, obj in enumerate(collision_objects):
            if obj['type'] == 'Cuboid':
                rect = patches.Rectangle(
                    (obj['position'][1] - obj['dimensions'][1]/2, 
                     obj['position'][2] - obj['dimensions'][2]/2),
                    obj['dimensions'][1], obj['dimensions'][2],
                    alpha=0.6, facecolor=colors[i % len(colors)],
                    edgecolor='black', linewidth=1
                )
                ax3.add_patch(rect)
                ax3.scatter([obj['position'][1]], [obj['position'][2]], 
                           c=colors[i % len(colors)], s=50, marker='x')
                ax3.text(obj['position'][1], obj['position'][2], obj['name'], 
                       fontsize=8, ha='center', va='bottom')
        
        ax3.legend()
        
        # Object details table
        ax4.axis('off')
        ax4.set_title('Object Details')
        
        # Create table data
        table_data = []
        headers = ['Object', 'Position (X,Y,Z)', 'Dimensions (W,H,D)', 'Volume (m³)']
        
        for obj in collision_objects:
            if obj['type'] == 'Cuboid':
                pos_str = f"({obj['position'][0]:.2f}, {obj['position'][1]:.2f}, {obj['position'][2]:.2f})"
                dim_str = f"({obj['dimensions'][0]:.2f}, {obj['dimensions'][1]:.2f}, {obj['dimensions'][2]:.2f})"
                volume = obj['dimensions'][0] * obj['dimensions'][1] * obj['dimensions'][2]
                table_data.append([obj['name'], pos_str, dim_str, f"{volume:.3f}"])
        
        # Create table
        table = ax4.table(cellText=table_data, colLabels=headers, 
                         cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        # Add statistics
        fig3.suptitle(f'RealSense Collision Objects Visualization\n{len(collision_objects)} objects detected from {len(pcd)} points', 
                    fontsize=14)
        
        # Add summary text
        total_volume = sum(obj['dimensions'][0] * obj['dimensions'][1] * obj['dimensions'][2] 
                          for obj in collision_objects if obj['type'] == 'Cuboid')
        
        summary_text = f"""
Summary:
• Total objects: {len(collision_objects)}
• Total volume: {total_volume:.3f} m³
• Point cloud: {len(pcd)} points
• Workspace: {workspace_bounds['x'][1] - workspace_bounds['x'][0]:.1f}m × {workspace_bounds['y'][1] - workspace_bounds['y'][0]:.1f}m × {workspace_bounds['z'][1] - workspace_bounds['z'][0]:.1f}m
        """
        
        fig3.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/collision_objects_comprehensive.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved comprehensive view to {output_dir}/collision_objects_comprehensive.png")
        plt.close()
        
        # 4. Point cloud visualization
        fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # X-Y projection (top-down view)
        scatter1 = ax1.scatter(pcd[:, 0], pcd[:, 1], c=pcd[:, 2], cmap='viridis', alpha=0.6, s=1)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Point Cloud - Top-Down View (X-Y)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, label='Z (m)')
        
        # X-Z projection (side view)
        scatter2 = ax2.scatter(pcd[:, 0], pcd[:, 2], c=pcd[:, 1], cmap='plasma', alpha=0.6, s=1)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Point Cloud - Side View (X-Z)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2, label='Y (m)')
        
        # Add statistics
        fig4.suptitle(f'RealSense Point Cloud Visualization\n{len(pcd)} points', fontsize=14)
        
        # Add statistics text
        stats_text = f"""
Point Cloud Statistics:
• Total points: {len(pcd)}
• X range: {pcd[:, 0].min():.3f} to {pcd[:, 0].max():.3f} m
• Y range: {pcd[:, 1].min():.3f} to {pcd[:, 1].max():.3f} m
• Z range: {pcd[:, 2].min():.3f} to {pcd[:, 2].max():.3f} m
• Mean position: ({pcd[:, 0].mean():.3f}, {pcd[:, 1].mean():.3f}, {pcd[:, 2].mean():.3f})
        """
        
        fig4.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/point_cloud_visualization.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved point cloud visualization to {output_dir}/point_cloud_visualization.png")
        plt.close()
        
        print("\n✅ All visualizations saved successfully!")
        
        # Print detailed information
        print("\n" + "="*50)
        print("DETAILED COLLISION OBJECTS INFORMATION")
        print("="*50)
        
        for i, obj in enumerate(collision_objects):
            if obj['type'] == 'Cuboid':
                volume = obj['dimensions'][0] * obj['dimensions'][1] * obj['dimensions'][2]
                print(f"\n{obj['name']}:")
                print(f"  Position: ({obj['position'][0]:.3f}, {obj['position'][1]:.3f}, {obj['position'][2]:.3f})")
                print(f"  Dimensions: {obj['dimensions'][0]:.3f}m × {obj['dimensions'][1]:.3f}m × {obj['dimensions'][2]:.3f}m")
                print(f"  Volume: {volume:.3f} m³")
                print(f"  Orientation: ({obj['orientation'][0]:.3f}, {obj['orientation'][1]:.3f}, {obj['orientation'][2]:.3f}, {obj['orientation'][3]:.3f})")
        
        print(f"\nTotal volume of collision objects: {total_volume:.3f} m³")
        print(f"\nVisualization files saved in: {output_dir}/")
        
    else:
        print("❌ Failed to capture point cloud")
    
    # Cleanup
    camera.stop()
    motion_planner.disconnect_robot()
    return True


if __name__ == "__main__":
    print("Collision Objects Visualization (Save to Files)")
    print("="*60)
    
    visualize_collision_objects_save()
    
    print("\n✅ Visualization script completed!") 