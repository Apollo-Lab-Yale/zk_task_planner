#!/usr/bin/env python3

import numpy as np
import cv2
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision.interaction_point_detector import RobustInteractionDetector, InteractionType

def create_test_depth_data():
    """Create synthetic depth data with known features for testing."""
    # Create a 200x200 depth image
    depth = np.zeros((200, 200), dtype=np.float32)
    
    # Create a cylindrical object (higher depth in center)
    center_x, center_y = 100, 100
    radius = 60
    
    for y in range(200):
        for x in range(200):
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist_from_center <= radius:
                # Create cylinder with some noise
                height = np.sqrt(radius**2 - dist_from_center**2)
                noise = np.random.normal(0, 0.1)
                depth[y, x] = 100 + height + noise
    
    # Add some ridges and valleys
    # Ridge along x=50
    depth[80:120, 45:55] += 20
    # Valley along y=150
    depth[145:155, 60:140] -= 15
    
    # Add some peaks
    depth[60:70, 160:170] += 30  # Peak 1
    depth[140:150, 30:40] += 25  # Peak 2
    
    # Create corresponding RGB image and mask
    rgb = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray background
    mask = depth > 0
    
    return rgb, depth, mask

def test_depth_integration():
    """Test the depth-enhanced interaction point detection."""
    print("Testing depth-enhanced interaction point detection...")
    
    # Create test data
    rgb, depth, mask = create_test_depth_data()
    
    # Initialize detector
    detector = RobustInteractionDetector(debug=True)
    
    # Detect interaction points with depth data
    points = detector.detect_interaction_points(
        image=rgb,
        mask=mask,
        depth_data=depth,
        max_points=20,
        min_distance=15
    )
    
    print(f"\nDetected {len(points)} interaction points:")
    
    # Analyze detected points
    for i, point in enumerate(points):
        print(f"\nPoint {i+1}:")
        print(f"  Position: ({point.x}, {point.y})")
        print(f"  Score: {point.score:.3f}")
        print(f"  Type: {point.interaction_type.value}")
        print(f"  Confidence: {point.confidence:.3f}")
        print(f"  Approach angle: {point.approach_angle:.1f}°" if point.approach_angle is not None else "  Approach angle: N/A")
        print(f"  Grasp width: {point.grasp_width:.1f}" if point.grasp_width is not None else "  Grasp width: N/A")
        print(f"  Stability: {point.stability:.3f}")
        print(f"  Accessibility: {point.accessibility:.3f}")
    
    # Test different interaction types
    type_counts = {}
    for point in points:
        type_name = point.interaction_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    print(f"\nInteraction type distribution:")
    for type_name, count in type_counts.items():
        print(f"  {type_name}: {count}")
    
    # Verify 3D features are being used
    points_with_3d = [p for p in points if p.approach_angle is not None]
    print(f"\nPoints with 3D information: {len(points_with_3d)}/{len(points)}")
    
    # Test without depth data for comparison
    print("\n" + "="*50)
    print("Testing without depth data for comparison...")
    
    points_no_depth = detector.detect_interaction_points(
        image=rgb,
        mask=mask,
        depth_data=None,
        max_points=20,
        min_distance=15
    )
    
    print(f"Points without depth: {len(points_no_depth)}")
    print(f"Points with depth: {len(points)}")
    
    # Compare scoring
    if points_no_depth and points:
        avg_score_no_depth = np.mean([p.score for p in points_no_depth])
        avg_score_with_depth = np.mean([p.score for p in points])
        print(f"Average score without depth: {avg_score_no_depth:.3f}")
        print(f"Average score with depth: {avg_score_with_depth:.3f}")
    
    return points, points_no_depth

def visualize_results(rgb, depth, mask, points, save_path=None):
    """Visualize the detection results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original RGB
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        
        # Depth map
        depth_vis = depth.copy()
        depth_vis[~mask] = np.nan
        im = axes[0, 1].imshow(depth_vis, cmap='viridis')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Interaction points on RGB
        axes[1, 0].imshow(rgb)
        for i, point in enumerate(points):
            color = 'red' if point.interaction_type == InteractionType.GRASP_EDGE else 'blue'
            if point.interaction_type == InteractionType.HANDLE:
                color = 'green'
            elif point.interaction_type == InteractionType.GRASP_SURFACE:
                color = 'yellow'
            
            axes[1, 0].scatter(point.x, point.y, c=color, s=50, alpha=0.8)
            axes[1, 0].text(point.x+5, point.y+5, f'{i+1}', fontsize=8, color='white')
        
        axes[1, 0].set_title('Detected Interaction Points')
        axes[1, 0].axis('off')
        
        # Interaction points on depth
        axes[1, 1].imshow(depth_vis, cmap='viridis')
        for i, point in enumerate(points):
            color = 'red' if point.interaction_type == InteractionType.GRASP_EDGE else 'blue'
            if point.interaction_type == InteractionType.HANDLE:
                color = 'green'
            elif point.interaction_type == InteractionType.GRASP_SURFACE:
                color = 'yellow'
            
            axes[1, 1].scatter(point.x, point.y, c=color, s=50, alpha=0.8)
            axes[1, 1].text(point.x+5, point.y+5, f'{i+1}', fontsize=8, color='white')
        
        axes[1, 1].set_title('Points on Depth Map')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    # Run the test
    points, points_no_depth = test_depth_integration()
    
    # Create visualization if matplotlib is available
    rgb, depth, mask = create_test_depth_data()
    visualize_results(rgb, depth, mask, points, 
                     save_path='depth_interaction_test_results.png')
    
    print("\nTest completed successfully!")