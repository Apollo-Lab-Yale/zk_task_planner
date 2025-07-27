#!/usr/bin/env python3

import numpy as np
import cv2
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision.interaction_point_detector import RobustInteractionDetector, InteractionType

def create_multi_surface_test_object():
    """Create a test object with multiple distinct surfaces."""
    # Create a 300x300 depth image
    depth = np.zeros((300, 300), dtype=np.float32)
    
    # Surface 1: Large flat top surface (rectangle)
    depth[50:120, 80:220] = 150
    
    # Surface 2: Side surface (tilted)
    for y in range(120, 180):
        for x in range(80, 220):
            # Create a tilted surface
            tilt_height = (x - 80) * 0.3
            depth[y, x] = 150 - (y - 120) * 0.8 + tilt_height
    
    # Surface 3: Small circular surface (raised button)
    center_x, center_y = 250, 100
    radius = 30
    for y in range(300):
        for x in range(300):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= radius:
                height = np.sqrt(radius**2 - dist**2) * 0.5
                depth[y, x] = 140 + height
    
    # Surface 4: Another flat surface (different orientation)
    depth[200:250, 50:150] = 130
    
    # Surface 5: Curved surface (cylindrical)
    for y in range(60, 140):
        for x in range(30, 70):
            # Create cylindrical curve
            curve_height = 20 * np.cos((x - 50) * np.pi / 40)
            depth[y, x] = 120 + curve_height
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.5, depth.shape)
    depth += noise
    depth[depth < 0] = 0
    
    # Create mask and RGB
    mask = depth > 0
    rgb = np.ones((300, 300, 3), dtype=np.uint8) * 128
    
    return rgb, depth, mask

def test_surface_centered_detection():
    """Test that points are properly centered on individual surfaces."""
    print("Testing surface-centered interaction point detection...")
    
    # Create test data with multiple distinct surfaces
    rgb, depth, mask = create_multi_surface_test_object()
    
    # Initialize detector
    detector = RobustInteractionDetector(debug=True)
    
    # Detect interaction points
    points = detector.detect_interaction_points(
        image=rgb,
        mask=mask,
        depth_data=depth,
        max_points=15,
        min_distance=20
    )
    
    print(f"\nDetected {len(points)} interaction points:")
    
    # Focus on surface points
    surface_points = [p for p in points if p.interaction_type == InteractionType.GRASP_SURFACE]
    
    print(f"\nSurface-centered points: {len(surface_points)}")
    
    for i, point in enumerate(surface_points):
        print(f"\nSurface Point {i+1}:")
        print(f"  Position: ({point.x}, {point.y})")
        print(f"  Score: {point.score:.3f}")
        print(f"  Confidence: {point.confidence:.3f}")
        print(f"  Approach angle: {point.approach_angle:.1f}°" if point.approach_angle is not None else "  Approach angle: N/A")
        print(f"  Surface area: {point.surface_area:.1f}" if point.surface_area is not None else "  Surface area: N/A")
        print(f"  Surface stability: {point.surface_stability:.3f}" if point.surface_stability is not None else "  Surface stability: N/A")
        print(f"  Surface planarity: {point.surface_planarity:.3f}" if point.surface_planarity is not None else "  Surface planarity: N/A")
    
    # Show top-ranked points (should include surface centers)
    print(f"\nTop 5 highest-scored points:")
    top_points = sorted(points, key=lambda p: p.score, reverse=True)[:5]
    
    for i, point in enumerate(top_points):
        print(f"  {i+1}. Type: {point.interaction_type.value}, Score: {point.score:.3f}, Position: ({point.x}, {point.y})")
    
    # Verify surface points have high scores
    surface_scores = [p.score for p in surface_points]
    if surface_scores:
        avg_surface_score = np.mean(surface_scores)
        max_surface_score = np.max(surface_scores)
        print(f"\nSurface point scores - Average: {avg_surface_score:.3f}, Max: {max_surface_score:.3f}")
    
    # Check if surface points are in top ranks
    surface_in_top5 = sum(1 for p in top_points if p.interaction_type == InteractionType.GRASP_SURFACE)
    print(f"Surface points in top 5: {surface_in_top5}")
    
    return points, surface_points

def verify_surface_centering(depth, mask, surface_points):
    """Verify that surface points are well-centered on their surfaces."""
    print("\nVerifying surface centering...")
    
    for i, point in enumerate(surface_points):
        # Check local neighborhood around the point
        x, y = point.x, point.y
        neighborhood_size = 20
        
        y_min = max(0, y - neighborhood_size)
        y_max = min(depth.shape[0], y + neighborhood_size)
        x_min = max(0, x - neighborhood_size)
        x_max = min(depth.shape[1], x + neighborhood_size)
        
        local_depth = depth[y_min:y_max, x_min:x_max]
        local_mask = mask[y_min:y_max, x_min:x_max]
        
        if np.any(local_mask):
            # Calculate depth variance in neighborhood
            local_depths = local_depth[local_mask]
            depth_std = np.std(local_depths)
            depth_range = np.ptp(local_depths)
            
            # Check if point is near the center of the surface
            # Calculate distance from edge within the surface region
            local_dist = cv2.distanceTransform(local_mask.astype(np.uint8), cv2.DIST_L2, 5)
            point_local_y = y - y_min
            point_local_x = x - x_min
            
            if (0 <= point_local_y < local_dist.shape[0] and 
                0 <= point_local_x < local_dist.shape[1]):
                distance_from_edge = local_dist[point_local_y, point_local_x]
                
                print(f"  Surface {i+1}: Depth std: {depth_std:.2f}, Range: {depth_range:.2f}, Distance from edge: {distance_from_edge:.1f}")
            else:
                print(f"  Surface {i+1}: Point outside local region")

def visualize_surface_results(rgb, depth, mask, points, surface_points):
    """Visualize surface-centered detection results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original depth
        depth_vis = depth.copy()
        depth_vis[~mask] = np.nan
        im1 = axes[0, 0].imshow(depth_vis, cmap='viridis')
        axes[0, 0].set_title('Original Depth Map')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # All interaction points
        axes[0, 1].imshow(depth_vis, cmap='viridis')
        for i, point in enumerate(points[:10]):  # Show top 10
            color = 'red' if point.interaction_type == InteractionType.GRASP_SURFACE else 'white'
            marker = 'o' if point.interaction_type == InteractionType.GRASP_SURFACE else 'x'
            size = 100 if point.interaction_type == InteractionType.GRASP_SURFACE else 50
            
            axes[0, 1].scatter(point.x, point.y, c=color, s=size, marker=marker, alpha=0.8)
            axes[0, 1].text(point.x+5, point.y+5, f'{i+1}', fontsize=8, color='white')
        
        axes[0, 1].set_title('All Interaction Points (Surface=red circles)')
        axes[0, 1].axis('off')
        
        # Surface points only
        axes[0, 2].imshow(depth_vis, cmap='viridis')
        for i, point in enumerate(surface_points):
            axes[0, 2].scatter(point.x, point.y, c='red', s=200, marker='o', alpha=0.9)
            axes[0, 2].text(point.x+8, point.y+8, f'S{i+1}', fontsize=10, color='white', weight='bold')
        
        axes[0, 2].set_title('Surface-Centered Points Only')
        axes[0, 2].axis('off')
        
        # Score distribution
        all_scores = [p.score for p in points]
        surface_scores = [p.score for p in surface_points]
        
        axes[1, 0].hist(all_scores, bins=10, alpha=0.7, label='All points', color='blue')
        if surface_scores:
            axes[1, 0].hist(surface_scores, bins=10, alpha=0.7, label='Surface points', color='red')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        
        # Interaction type distribution
        type_counts = {}
        for point in points:
            type_name = point.interaction_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        axes[1, 1].bar(types, counts)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Interaction Type Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Surface quality metrics
        if surface_points:
            stabilities = [p.surface_stability for p in surface_points if p.surface_stability is not None]
            planarities = [p.surface_planarity for p in surface_points if p.surface_planarity is not None]
            
            axes[1, 2].scatter(stabilities, planarities, c='red', s=100, alpha=0.7)
            axes[1, 2].set_xlabel('Surface Stability')
            axes[1, 2].set_ylabel('Surface Planarity')
            axes[1, 2].set_title('Surface Quality Metrics')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No surface points detected', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Surface Quality Metrics')
        
        plt.tight_layout()
        plt.savefig('surface_centered_test_results.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to surface_centered_test_results.png")
        
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    # Run the test
    points, surface_points = test_surface_centered_detection()
    
    # Verify centering
    rgb, depth, mask = create_multi_surface_test_object()
    verify_surface_centering(depth, mask, surface_points)
    
    # Create visualization
    visualize_surface_results(rgb, depth, mask, points, surface_points)
    
    print(f"\nTest completed! Found {len(surface_points)} surface-centered points out of {len(points)} total points.")