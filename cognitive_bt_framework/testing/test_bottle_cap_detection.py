#!/usr/bin/env python3

import numpy as np
import cv2
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision.interaction_point_detector import RobustInteractionDetector, InteractionType

def create_bottle_with_cap():
    """Create a realistic bottle with cap depth data for testing."""
    # Create a 300x400 depth image
    depth = np.zeros((400, 300), dtype=np.float32)
    
    # Bottle body (cylindrical)
    bottle_center_x = 150
    bottle_radius = 40
    
    for y in range(100, 350):  # Bottle height
        for x in range(300):
            dist_from_center = abs(x - bottle_center_x)
            if dist_from_center <= bottle_radius:
                # Create cylindrical bottle with slight curve
                curve_depth = np.sqrt(bottle_radius**2 - dist_from_center**2) * 0.3
                depth[y, x] = 120 + curve_depth
    
    # Bottle neck (narrower)
    neck_radius = 15
    for y in range(60, 100):
        for x in range(300):
            dist_from_center = abs(x - bottle_center_x)
            if dist_from_center <= neck_radius:
                curve_depth = np.sqrt(neck_radius**2 - dist_from_center**2) * 0.2
                depth[y, x] = 125 + curve_depth
    
    # Bottle cap (circular, raised, flat top)
    cap_center_x, cap_center_y = 150, 50
    cap_radius = 20
    cap_height = 140  # Higher than bottle neck
    
    for y in range(400):
        for x in range(300):
            dist_from_cap_center = np.sqrt((x - cap_center_x)**2 + (y - cap_center_y)**2)
            if dist_from_cap_center <= cap_radius:
                if dist_from_cap_center <= cap_radius - 3:
                    # Flat top of cap
                    depth[y, x] = cap_height
                else:
                    # Rounded edge of cap
                    edge_height = cap_height - (dist_from_cap_center - (cap_radius - 3)) * 2
                    depth[y, x] = max(edge_height, depth[y, x])
    
    # Add some realistic texture and noise
    noise = np.random.normal(0, 0.3, depth.shape)
    depth += noise
    depth[depth < 0] = 0
    
    # Create mask and RGB
    mask = depth > 0
    rgb = np.ones((400, 300, 3), dtype=np.uint8) * 128
    
    # Add some visual texture to help understand structure
    rgb[mask] = [100, 150, 200]  # Bottle color
    cap_mask = np.zeros_like(mask)
    for y in range(400):
        for x in range(300):
            if np.sqrt((x - cap_center_x)**2 + (y - cap_center_y)**2) <= cap_radius:
                cap_mask[y, x] = True
    rgb[cap_mask] = [200, 100, 100]  # Cap color (reddish)
    
    return rgb, depth, mask, (cap_center_x, cap_center_y, cap_radius)

def test_bottle_cap_detection():
    """Test surface-centered detection specifically for bottle cap scenario."""
    print("Testing bottle cap surface detection...")
    
    # Create bottle with cap
    rgb, depth, mask, cap_info = create_bottle_with_cap()
    cap_center_x, cap_center_y, cap_radius = cap_info
    
    print(f"Ground truth cap center: ({cap_center_x}, {cap_center_y})")
    print(f"Ground truth cap radius: {cap_radius}")
    
    # Initialize detector with debug features enabled
    detector = RobustInteractionDetector(debug=True, save_debug_images=True, debug_output_dir="bottle_cap_debug")
    
    # Detect interaction points
    points = detector.detect_interaction_points(
        image=rgb,
        mask=mask,
        depth_data=depth,
        max_points=20,
        min_distance=15
    )
    
    print(f"\nDetected {len(points)} total interaction points")
    
    # Analyze surface points specifically
    surface_points = [p for p in points if p.interaction_type == InteractionType.GRASP_SURFACE]
    print(f"Surface-centered points: {len(surface_points)}")
    
    # Find points on the cap
    cap_points = []
    for point in points:
        dist_from_cap = np.sqrt((point.x - cap_center_x)**2 + (point.y - cap_center_y)**2)
        if dist_from_cap <= cap_radius + 5:  # Allow small tolerance
            cap_points.append((point, dist_from_cap))
    
    print(f"\nPoints detected on/near bottle cap: {len(cap_points)}")
    
    # Analyze cap points
    if cap_points:
        print("\nCap point analysis:")
        cap_points.sort(key=lambda x: x[1])  # Sort by distance from cap center
        
        for i, (point, dist) in enumerate(cap_points):
            print(f"  Point {i+1}:")
            print(f"    Position: ({point.x}, {point.y})")
            print(f"    Distance from cap center: {dist:.1f} pixels")
            print(f"    Type: {point.interaction_type.value}")
            print(f"    Score: {point.score:.3f}")
            print(f"    Confidence: {point.confidence:.3f}")
            if point.surface_area is not None:
                print(f"    Surface area: {point.surface_area:.1f}")
            if point.surface_stability is not None:
                print(f"    Surface stability: {point.surface_stability:.3f}")
        
        # Check if we found a well-centered point on the cap
        best_cap_point = cap_points[0][0]  # Closest to cap center
        best_distance = cap_points[0][1]
        
        print(f"\nBest cap point distance from center: {best_distance:.1f} pixels")
        
        # Quality assessment
        if best_distance <= cap_radius * 0.3:  # Within 30% of radius from center
            print("✅ EXCELLENT: Point is well-centered on cap")
        elif best_distance <= cap_radius * 0.5:  # Within 50% of radius
            print("✅ GOOD: Point is reasonably centered on cap")
        elif best_distance <= cap_radius * 0.7:  # Within 70% of radius
            print("⚠️  OK: Point is somewhat centered on cap")
        else:
            print("❌ POOR: Point is not well-centered on cap")
        
        # Check if surface points are prioritized
        surface_cap_points = [p for p, d in cap_points if p.interaction_type == InteractionType.GRASP_SURFACE]
        if surface_cap_points:
            print(f"✅ Found {len(surface_cap_points)} surface-type points on cap")
        else:
            print("⚠️  No surface-type points found on cap")
    
    else:
        print("❌ No points detected on bottle cap!")
    
    # Check overall point distribution
    print(f"\nPoint type distribution:")
    type_counts = {}
    for point in points:
        type_name = point.interaction_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    for type_name, count in sorted(type_counts.items()):
        print(f"  {type_name}: {count}")
    
    return points, cap_points, cap_info

def visualize_bottle_cap_results(rgb, depth, mask, points, cap_points, cap_info):
    """Visualize bottle cap detection results."""
    try:
        import matplotlib.pyplot as plt
        
        cap_center_x, cap_center_y, cap_radius = cap_info
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original RGB
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Bottle with Cap (RGB)')
        axes[0, 0].axis('off')
        
        # Depth map
        depth_vis = depth.copy()
        depth_vis[~mask] = np.nan
        im1 = axes[0, 1].imshow(depth_vis, cmap='viridis')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # All interaction points on RGB
        axes[0, 2].imshow(rgb)
        for i, point in enumerate(points[:15]):  # Show top 15
            # Color code by type
            if point.interaction_type == InteractionType.GRASP_SURFACE:
                color, marker, size = 'red', 'o', 120
            elif point.interaction_type == InteractionType.HANDLE:
                color, marker, size = 'green', 's', 100
            elif point.interaction_type == InteractionType.GRASP_EDGE:
                color, marker, size = 'blue', '^', 80
            else:
                color, marker, size = 'white', 'x', 60
            
            axes[0, 2].scatter(point.x, point.y, c=color, marker=marker, s=size, alpha=0.8)
            axes[0, 2].text(point.x+3, point.y-8, f'{i+1}', fontsize=8, color='black', weight='bold')
        
        # Draw cap outline
        cap_circle = plt.Circle((cap_center_x, cap_center_y), cap_radius, 
                               fill=False, color='yellow', linewidth=3, linestyle='--')
        axes[0, 2].add_patch(cap_circle)
        axes[0, 2].plot(cap_center_x, cap_center_y, 'y*', markersize=15, label='True Cap Center')
        
        axes[0, 2].set_title('All Interaction Points\n(Surface=red○, Handle=green□, Edge=blue△)')
        axes[0, 2].axis('off')
        axes[0, 2].legend()
        
        # Cap points only on depth
        axes[1, 0].imshow(depth_vis, cmap='viridis')
        for i, (point, dist) in enumerate(cap_points):
            color = 'red' if point.interaction_type == InteractionType.GRASP_SURFACE else 'white'
            axes[1, 0].scatter(point.x, point.y, c=color, s=150, alpha=0.9)
            axes[1, 0].text(point.x+5, point.y+5, f'{i+1}\n({dist:.1f})', 
                           fontsize=8, color='white', weight='bold')
        
        # Draw cap outline
        cap_circle2 = plt.Circle((cap_center_x, cap_center_y), cap_radius, 
                                fill=False, color='yellow', linewidth=2)
        axes[1, 0].add_patch(cap_circle2)
        axes[1, 0].plot(cap_center_x, cap_center_y, 'y*', markersize=12)
        
        axes[1, 0].set_title('Cap Points on Depth\n(Numbers show distance from center)')
        axes[1, 0].axis('off')
        
        # Point quality analysis
        if cap_points:
            distances = [dist for _, dist in cap_points]
            scores = [point.score for point, _ in cap_points]
            
            axes[1, 1].scatter(distances, scores, c='blue', alpha=0.7)
            axes[1, 1].axvline(x=cap_radius*0.3, color='green', linestyle='--', label='Excellent threshold')
            axes[1, 1].axvline(x=cap_radius*0.5, color='orange', linestyle='--', label='Good threshold')
            axes[1, 1].set_xlabel('Distance from Cap Center (pixels)')
            axes[1, 1].set_ylabel('Point Score')
            axes[1, 1].set_title('Cap Point Quality vs Centering')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No cap points detected', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Cap Point Quality Analysis')
        
        # Surface detection analysis
        surface_points = [p for p in points if p.interaction_type == InteractionType.GRASP_SURFACE]
        if surface_points:
            # Show surface quality metrics
            stabilities = [p.surface_stability for p in surface_points if p.surface_stability is not None]
            planarities = [p.surface_planarity for p in surface_points if p.surface_planarity is not None]
            areas = [p.surface_area for p in surface_points if p.surface_area is not None]
            
            if stabilities and planarities:
                axes[1, 2].scatter(stabilities, planarities, s=[a/5 if a else 50 for a in areas], 
                                  c='red', alpha=0.7)
                axes[1, 2].set_xlabel('Surface Stability')
                axes[1, 2].set_ylabel('Surface Planarity')
                axes[1, 2].set_title('Surface Quality Metrics\n(Size = Area)')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'No surface quality data', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Surface Quality Metrics')
        else:
            axes[1, 2].text(0.5, 0.5, 'No surface points detected', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Surface Quality Metrics')
        
        plt.tight_layout()
        plt.savefig('bottle_cap_detection_results.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to bottle_cap_detection_results.png")
        
    except ImportError:
        print("Matplotlib not available for visualization")

def analyze_centering_accuracy(cap_points, cap_info):
    """Analyze how well points are centered on the cap."""
    if not cap_points:
        return
    
    cap_center_x, cap_center_y, cap_radius = cap_info
    
    print("\n" + "="*50)
    print("CENTERING ACCURACY ANALYSIS")
    print("="*50)
    
    # Find the best centered point
    best_point, best_distance = min(cap_points, key=lambda x: x[1])
    
    print(f"Best centered point:")
    print(f"  Position: ({best_point.x}, {best_point.y})")
    print(f"  True cap center: ({cap_center_x}, {cap_center_y})")
    print(f"  Distance from center: {best_distance:.1f} pixels")
    print(f"  Relative to cap radius: {best_distance/cap_radius:.1%}")
    print(f"  Point type: {best_point.interaction_type.value}")
    print(f"  Score: {best_point.score:.3f}")
    
    # Calculate centering quality
    centering_quality = max(0, 1 - (best_distance / cap_radius))
    print(f"  Centering quality: {centering_quality:.1%}")
    
    if centering_quality > 0.7:
        print("  Assessment: EXCELLENT centering ✅")
    elif centering_quality > 0.5:
        print("  Assessment: GOOD centering ✅")
    elif centering_quality > 0.3:
        print("  Assessment: ACCEPTABLE centering ⚠️")
    else:
        print("  Assessment: POOR centering ❌")

if __name__ == "__main__":
    print("Starting bottle cap detection test...")
    
    # Run the test
    points, cap_points, cap_info = test_bottle_cap_detection()
    
    # Analyze centering accuracy
    analyze_centering_accuracy(cap_points, cap_info)
    
    # Create visualization
    rgb, depth, mask, _ = create_bottle_with_cap()
    visualize_bottle_cap_results(rgb, depth, mask, points, cap_points, cap_info)
    
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total points detected: {len(points)}")
    print(f"Points on cap: {len(cap_points)}")
    surface_points = [p for p in points if p.interaction_type == InteractionType.GRASP_SURFACE]
    print(f"Surface points total: {len(surface_points)}")
    
    if cap_points:
        best_distance = min(cap_points, key=lambda x: x[1])[1]
        cap_radius = cap_info[2]
        success = best_distance <= cap_radius * 0.5
        print(f"Well-centered cap detection: {'✅ SUCCESS' if success else '❌ NEEDS IMPROVEMENT'}")
    else:
        print("Cap detection: ❌ FAILED - No points on cap")
    
    print("\nTest completed!")