#!/usr/bin/env python3
"""
Test script for the enhanced interaction point detection system.
Demonstrates the new depth clustering and visual feature integration.
"""
import sys
import numpy as np
import cv2

# Add the project path
sys.path.append('cognitive_bt_framework/src')

from vision.interaction_point_detector import RobustInteractionDetector, InteractionType

def create_synthetic_test_data():
    """Create synthetic RGB image, depth data, and mask for testing."""
    # Create a synthetic object with varying depth features
    height, width = 240, 320
    
    # RGB image with texture and edges
    image = np.random.randint(80, 180, (height, width, 3), dtype=np.uint8)
    
    # Create circular mask for object
    center = (width//2, height//2)
    radius = min(width, height) // 3
    y, x = np.ogrid[:height, :width]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Add visual texture to the object region
    object_region = image[mask]
    # Create vertical stripes pattern
    for i in range(0, width, 20):
        if i < width:
            image[:, i:i+5] = [200, 150, 100]  # Bright stripes
    
    # Create depth data with distinct surface clusters
    depth = np.zeros((height, width), dtype=np.float32)
    depth[mask] = 0.5  # Base depth of 50cm
    
    # Create multiple distinct depth regions for clustering
    # Center raised platform
    center_mask = (x - center[0])**2 + (y - center[1])**2 <= (radius//3)**2
    depth[center_mask & mask] = 0.42  # Much closer surface
    
    # Left side depression  
    left_mask = mask & (x < center[0] - radius//3)
    depth[left_mask] = 0.58  # Much further surface
    
    # Right side slight elevation
    right_mask = mask & (x > center[0] + radius//3)
    depth[right_mask] = 0.47  # Slightly closer
    
    # Add minimal noise to preserve clusters
    depth[mask] += np.random.normal(0, 0.002, np.sum(mask))
    
    return image, depth, mask

def test_standard_vs_enhanced_detection():
    """Compare standard depth plane detection vs enhanced clustering."""
    print("=== Testing Enhanced Interaction Point Detection ===\n")
    
    # Create test data
    image, depth, mask = create_synthetic_test_data()
    print(f"Created synthetic test data: {image.shape} image, {depth.shape} depth")
    
    # Initialize detector
    detector = RobustInteractionDetector(debug=True)
    
    # Test the enhanced clustering method with custom parameters
    def test_enhanced_clustering():
        visual_features = detector._extract_visual_features(image, mask)
        return detector._detect_depth_clusters_advanced(
            depth, mask, 
            visual_features=visual_features,
            cluster_eps=0.5,  # More lenient clustering
            min_cluster_size=20,  # Smaller minimum cluster size
            use_visual_guidance=True
        )
    
    print("\n--- Testing Standard Depth Plane Detection ---")
    standard_points = detector.detect_interaction_points(
        image=image,
        mask=mask,
        depth_data=depth,
        max_points=10,
        depth_plane_only=True
    )
    
    print(f"Standard method found {len(standard_points)} points:")
    for i, point in enumerate(standard_points[:3]):  # Show first 3
        print(f"  Point {i+1}: ({point.x}, {point.y}) - {point.interaction_type.value} (conf: {point.confidence:.2f})")
    
    print("\n--- Testing Enhanced Depth Clustering with Visual Guidance ---")
    # Test the enhanced clustering directly with better parameters
    enhanced_points = test_enhanced_clustering()
    
    print(f"Enhanced method found {len(enhanced_points)} points:")
    for i, point in enumerate(enhanced_points[:3]):  # Show first 3
        stability = getattr(point, 'surface_stability', 'N/A')
        planarity = getattr(point, 'surface_planarity', 'N/A')
        print(f"  Point {i+1}: ({point.x}, {point.y}) - {point.interaction_type.value} (conf: {point.confidence:.2f}, stability: {stability}, planarity: {planarity})")
    
    print(f"\n=== Summary ===")
    print(f"Standard method: {len(standard_points)} points")
    print(f"Enhanced method: {len(enhanced_points)} points")
    print("Enhanced method provides additional surface analysis and visual guidance.")

if __name__ == "__main__":
    try:
        test_standard_vs_enhanced_detection()
        print("\n✓ Test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()