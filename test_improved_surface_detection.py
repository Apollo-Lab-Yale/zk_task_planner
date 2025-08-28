#!/usr/bin/env python3

"""
Test script for improved surface detection in interaction_point_detector_v2.py
Tests the new surface normal segmentation and boundary detection features.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from cognitive_bt_framework.src.vision.interaction_point_detector_v2 import RobustInteractionDetector

def create_synthetic_multi_surface_depth():
    """Create a synthetic depth image with multiple distinct surfaces."""
    h, w = 200, 200
    depth = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    
    # Create multiple surfaces with different orientations
    
    # Horizontal surface (top - like a bottle cap)
    depth[50:80, 80:120] = 0.5  # 50cm depth
    mask[50:80, 80:120] = True
    
    # Vertical surface (side - like bottle wall)
    for i in range(30):
        depth[80+i:120, 70+i:110] = 0.5 + i * 0.01  # Slanted depth
        mask[80+i:120, 70+i:110] = True
    
    # Another horizontal surface at different depth (bottom surface)
    depth[120:150, 60:140] = 0.7  # 70cm depth  
    mask[120:150, 60:140] = True
    
    # Angled surface (ramp)
    for i in range(40):
        for j in range(30):
            depth[160-i, 90+j] = 0.6 + i * 0.005
            mask[160-i, 90+j] = True
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.002, depth.shape)
    depth += noise
    depth[~mask] = 0
    
    return depth, mask

def create_synthetic_rgb():
    """Create a synthetic RGB image to go with the depth."""
    h, w = 200, 200
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add different colors for different surfaces
    rgb[50:80, 80:120] = [200, 100, 100]    # Red for top surface
    rgb[80:120, 70:110] = [100, 200, 100]   # Green for side surface  
    rgb[120:150, 60:140] = [100, 100, 200]  # Blue for bottom surface
    rgb[120:160, 90:120] = [200, 200, 100]  # Yellow for ramp
    
    return rgb

def test_surface_detection():
    """Test the improved surface detection algorithm."""
    print("Testing improved surface detection...")
    
    # Create synthetic data
    depth, mask = create_synthetic_multi_surface_depth()
    rgb = create_synthetic_rgb()
    
    print(f"Created synthetic data: depth shape {depth.shape}, mask has {np.sum(mask)} pixels")
    
    # Initialize detector with debug enabled
    detector = RobustInteractionDetector(
        debug=True,
        min_distance=20,
        cascade_enabled=False  # Disable to see all detection methods
    )
    
    # Detect interaction points
    print("\n" + "="*50)
    print("RUNNING INTERACTION POINT DETECTION")
    print("="*50)
    
    points = detector.detect_interaction_points(
        image=rgb,
        mask=mask,
        depth_data=depth,
        max_points=20,
        fast_mode=False
    )
    
    print(f"\nDetected {len(points)} interaction points:")
    for i, point in enumerate(points):
        print(f"  {i+1}. ({point.x:3d}, {point.y:3d}) - {point.interaction_type.value:12s} - "
              f"score: {point.score:.2f}, conf: {point.confidence:.2f}")
    
    # Visualize results
    vis_img = detector.visualize_points(rgb, points, show_scores=True, show_types=True)
    
    # Create a combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis('off')
    
    # Depth with mask overlay
    masked_depth = depth.copy()
    masked_depth[~mask] = np.nan
    im1 = axes[0, 1].imshow(masked_depth, cmap='viridis')
    axes[0, 1].set_title("Depth Data")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Detected points
    axes[1, 0].imshow(vis_img)
    axes[1, 0].set_title(f"Detected Points ({len(points)} total)")
    axes[1, 0].axis('off')
    
    # Surface type distribution
    type_counts = {}
    for point in points:
        type_name = point.interaction_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    if type_counts:
        axes[1, 1].bar(type_counts.keys(), type_counts.values())
        axes[1, 1].set_title("Surface Type Distribution")
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, "No points detected", ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Surface Type Distribution")
    
    plt.tight_layout()
    plt.savefig('improved_surface_detection_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'improved_surface_detection_test.png'")
    
    # Test expectations
    print("\n" + "="*50)
    print("TEST RESULTS ANALYSIS")
    print("="*50)
    
    success = True
    
    # We should detect multiple surface types
    unique_types = set(p.interaction_type for p in points)
    print(f"Detected {len(unique_types)} unique surface types: {[t.value for t in unique_types]}")
    
    if len(unique_types) < 2:
        print("❌ FAIL: Should detect at least 2 different surface types")
        success = False
    else:
        print("✅ PASS: Multiple surface types detected")
    
    # Points should be reasonably distributed
    if len(points) < 4:
        print("❌ FAIL: Should detect at least 4 interaction points for multi-surface object")
        success = False
    else:
        print("✅ PASS: Adequate number of points detected")
    
    # Check if we have points on different surfaces (rough spatial distribution)
    if points:
        y_coords = [p.y for p in points]
        y_range = max(y_coords) - min(y_coords)
        if y_range < 50:  # Points should span at least 50 pixels vertically
            print("❌ FAIL: Points should be more spatially distributed")
            success = False
        else:
            print("✅ PASS: Points are spatially distributed across surfaces")
    
    print(f"\nOverall test result: {'✅ PASS' if success else '❌ FAIL'}")
    return success

if __name__ == "__main__":
    success = test_surface_detection()
    exit(0 if success else 1)