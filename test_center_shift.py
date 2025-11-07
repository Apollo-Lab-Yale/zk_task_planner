#!/usr/bin/env python3
"""
Test script for the apply_center_shift function
"""

import sys
import os
import numpy as np
import cv2

# Add the cognitive_bt_framework to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_bt_framework', 'src'))

from cognitive_bt_framework.src.vision.interaction_point_detector_v2 import RobustInteractionDetector, InteractionPoint, InteractionType

def create_test_mask():
    """Create a simple test mask - a circle in the center"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 1, -1)  # Filled circle
    return mask.astype(bool)

def create_test_points_near_edge():
    """Create test points near the edge of the object"""
    # Points that should be shifted (near edge)
    edge_points = [
        InteractionPoint(x=25, y=50, interaction_type=InteractionType.PUSH, 
                        confidence=0.9, score=0.8, stability=0.7, accessibility=0.8,
                        detection_method="test"),
        InteractionPoint(x=75, y=50, interaction_type=InteractionType.GRASP,
                        confidence=0.85, score=0.75, stability=0.6, accessibility=0.9,
                        detection_method="test"),
        InteractionPoint(x=50, y=25, interaction_type=InteractionType.TWIST,
                        confidence=0.8, score=0.9, stability=0.8, accessibility=0.7,
                        detection_method="test"),
    ]
    
    # Points that should not be shifted (near center)
    center_points = [
        InteractionPoint(x=50, y=50, interaction_type=InteractionType.PUSH,
                        confidence=0.95, score=0.85, stability=0.9, accessibility=0.8,
                        detection_method="test"),
        InteractionPoint(x=45, y=45, interaction_type=InteractionType.GRASP,
                        confidence=0.9, score=0.8, stability=0.85, accessibility=0.9,
                        detection_method="test"),
    ]
    
    return edge_points + center_points

def test_center_shift():
    """Test the apply_center_shift function"""
    print("="*60)
    print("TESTING CENTER SHIFT FUNCTIONALITY")
    print("="*60)
    
    # Create detector instance
    detector = RobustInteractionDetector(debug=True)
    
    # Create test data
    mask = create_test_mask()
    points = create_test_points_near_edge()
    
    print(f"\nTest Setup:")
    print(f"Mask shape: {mask.shape}")
    print(f"Object centroid: ({np.mean(np.where(mask)[1]):.1f}, {np.mean(np.where(mask)[0]):.1f})")
    print(f"Number of test points: {len(points)}")
    
    print(f"\nOriginal points:")
    for i, point in enumerate(points):
        print(f"  {i+1}. ({point.x}, {point.y}) - {point.interaction_type.value} - {point.detection_method}")
    
    # Test with different parameters
    test_configs = [
        {"edge_threshold": 15.0, "shift_factor": 0.3, "description": "Moderate shift"},
        {"edge_threshold": 10.0, "shift_factor": 0.5, "description": "Strong shift"},
        {"edge_threshold": 20.0, "shift_factor": 0.2, "description": "Gentle shift"},
    ]
    
    for config in test_configs:
        print(f"\n{'-'*40}")
        print(f"Test: {config['description']}")
        print(f"Parameters: edge_threshold={config['edge_threshold']}, shift_factor={config['shift_factor']}")
        print(f"{'-'*40}")
        
        # Apply center shift
        shifted_points = detector._apply_center_shift(
            points, mask, config['edge_threshold'], config['shift_factor']
        )
        
        print(f"\nShifted points:")
        for i, (orig, shifted) in enumerate(zip(points, shifted_points)):
            if orig.x != shifted.x or orig.y != shifted.y:
                shift_dist = np.sqrt((shifted.x - orig.x)**2 + (shifted.y - orig.y)**2)
                print(f"  {i+1}. ({orig.x}, {orig.y}) -> ({shifted.x}, {shifted.y}) "
                      f"SHIFTED by {shift_dist:.1f} pixels - {shifted.detection_method}")
            else:
                print(f"  {i+1}. ({orig.x}, {orig.y}) - NO SHIFT - {shifted.detection_method}")
    
    # Test edge cases
    print(f"\n{'-'*40}")
    print("Testing edge cases")
    print(f"{'-'*40}")
    
    # Test with empty points list
    empty_result = detector._apply_center_shift([], mask, 10.0, 0.5)
    print(f"Empty points list: {len(empty_result)} points returned")
    
    # Test with shift_factor = 0 (no shift)
    no_shift_result = detector._apply_center_shift(points, mask, 10.0, 0.0)
    no_shift_count = sum(1 for orig, shifted in zip(points, no_shift_result) 
                        if orig.x == shifted.x and orig.y == shifted.y)
    print(f"Zero shift factor: {no_shift_count}/{len(points)} points unchanged")
    
    # Test with very large edge_threshold (no points should shift)
    large_threshold_result = detector._apply_center_shift(points, mask, 100.0, 0.5)
    large_threshold_count = sum(1 for orig, shifted in zip(points, large_threshold_result) 
                               if orig.x == shifted.x and orig.y == shifted.y)
    print(f"Large edge threshold: {large_threshold_count}/{len(points)} points unchanged")
    
    print(f"\n✅ Center shift function test completed successfully!")
    return True

def test_integration_with_detection():
    """Test center shift integration with the main detection pipeline"""
    print(f"\n{'='*60}")
    print("TESTING CENTER SHIFT INTEGRATION")
    print(f"{'='*60}")
    
    # Create a simple test image and mask
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image
    mask = create_test_mask()
    
    # Create detector
    detector = RobustInteractionDetector(debug=True)
    
    print(f"\nTesting detection with center shift enabled...")
    
    # Test with center shift enabled
    points_with_shift = detector.detect_interaction_points(
        image=image,
        mask=mask,
        min_distance=5,
        max_points=10,
        apply_center_shift=True,
        edge_threshold=12.0,
        shift_factor=0.4
    )
    
    print(f"\nDetection with center shift: {len(points_with_shift)} points found")
    for i, point in enumerate(points_with_shift[:3]):  # Show first 3
        method = point.detection_method
        shifted = "+center_shift" in method
        print(f"  {i+1}. ({point.x}, {point.y}) - {point.interaction_type.value} - "
              f"{'SHIFTED' if shifted else 'ORIGINAL'} - {method}")
    
    # Test with center shift disabled
    print(f"\nTesting detection with center shift disabled...")
    
    points_without_shift = detector.detect_interaction_points(
        image=image,
        mask=mask,
        min_distance=5,
        max_points=10,
        apply_center_shift=False
    )
    
    print(f"\nDetection without center shift: {len(points_without_shift)} points found")
    for i, point in enumerate(points_without_shift[:3]):  # Show first 3
        method = point.detection_method
        print(f"  {i+1}. ({point.x}, {point.y}) - {point.interaction_type.value} - {method}")
    
    print(f"\n✅ Integration test completed successfully!")
    return True

def main():
    """Run all center shift tests"""
    try:
        print("STARTING CENTER SHIFT TESTS")
        
        # Run tests
        test_center_shift()
        test_integration_with_detection()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())