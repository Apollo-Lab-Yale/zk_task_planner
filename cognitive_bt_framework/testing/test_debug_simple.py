#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision.interaction_point_detector import RobustInteractionDetector

def test_debug_functionality():
    """Simple test to verify debug functionality works."""
    print("Testing debug functionality...")
    
    # Create simple test data
    rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
    depth = np.random.rand(100, 100) * 50 + 100
    mask = np.ones((100, 100), dtype=bool)
    
    # Add a simple surface (flat region)
    depth[40:60, 40:60] = 150  # Flat square
    
    # Initialize detector with debug
    detector = RobustInteractionDetector(
        debug=True, 
        save_debug_images=True, 
        debug_output_dir="simple_debug"
    )
    
    # Detect points
    points = detector.detect_interaction_points(
        image=rgb,
        mask=mask,
        depth_data=depth,
        max_points=5
    )
    
    print(f"Detected {len(points)} points")
    for i, point in enumerate(points):
        print(f"Point {i+1}: ({point.x}, {point.y}) - {point.interaction_type.value} - Score: {point.score:.3f}")
    
    return points

if __name__ == "__main__":
    test_debug_functionality()
    print("Debug test completed!")