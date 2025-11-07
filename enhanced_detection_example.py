#!/usr/bin/env python3
"""
Example usage of the enhanced interaction point detection system.
Shows how to use the new depth clustering with visual feature guidance.
"""

import sys
import numpy as np
sys.path.append('cognitive_bt_framework/src')

from vision.interaction_point_detector import RobustInteractionDetector

def example_usage():
    """Example of how to use the enhanced detection system."""
    print("Enhanced Interaction Point Detection - Usage Example")
    print("=" * 55)
    
    # Initialize the detector
    detector = RobustInteractionDetector(debug=False)
    
    # Your RGB image, depth data, and object mask would come from your vision system
    # For example: image, depth, mask = your_vision_system.capture()
    
    print("\n1. STANDARD DEPTH PLANE DETECTION (current default)")
    print("   Best for: General object interaction, fast processing")
    print("   Code:")
    print("""   points = detector.detect_interaction_points(
       image=rgb_image,
       mask=object_mask, 
       depth_data=depth_image,
       depth_plane_only=True,  # Current default
       use_advanced_clustering=False
   )""")
    
    print("\n2. ENHANCED DEPTH CLUSTERING WITH VISUAL GUIDANCE (new)")
    print("   Best for: Complex objects, objects with multiple surface features")
    print("   Provides: Surface stability, planarity metrics, visual feature guidance")
    print("   Code:")
    print("""   points = detector.detect_interaction_points(
       image=rgb_image,
       mask=object_mask,
       depth_data=depth_image, 
       depth_plane_only=False,
       use_advanced_clustering=True,  # Enable new method
       max_points=15
   )""")
    
    print("\n3. WHAT THE ENHANCED METHOD PROVIDES:")
    print("   ✓ Depth clustering using DBSCAN to find surface regions")
    print("   ✓ Visual feature extraction (contours, edges, texture)")
    print("   ✓ Visual guidance weights for depth clustering")
    print("   ✓ Surface stability and planarity analysis")
    print("   ✓ Multiple interaction types per surface cluster")
    print("   ✓ Robust region-of-interest generation")
    
    print("\n4. KEY PARAMETERS:")
    print("   • cluster_eps: DBSCAN epsilon for depth clustering (default: 0.01)")
    print("   • min_cluster_size: Minimum points per cluster (default: 50)")
    print("   • use_visual_guidance: Use visual features to guide clustering (default: True)")
    
    print("\n5. OUTPUT ENHANCEMENTS:")
    print("   Each InteractionPoint now includes:")
    print("   • surface_area: Size of the surface region")
    print("   • surface_stability: How stable the surface is (0-1)")
    print("   • surface_planarity: How flat the surface is (0-1)")
    
    print("\nThe enhanced system finds clusters of similar depth values,")
    print("uses visual features like edges and contours to guide the search,")
    print("and generates robust regions of interest for manipulation tasks.")

if __name__ == "__main__":
    example_usage()