#!/usr/bin/env python3
"""
Complete test script for debugging ZED to robot base coordinate transformations.
This script tests both the original and corrected transformation methods.
"""

import numpy as np
import json
import time
from scipy.spatial.transform import Rotation

# Optional matplotlib import
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib available - 3D visualization enabled")
except ImportError as e:
    print(f"Matplotlib not available ({e}) - 3D visualization disabled")
    print("Install matplotlib with: pip install matplotlib")

class TransformationTester:
    def __init__(self, calibration_file=None):
        """
        Initialize the transformation tester
        
        Args:
            calibration_file: Path to the calibration JSON file
        """
        self.original_transform = None
        self.corrected_transform = None
        self.transform_confidence = 0.0
        
        if calibration_file:
            self.load_calibration(calibration_file)
        else:
            # Use the provided calibration data as default
            self.load_default_calibration()
    
    def load_calibration(self, filename):
        """Load calibration from JSON file"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.original_transform = np.array(calibration_data['T_zed_to_robot_base'])
            self.transform_confidence = calibration_data.get('transform_confidence', 0.0)
            
            print(f"Loaded calibration from {filename}")
            print(f"Confidence: {self.transform_confidence:.3f}")
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            self.load_default_calibration()
    
    def load_default_calibration(self):
        """Load the default calibration data from the provided JSON"""
        print("Using default calibration data...")
        self.original_transform = np.array([
            [0.8447600156677891, -0.5067964174424076, -0.17186595706100946, -0.2961952424159702],
            [-0.37515873209570005, -0.3318151253534364, -0.8655372021577291, 0.44793084816922457],
            [0.38162342913047487, 0.7956482349972404, -0.4704332518907152, 0.6428095325377009],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.transform_confidence = 0.9
    
    def analyze_original_transform(self):
        """Analyze the original transformation matrix"""
        print("\n" + "="*60)
        print("ORIGINAL TRANSFORMATION ANALYSIS")
        print("="*60)
        
        print("Original transformation matrix:")
        print(self.original_transform)
        
        # Extract components
        rotation_matrix = self.original_transform[:3, :3]
        translation = self.original_transform[:3, 3]
        
        print(f"\nTranslation component: {translation}")
        print(f"Translation magnitude: {np.linalg.norm(translation):.3f} meters")
        
        # Convert rotation to more interpretable formats
        rotation = Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        quaternion = rotation.as_quat()
        
        print(f"Rotation (Euler XYZ degrees): {euler_angles}")
        print(f"Rotation (quaternion x,y,z,w): {quaternion}")
        
        # Check if matrix is valid
        det = np.linalg.det(rotation_matrix)
        print(f"Rotation matrix determinant: {det:.6f} (should be ~1.0)")
        
        # Check orthogonality
        should_be_identity = rotation_matrix @ rotation_matrix.T
        orthogonality_error = np.linalg.norm(should_be_identity - np.eye(3))
        print(f"Orthogonality error: {orthogonality_error:.6f} (should be ~0.0)")
    
    def create_corrected_transform(self):
        """Create the corrected transformation matrix"""
        print("\n" + "="*60)
        print("CREATING CORRECTED TRANSFORMATION")
        print("="*60)
        
        # The original matrix appears to transform FROM robot base TO ZED
        # We need the inverse to transform FROM ZED TO robot base
        self.corrected_transform = np.linalg.inv(self.original_transform)
        
        print("Corrected transformation matrix (ZED → robot base):")
        print(self.corrected_transform)
        
        # Extract components of corrected transform
        rotation_matrix = self.corrected_transform[:3, :3]
        translation = self.corrected_transform[:3, 3]
        
        print(f"\nCorrected translation component: {translation}")
        print(f"Corrected translation magnitude: {np.linalg.norm(translation):.3f} meters")
        
        # Convert rotation to more interpretable formats
        rotation = Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        quaternion = rotation.as_quat()
        
        print(f"Corrected rotation (Euler XYZ degrees): {euler_angles}")
        print(f"Corrected rotation (quaternion x,y,z,w): {quaternion}")
    
    def test_point_transformations(self):
        """Test transformations with various test points"""
        print("\n" + "="*60)
        print("POINT TRANSFORMATION TESTS")
        print("="*60)
        
        # Define test points in ZED camera frame
        test_points = {
            "Origin": [0, 0, 0],
            "Forward 50cm": [0, 0, 0.5],
            "Forward 1m": [0, 0, 1.0],
            "Right 30cm": [0.3, 0, 0.5],
            "Left 30cm": [-0.3, 0, 0.5],
            "Up 20cm": [0, -0.2, 0.5],  # Note: ZED Y-axis points down
            "Down 20cm": [0, 0.2, 0.5],
            "Your test point": [0.6, -0.1, 0.1]  # The point from your description
        }
        
        print("Testing point transformations:")
        print("ZED Frame → Robot Base Frame")
        print("-" * 40)
        
        for name, point in test_points.items():
            print(f"\n{name}: {point}")
            
            # Test with original transform
            original_result = self.transform_point_original(point)
            print(f"  Original method: {original_result}")
            
            # Test with corrected transform
            corrected_result = self.transform_point_corrected(point)
            print(f"  Corrected method: {corrected_result}")
            
            # Calculate difference
            if original_result is not None and corrected_result is not None:
                diff = np.array(corrected_result) - np.array(original_result)
                print(f"  Difference: {diff} (magnitude: {np.linalg.norm(diff):.3f}m)")
    
    def transform_point_original(self, point):
        """Transform point using the original method (potentially incorrect)"""
        try:
            pos_homogeneous = np.array([point[0], point[1], point[2], 1.0])
            transformed_homogeneous = self.original_transform @ pos_homogeneous
            return transformed_homogeneous[:3]
        except Exception as e:
            print(f"Error in original transform: {e}")
            return None
    
    def transform_point_corrected(self, point):
        """Transform point using the corrected method"""
        try:
            if self.corrected_transform is None:
                return None
            pos_homogeneous = np.array([point[0], point[1], point[2], 1.0])
            transformed_homogeneous = self.corrected_transform @ pos_homogeneous
            return transformed_homogeneous[:3]
        except Exception as e:
            print(f"Error in corrected transform: {e}")
            return None
    
    def test_round_trip_transformation(self):
        """Test round-trip transformation accuracy"""
        print("\n" + "="*60)
        print("ROUND-TRIP TRANSFORMATION TEST")
        print("="*60)
        
        test_points = [
            [0, 0, 0.5],     # 50cm forward
            [0.3, 0, 0.5],   # 30cm right, 50cm forward
            [-0.2, 0.1, 1.0] # 20cm left, 10cm down, 1m forward
        ]
        
        print("Testing round-trip accuracy (ZED → Robot → ZED):")
        
        for i, point in enumerate(test_points):
            print(f"\nTest point {i+1}: {point}")
            
            # Forward transformation
            robot_point = self.transform_point_corrected(point)
            if robot_point is None:
                continue
                
            print(f"  In robot frame: {robot_point}")
            
            # Reverse transformation
            reverse_homogeneous = self.original_transform @ np.array([robot_point[0], robot_point[1], robot_point[2], 1.0])
            reverse_point = reverse_homogeneous[:3]
            
            print(f"  Back to ZED frame: {reverse_point}")
            
            # Calculate error
            error = np.linalg.norm(np.array(reverse_point) - np.array(point))
            print(f"  Round-trip error: {error:.6f} meters")
            
            if error > 0.001:  # 1mm tolerance
                print(f"  ⚠️  WARNING: High round-trip error!")
            else:
                print(f"  ✅ Good round-trip accuracy")
    
    def visualize_coordinate_frames(self):
        """Create a 3D visualization of the coordinate frames"""
        print("\n" + "="*60)
        print("CREATING 3D VISUALIZATION")
        print("="*60)
        
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - creating text-based visualization instead")
            self.create_text_visualization()
            return
        
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Robot base frame (origin)
            ax.quiver(0, 0, 0, 0.2, 0, 0, color='red', arrow_length_ratio=0.1, label='Robot X')
            ax.quiver(0, 0, 0, 0, 0.2, 0, color='green', arrow_length_ratio=0.1, label='Robot Y')
            ax.quiver(0, 0, 0, 0, 0, 0.2, color='blue', arrow_length_ratio=0.1, label='Robot Z')
            
            # ZED camera frame
            if self.corrected_transform is not None:
                # ZED position in robot frame
                zed_pos = self.corrected_transform[:3, 3]
                zed_rot = self.corrected_transform[:3, :3]
                
                # ZED coordinate axes
                zed_x = zed_rot @ np.array([0.2, 0, 0])
                zed_y = zed_rot @ np.array([0, 0.2, 0])
                zed_z = zed_rot @ np.array([0, 0, 0.2])
                
                ax.quiver(zed_pos[0], zed_pos[1], zed_pos[2], 
                         zed_x[0], zed_x[1], zed_x[2], 
                         color='darkred', arrow_length_ratio=0.1, label='ZED X')
                ax.quiver(zed_pos[0], zed_pos[1], zed_pos[2], 
                         zed_y[0], zed_y[1], zed_y[2], 
                         color='darkgreen', arrow_length_ratio=0.1, label='ZED Y')
                ax.quiver(zed_pos[0], zed_pos[1], zed_pos[2], 
                         zed_z[0], zed_z[1], zed_z[2], 
                         color='darkblue', arrow_length_ratio=0.1, label='ZED Z')
                
                # Mark ZED position
                ax.scatter(zed_pos[0], zed_pos[1], zed_pos[2], 
                          color='purple', s=100, label='ZED Camera')
                
                print(f"ZED camera position in robot frame: {zed_pos}")
            
            # Test points
            test_points_zed = [
                [0, 0, 0.5],     # 50cm forward in ZED frame
                [0.3, 0, 0.5],   # 30cm right, 50cm forward
                [-0.3, 0, 0.5],  # 30cm left, 50cm forward
            ]
            
            for i, point_zed in enumerate(test_points_zed):
                point_robot = self.transform_point_corrected(point_zed)
                if point_robot is not None:
                    ax.scatter(point_robot[0], point_robot[1], point_robot[2], 
                              color='orange', s=50, alpha=0.7)
                    ax.text(point_robot[0], point_robot[1], point_robot[2], 
                           f'P{i+1}', fontsize=8)
            
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (meters)')
            ax.legend()
            ax.set_title('Robot Base and ZED Camera Coordinate Frames')
            
            # Set equal aspect ratio
            max_range = 0.5
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, max_range])
            
            plt.tight_layout()
            plt.savefig('coordinate_frames_visualization.png', dpi=150, bbox_inches='tight')
            print("Visualization saved as 'coordinate_frames_visualization.png'")
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Falling back to text-based visualization")
            self.create_text_visualization()
    
    def create_text_visualization(self):
        """Create a text-based visualization of coordinate frames"""
        print("\nTEXT-BASED COORDINATE FRAME VISUALIZATION")
        print("-" * 50)
        
        if self.corrected_transform is not None:
            zed_pos = self.corrected_transform[:3, 3]
            zed_rot = self.corrected_transform[:3, :3]
            
            print(f"Robot Base Frame: Origin at [0, 0, 0]")
            print(f"  X-axis: [1, 0, 0] (red)")
            print(f"  Y-axis: [0, 1, 0] (green)")
            print(f"  Z-axis: [0, 0, 1] (blue)")
            print()
            
            print(f"ZED Camera Frame: Origin at {zed_pos}")
            print(f"  X-axis direction: {zed_rot @ np.array([1, 0, 0])}")
            print(f"  Y-axis direction: {zed_rot @ np.array([0, 1, 0])}")
            print(f"  Z-axis direction: {zed_rot @ np.array([0, 0, 1])}")
            print()
            
            # Distance and orientation info
            distance = np.linalg.norm(zed_pos)
            print(f"ZED Camera Distance from Robot Base: {distance:.3f} meters")
            
            # Convert to Euler angles for easier interpretation
            rotation = Rotation.from_matrix(zed_rot)
            euler = rotation.as_euler('xyz', degrees=True)
            print(f"ZED Camera Orientation (Euler XYZ): {euler} degrees")
            
            print("\nTest Points Visualization:")
            print("(ZED Frame → Robot Frame)")
            
            test_points = [
                ([0, 0, 0.5], "50cm forward in ZED"),
                ([0.3, 0, 0.5], "30cm right, 50cm forward in ZED"),
                ([-0.3, 0, 0.5], "30cm left, 50cm forward in ZED"),
            ]
            
            for point_zed, description in test_points:
                point_robot = self.transform_point_corrected(point_zed)
                if point_robot is not None:
                    print(f"  {description}")
                    print(f"    ZED: {point_zed} → Robot: {point_robot}")
        else:
            print("No corrected transform available for visualization")
    
    def test_your_specific_case(self):
        """Test the specific case mentioned in your question"""
        print("\n" + "="*60)
        print("YOUR SPECIFIC TEST CASE")
        print("="*60)
        
        # Your point: "roughly 0.6, -0.1, 0.1 in the robot frame"
        # But after transform: "[-0.24416, -0.28108, 0.53593]"
        
        print("Testing your specific case:")
        print("Point that should be at [0.6, -0.1, 0.1] in robot frame")
        print("But transforms to [-0.24416, -0.28108, 0.53593]")
        
        # Let's work backwards - if the final result should be [0.6, -0.1, 0.1]
        # what should the ZED point be?
        desired_robot_point = [0.6, -0.1, 0.1]
        
        # Transform back to ZED frame using original transform
        desired_robot_homogeneous = np.array([desired_robot_point[0], desired_robot_point[1], desired_robot_point[2], 1.0])
        zed_point_homogeneous = self.original_transform @ desired_robot_homogeneous
        zed_point = zed_point_homogeneous[:3]
        
        print(f"\nIf robot point should be: {desired_robot_point}")
        print(f"Then ZED point should be: {zed_point}")
        
        # Now test both transformations
        print(f"\nUsing original transform:")
        original_result = self.transform_point_original(zed_point)
        print(f"  ZED {zed_point} → Robot {original_result}")
        
        print(f"\nUsing corrected transform:")
        corrected_result = self.transform_point_corrected(zed_point)
        print(f"  ZED {zed_point} → Robot {corrected_result}")
        
        # Test the point that's actually being transformed
        actual_zed_point = [0.6, -0.1, 0.1]  # Assuming this is in ZED frame
        print(f"\nIf ZED point is actually: {actual_zed_point}")
        
        original_result = self.transform_point_original(actual_zed_point)
        print(f"  Original transform result: {original_result}")
        
        corrected_result = self.transform_point_corrected(actual_zed_point)
        print(f"  Corrected transform result: {corrected_result}")
    
    def generate_corrected_calibration_file(self):
        """Generate a corrected calibration file"""
        print("\n" + "="*60)
        print("GENERATING CORRECTED CALIBRATION FILE")
        print("="*60)
        
        if self.corrected_transform is None:
            print("No corrected transform available")
            return
        
        # Create corrected calibration data
        corrected_calibration = {
            'T_zed_to_robot_base': self.corrected_transform.tolist(),
            'T_robot_base_to_zed': self.original_transform.tolist(),  # Keep original for reference
            'transform_confidence': self.transform_confidence,
            'timestamp': time.time(),
            'calibration_method': 'corrected_simplified_stereo_aruco',
            'notes': 'Corrected transformation matrix - inverted from original calibration'
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"corrected_zed_to_robot_calibration_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(corrected_calibration, f, indent=2)
        
        print(f"Corrected calibration saved to: {filename}")
        print("You can use this file in your robot interface")
    
    def run_all_tests(self):
        """Run all transformation tests"""
        print("TRANSFORMATION DEBUGGING TEST SUITE")
        print("=" * 80)
        
        self.analyze_original_transform()
        self.create_corrected_transform()
        self.test_point_transformations()
        self.test_round_trip_transformation()
        self.test_your_specific_case()
        self.generate_corrected_calibration_file()
        self.visualize_coordinate_frames()
        
        print("\n" + "="*80)
        print("TEST SUITE COMPLETE")
        print("="*80)
        print("\nSUMMARY:")
        print("1. Check the point transformation results above")
        print("2. Look for the corrected calibration file generated")
        print("3. Review the 3D visualization to understand coordinate frames")
        print("4. Use the corrected transform in your robot interface")
        print("\nRECOMMENDED FIXES:")
        print("1. Use the corrected transformation matrix (inverse of original)")
        print("2. Remove manual position adjustments in move_to_pose")
        print("3. Use homogeneous coordinates for transformations")
        print("4. Verify transformation direction in ArUco calibration code")


def main():
    """Main function to run the transformation tests"""
    print("ZED to Robot Base Transformation Debugging Tool")
    print("=" * 50)
    
    # Option to load custom calibration file
    calibration_file = input("Enter calibration file path (or press Enter to use default): ").strip()
    if not calibration_file:
        calibration_file = None
    
    # Create tester and run all tests
    tester = TransformationTester(calibration_file)
    tester.run_all_tests()
    
    # Interactive testing
    print("\nInteractive testing mode:")
    while True:
        try:
            user_input = input("\nEnter ZED point as 'x,y,z' (or 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                break
            
            # Parse input
            coords = [float(x.strip()) for x in user_input.split(',')]
            if len(coords) != 3:
                print("Please enter exactly 3 coordinates")
                continue
            
            print(f"\nTransforming ZED point: {coords}")
            
            original_result = tester.transform_point_original(coords)
            corrected_result = tester.transform_point_corrected(coords)
            
            print(f"Original method result:  {original_result}")
            print(f"Corrected method result: {corrected_result}")
            
            if original_result and corrected_result:
                diff = np.array(corrected_result) - np.array(original_result)
                print(f"Difference: {diff} (magnitude: {np.linalg.norm(diff):.3f}m)")
        
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
        except KeyboardInterrupt:
            break
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()