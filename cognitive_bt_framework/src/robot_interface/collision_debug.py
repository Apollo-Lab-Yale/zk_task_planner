import numpy as np
import time
import torch
from typing import List, Dict, Optional, Tuple, Union, Any
import open3d as o3d
from curobo.geom.types import Cuboid, WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState as CuroboJointState


class CollisionDebugger:
    """Debug and verification tools for collision objects in CuRobo motion planning"""
    
    def __init__(self, motion_planner):
        """
        Initialize collision debugger
        
        Args:
            motion_planner: CuRoboMotionPlanner instance
        """
        self.motion_planner = motion_planner
        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        
    def list_collision_objects(self) -> List[Dict]:
        """
        List all collision objects currently in the world
        
        Returns:
            List of dictionaries containing collision object information
        """
        objects_info = []
        
        with self.motion_planner.collision_object_lock:
            for i, obj in enumerate(self.motion_planner.collision_objects):
                obj_info = {
                    'index': i,
                    'name': obj.name,
                    'type': type(obj).__name__,
                    'position': None,
                    'orientation': None,
                    'dimensions': None,
                    'scale': None,
                    'file_path': None
                }
                
                if isinstance(obj, Cuboid):
                    # Handle both tensor and list formats
                    if hasattr(obj.pose, 'cpu'):
                        obj_info['position'] = obj.pose[:3].cpu().numpy().tolist()
                        obj_info['orientation'] = obj.pose[3:].cpu().numpy().tolist()
                    else:
                        obj_info['position'] = obj.pose[:3]
                        obj_info['orientation'] = obj.pose[3:]
                    
                    if hasattr(obj.dims, 'cpu'):
                        obj_info['dimensions'] = obj.dims.cpu().numpy().tolist()
                    else:
                        obj_info['dimensions'] = obj.dims
                elif isinstance(obj, Mesh):
                    obj_info['position'] = obj.position
                    obj_info['orientation'] = obj.orientation
                    obj_info['scale'] = obj.scale
                    obj_info['file_path'] = obj.file_path
                
                objects_info.append(obj_info)
        
        return objects_info
    
    def print_collision_objects(self):
        """Print detailed information about all collision objects"""
        objects = self.list_collision_objects()
        
        print(f"\n=== Collision Objects ({len(objects)} total) ===")
        
        if not objects:
            print("No collision objects found")
            return
        
        for obj in objects:
            print(f"\nObject {obj['index']}: {obj['name']} ({obj['type']})")
            print(f"  Position: {obj['position']}")
            print(f"  Orientation: {obj['orientation']}")
            
            if obj['dimensions']:
                print(f"  Dimensions: {obj['dimensions']}")
            if obj['scale']:
                print(f"  Scale: {obj['scale']}")
            if obj['file_path']:
                print(f"  File: {obj['file_path']}")
    
    def verify_world_config(self) -> bool:
        """
        Verify that the world configuration is properly set up
        
        Returns:
            bool: True if world config is valid, False otherwise
        """
        try:
            if not self.motion_planner.motion_gen:
                print("❌ Motion generator not initialized")
                return False
            
            if not self.motion_planner.motion_gen.world_collision:
                print("❌ World collision not initialized")
                return False
            
            # Check if world collision has objects
            world_objects = self.list_collision_objects()
            print(f"✅ World collision initialized with {len(world_objects)} objects")
            
            # Verify IK solver
            if not self.motion_planner.ik_solver:
                print("❌ IK solver not initialized")
                return False
            
            print("✅ IK solver initialized")
            return True
            
        except Exception as e:
            print(f"❌ World config verification failed: {str(e)}")
            return False
    
    def test_collision_detection(self, test_positions: List[List[float]], 
                                test_orientations: Optional[List[List[float]]] = None) -> Dict:
        """
        Test collision detection at specific positions
        
        Args:
            test_positions: List of [x, y, z] positions to test
            test_orientations: Optional list of [w, x, y, z] orientations to test
            
        Returns:
            Dictionary with collision test results
        """
        if not self.motion_planner.motion_gen or not self.motion_planner.motion_gen.world_collision:
            print("❌ Motion generator or world collision not available")
            return {}
        
        results = {
            'test_positions': test_positions,
            'collision_results': [],
            'summary': {}
        }
        
        print(f"\n=== Testing Collision Detection at {len(test_positions)} positions ===")
        
        for i, pos in enumerate(test_positions):
            # Create test pose
            if test_orientations and i < len(test_orientations):
                orientation = test_orientations[i]
            else:
                # Default top-down orientation
                orientation = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
            
            pose = pos + orientation
            curobo_pose = CuroboPose.from_list(pose, self.tensor_args)
            
            # Test collision
            try:
                # This is a simplified collision check - in practice you'd use the actual collision checker
                collision_detected = False
                
                # Check against each collision object
                for obj in self.motion_planner.collision_objects:
                    if isinstance(obj, Cuboid):
                        # Simple distance check for cuboids
                        obj_center = obj.pose[:3]
                        if hasattr(obj_center, 'cpu'):
                            obj_center = obj_center.cpu().numpy()
                        else:
                            obj_center = np.array(obj_center)
                        
                        # Convert curobo pose to numpy
                        test_position = curobo_pose.position.cpu().numpy()
                        distance = np.linalg.norm(test_position - obj_center)
                        
                        # Get object dimensions
                        obj_dims = obj.dims
                        if hasattr(obj_dims, 'cpu'):
                            obj_dims = obj_dims.cpu().numpy()
                        else:
                            obj_dims = np.array(obj_dims)
                        
                        obj_radius = np.linalg.norm(obj_dims) / 2
                        
                        if distance < obj_radius:
                            collision_detected = True
                            break
                
                result = {
                    'position': pos,
                    'orientation': orientation,
                    'collision_detected': collision_detected
                }
                
                results['collision_results'].append(result)
                
                status = "❌ COLLISION" if collision_detected else "✅ FREE"
                print(f"Position {i}: {pos} -> {status}")
                
            except Exception as e:
                print(f"❌ Error testing position {i}: {str(e)}")
                results['collision_results'].append({
                    'position': pos,
                    'orientation': orientation,
                    'collision_detected': None,
                    'error': str(e)
                })
        
        # Summary
        collision_count = sum(1 for r in results['collision_results'] if r.get('collision_detected', False))
        results['summary'] = {
            'total_positions': len(test_positions),
            'collision_positions': collision_count,
            'free_positions': len(test_positions) - collision_count
        }
        
        print(f"\nSummary: {collision_count}/{len(test_positions)} positions have collisions")
        return results
    
    def visualize_collision_objects(self, save_path: Optional[str] = None):
        """
        Create a 3D visualization of collision objects using Open3D
        
        Args:
            save_path: Optional path to save the visualization as image
        """
        try:
            import open3d as o3d
            
            # Create coordinate frame for reference
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            geometries = [coordinate_frame]
            
            objects = self.list_collision_objects()
            
            print(f"\n=== Creating 3D Visualization of {len(objects)} Collision Objects ===")
            
            for obj in objects:
                if obj['type'] == 'Cuboid':
                    # Create cuboid mesh
                    box = o3d.geometry.TriangleMesh.create_box(
                        width=obj['dimensions'][0],
                        height=obj['dimensions'][1], 
                        depth=obj['dimensions'][2]
                    )
                    
                    # Apply transformation
                    position = np.array(obj['position'])
                    orientation = np.array(obj['orientation'])
                    
                    # Convert quaternion to rotation matrix
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
                    rotation_matrix = r.as_matrix()
                    
                    # Create transformation matrix
                    transform = np.eye(4)
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = position
                    
                    box.transform(transform)
                    box.paint_uniform_color([0.8, 0.2, 0.2])  # Red for collision objects
                    
                    geometries.append(box)
                    
                    print(f"Added cuboid: {obj['name']} at {position}")
                
                elif obj['type'] == 'Mesh':
                    try:
                        # Load mesh file
                        mesh = o3d.io.read_triangle_mesh(obj['file_path'])
                        
                        # Apply transformation
                        position = np.array(obj['position'])
                        orientation = np.array(obj['orientation'])
                        scale = np.array(obj['scale'])
                        
                        # Create transformation matrix
                        from scipy.spatial.transform import Rotation as R
                        r = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
                        rotation_matrix = r.as_matrix()
                        
                        transform = np.eye(4)
                        transform[:3, :3] = rotation_matrix * scale
                        transform[:3, 3] = position
                        
                        mesh.transform(transform)
                        mesh.paint_uniform_color([0.2, 0.8, 0.2])  # Green for mesh objects
                        
                        geometries.append(mesh)
                        print(f"Added mesh: {obj['name']} from {obj['file_path']}")
                        
                    except Exception as e:
                        print(f"❌ Failed to load mesh {obj['name']}: {str(e)}")
            
            # Create visualization
            o3d.visualization.draw_geometries(geometries, window_name="Collision Objects")
            
            if save_path:
                # Save screenshot
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                for geom in geometries:
                    vis.add_geometry(geom)
                vis.run()
                vis.capture_screen_image(save_path)
                vis.destroy_window()
                print(f"Saved visualization to: {save_path}")
                
        except ImportError:
            print("❌ Open3D not available for visualization")
        except Exception as e:
            print(f"❌ Visualization failed: {str(e)}")
    
    def test_motion_planning_with_collisions(self, start_pos: List[float], 
                                           target_pos: List[float],
                                           start_orientation: Optional[List[float]] = None,
                                           target_orientation: Optional[List[float]] = None) -> Dict:
        """
        Test motion planning with collision objects present
        
        Args:
            start_pos: [x, y, z] start position
            target_pos: [x, y, z] target position
            start_orientation: Optional [w, x, y, z] start orientation
            target_orientation: Optional [w, x, y, z] target orientation
            
        Returns:
            Dictionary with planning results
        """
        print(f"\n=== Testing Motion Planning with Collisions ===")
        print(f"Start: {start_pos}")
        print(f"Target: {target_pos}")
        
        # Verify world config first
        if not self.verify_world_config():
            return {'success': False, 'error': 'World config not valid'}
        
        try:
            # Use default orientations if not provided
            if start_orientation is None:
                start_orientation = [1.0, 0.0, 0.0, 0.0]  # Top-down
            if target_orientation is None:
                target_orientation = [1.0, 0.0, 0.0, 0.0]  # Top-down
            
            # Test motion planning
            success, trajectory, status = self.motion_planner.plan_cartesian_path(
                start_position=start_pos,
                start_orientation=start_orientation,
                target_position=target_pos,
                target_orientation=target_orientation,
                planning_timeout=10.0,
                execute=False,
                speed_factor=0.5
            )
            
            result = {
                'success': success,
                'status': status,
                'trajectory_length': len(trajectory) if trajectory is not None else 0,
                'start_pos': start_pos,
                'target_pos': target_pos,
                'start_orientation': start_orientation,
                'target_orientation': target_orientation
            }
            
            if success:
                print(f"✅ Motion planning successful")
                print(f"   Trajectory length: {len(trajectory)} waypoints")
                print(f"   Status: {status}")
            else:
                print(f"❌ Motion planning failed")
                print(f"   Status: {status}")
            
            return result
            
        except Exception as e:
            print(f"❌ Motion planning test failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def benchmark_collision_performance(self, num_tests: int = 100) -> Dict:
        """
        Benchmark collision detection performance
        
        Args:
            num_tests: Number of random positions to test
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"\n=== Benchmarking Collision Detection ({num_tests} tests) ===")
        
        if not self.motion_planner.motion_gen or not self.motion_planner.motion_gen.world_collision:
            return {'error': 'Motion generator not available'}
        
        # Generate random test positions
        np.random.seed(42)  # For reproducible results
        test_positions = []
        
        # Generate positions in a reasonable workspace
        for _ in range(num_tests):
            pos = [
                np.random.uniform(-0.5, 0.5),  # X
                np.random.uniform(-0.5, 0.5),  # Y  
                np.random.uniform(0.0, 0.8)    # Z
            ]
            test_positions.append(pos)
        
        # Time the collision detection
        start_time = time.time()
        
        collision_results = []
        for pos in test_positions:
            # Create test pose with top-down orientation
            pose = pos + [1.0, 0.0, 0.0, 0.0]
            curobo_pose = CuroboPose.from_list(pose, self.tensor_args)
            
            # Simple collision check (this would be replaced with actual collision detection)
            collision_detected = False
            for obj in self.motion_planner.collision_objects:
                if isinstance(obj, Cuboid):
                    obj_center = obj.pose[:3]
                    if hasattr(obj_center, 'cpu'):
                        obj_center = obj_center.cpu().numpy()
                    else:
                        obj_center = np.array(obj_center)
                    
                    # Convert curobo pose to numpy
                    test_position = curobo_pose.position.cpu().numpy()
                    distance = np.linalg.norm(test_position - obj_center)
                    
                    # Get object dimensions
                    obj_dims = obj.dims
                    if hasattr(obj_dims, 'cpu'):
                        obj_dims = obj_dims.cpu().numpy()
                    else:
                        obj_dims = np.array(obj_dims)
                    
                    obj_radius = np.linalg.norm(obj_dims) / 2
                    if distance < obj_radius:
                        collision_detected = True
                        break
            
            collision_results.append(collision_detected)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        collision_count = sum(collision_results)
        avg_time_per_test = total_time / num_tests
        
        results = {
            'total_tests': num_tests,
            'collision_count': collision_count,
            'free_count': num_tests - collision_count,
            'collision_rate': collision_count / num_tests,
            'total_time': total_time,
            'avg_time_per_test': avg_time_per_test,
            'tests_per_second': num_tests / total_time
        }
        
        print(f"Results:")
        print(f"  Total tests: {num_tests}")
        print(f"  Collisions: {collision_count} ({collision_count/num_tests*100:.1f}%)")
        print(f"  Free space: {num_tests - collision_count} ({(num_tests-collision_count)/num_tests*100:.1f}%)")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per test: {avg_time_per_test*1000:.2f}ms")
        print(f"  Tests per second: {num_tests/total_time:.1f}")
        
        return results


def create_test_collision_objects(motion_planner):
    """
    Create test collision objects for debugging
    
    Args:
        motion_planner: CuRoboMotionPlanner instance
    """
    print("\n=== Creating Test Collision Objects ===")
    
    # Clear existing objects
    motion_planner.clear_collision_objects()
    
    # Add a table surface
    motion_planner.add_collision_object(
        name="table",
        dimensions=[1.0, 0.6, 0.02],  # 1m x 0.6m x 2cm
        position=[0.0, 0.0, -0.01],   # Slightly below origin
        orientation=[1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    )
    
    # Add a wall behind the robot
    motion_planner.add_collision_object(
        name="back_wall", 
        dimensions=[0.1, 0.6, 0.8],   # 10cm thick, 0.6m wide, 0.8m tall
        position=[-0.4, 0.0, 0.4],    # Behind the robot
        orientation=[1.0, 0.0, 0.0, 0.0]
    )
    
    # Add a small obstacle
    motion_planner.add_collision_object(
        name="obstacle",
        dimensions=[0.1, 0.1, 0.2],   # 10cm cube
        position=[0.2, 0.1, 0.1],     # In front of robot
        orientation=[1.0, 0.0, 0.0, 0.0]
    )
    
    print("✅ Created 3 test collision objects: table, back_wall, obstacle")


def run_collision_debug_tests(motion_planner):
    """
    Run comprehensive collision debugging tests
    
    Args:
        motion_planner: CuRoboMotionPlanner instance
    """
    print("\n" + "="*60)
    print("COLLISION DEBUGGING TESTS")
    print("="*60)
    
    # Create debugger
    debugger = CollisionDebugger(motion_planner)
    
    # 1. List and print collision objects
    debugger.print_collision_objects()
    
    # 2. Verify world configuration
    debugger.verify_world_config()
    
    # 3. Test collision detection at specific points
    test_positions = [
        [0.0, 0.0, 0.1],   # Above table
        [0.2, 0.1, 0.1],   # Inside obstacle
        [0.5, 0.0, 0.1],   # Far from obstacles
        [-0.4, 0.0, 0.4],  # Inside back wall
    ]
    
    debugger.test_collision_detection(test_positions)
    
    # 4. Test motion planning with collisions
    debugger.test_motion_planning_with_collisions(
        start_pos=[0.0, 0.0, 0.3],
        target_pos=[0.4, 0.0, 0.3]
    )
    
    # 5. Benchmark performance
    debugger.benchmark_collision_performance(num_tests=50)
    
    # 6. Create visualization (if Open3D available)
    try:
        debugger.visualize_collision_objects()
    except Exception as e:
        print(f"Visualization not available: {str(e)}")
    
    print("\n" + "="*60)
    print("COLLISION DEBUGGING TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Example usage
    print("Collision Debug Module")
    print("Use this module with a CuRoboMotionPlanner instance:")
    print("  debugger = CollisionDebugger(motion_planner)")
    print("  debugger.print_collision_objects()")
    print("  run_collision_debug_tests(motion_planner)") 