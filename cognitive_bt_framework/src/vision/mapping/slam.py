from cognitive_bt_framework.src.vision.realsense import Camera
import open3d as o3d
import numpy as np
import copy
from typing import List, Optional, Tuple
import time
import threading
import queue


class RealSenseSLAM:
    def __init__(self, camera: Camera):
        self.camera = camera
        # Updated camera intrinsics - these should match your RealSense camera model
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            camera.width,
            camera.height,
            camera.intrinsics.fx,  # Use actual camera intrinsics
            camera.intrinsics.fy,
            camera.intrinsics.ppx,
            camera.intrinsics.ppy
        )

        # Refined parameters for better reconstruction
        self.voxel_size = 0.02  # Reduced voxel size for better detail
        self.max_correspondence_distance = 0.05  # Reduced for more precise matching
        self.icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        # ICP convergence criteria
        self.icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50
        )

        # Tracking state
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
        self.global_transform = np.identity(4)
        self.previous_rgb: Optional[o3d.geometry.Image] = None
        self.previous_depth: Optional[o3d.geometry.Image] = None
        self.previous_pcd: Optional[o3d.geometry.PointCloud] = None

        # Keyframe selection parameters
        self.min_transform_distance = 0.05  # Minimum translation between keyframes
        self.min_rotation_angle = 5  # Minimum rotation angle in degrees

        # Reconstruction data
        self.fragments: List[o3d.geometry.PointCloud] = []
        self.max_fragments = 30

        # Processing queue and thread
        self.frame_queue = queue.Queue(maxsize=5)
        self.processing_thread = None
        self.running = False

        # Visualization update control
        self.last_vis_update = time.time()
        self.vis_update_interval = 0.5  # Increased update frequency

        # Frame processing control
        self.frame_count = 0
        self.process_every_n_frames = 3

    def start(self):
        """Start the SLAM processing thread."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def stop(self):
        """Stop the SLAM processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()

    def _processing_loop(self):
        """Background processing loop."""
        while self.running:
            try:
                frames = self.camera.get_frames()
                if not frames:
                    time.sleep(0.01)
                    continue

                # Skip frames for performance
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    continue

                if not self.frame_queue.full():
                    self.frame_queue.put(frames)
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Preprocess point cloud for better registration."""
        # Convert CUDA tensor to CPU if necessary
        if hasattr(pcd, 'cpu'):
            pcd = pcd.cpu()

        # Basic filtering using voxel downsampling
        pcd = pcd.voxel_down_sample(self.voxel_size)

        # Filter out points with zero or invalid values
        points = np.asarray(pcd.points)
        mask = np.all(np.isfinite(points), axis=1)
        pcd.points = o3d.utility.Vector3dVector(points[mask])

        if len(np.asarray(pcd.colors)) > 0:
            colors = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        # Estimate normals with optimized parameters
        try:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
        except Exception as e:
            print(f"Warning: Normal estimation failed: {e}")

        return pcd

    def is_keyframe(self, transform: np.ndarray) -> bool:
        """Determine if the current frame should be a keyframe."""
        translation = np.linalg.norm(transform[:3, 3])
        rotation = np.arccos((np.trace(transform[:3, :3]) - 1) / 2) * 180 / np.pi

        return translation > self.min_transform_distance or rotation > self.min_rotation_angle

    def process_frame(self) -> bool:
        """Process a single frame from the queue with improved registration."""
        try:
            frames = self.frame_queue.get_nowait()
        except queue.Empty:
            return False

        color_image, depth_image = frames

        try:
            # Convert to Open3D format
            rgb = o3d.geometry.Image(color_image)
            depth = o3d.geometry.Image(depth_image)

            # Create RGBD image with adjusted parameters
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1.0 / self.camera.depth_scale,
                depth_trunc=5.0,  # Increased depth truncation
                convert_rgb_to_intensity=False
            )

            # Create and preprocess point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.pinhole_camera_intrinsic
            )
            pcd = self.preprocess_point_cloud(pcd)

            if self.previous_pcd is None:
                self.previous_rgb = rgb
                self.previous_depth = depth
                self.previous_pcd = pcd
                return True

            # Initial alignment using RGB-D odometry
            success, trans_init, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd,
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    self.previous_rgb,
                    self.previous_depth,
                    depth_scale=1.0 / self.camera.depth_scale,
                    depth_trunc=5.0,
                    convert_rgb_to_intensity=False
                ),
                self.pinhole_camera_intrinsic,
                np.identity(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                o3d.pipelines.odometry.OdometryOption()
            )

            if not success:
                print("Failed to compute initial alignment")
                return False

            # Refine alignment using ICP
            registration = o3d.pipelines.registration.registration_icp(
                pcd, self.previous_pcd,
                self.max_correspondence_distance,
                trans_init,
                self.icp_method,
                self.icp_criteria
            )

            if registration.fitness < 0.3:  # Minimum overlap threshold
                print(f"Poor registration fitness: {registration.fitness}")
                return False

            trans = registration.transformation

            # Update global transform
            self.global_transform = np.dot(self.global_transform, trans)

            # Only add keyframes to pose graph
            if self.is_keyframe(trans):
                # Add to pose graph
                self.pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(self.global_transform)
                )

                # Add edge between consecutive nodes
                self.pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        len(self.pose_graph.nodes) - 2,
                        len(self.pose_graph.nodes) - 1,
                        trans,
                        np.identity(6),
                        uncertain=False
                    )
                )

                # Add to reconstruction
                transformed_pcd = copy.deepcopy(pcd)
                transformed_pcd.transform(self.global_transform)
                self.fragments.append(transformed_pcd)

                # Limit the number of stored fragments
                if len(self.fragments) > self.max_fragments:
                    self.fragments.pop(0)

            # Store current frame as previous
            self.previous_rgb = rgb
            self.previous_depth = depth
            self.previous_pcd = pcd

            return True

        except Exception as e:
            print(f"Error processing frame: {e}")
            return False

    def get_current_map(self) -> o3d.geometry.PointCloud:
        """Get the current accumulated point cloud map with improved filtering."""
        if not self.fragments:
            return o3d.geometry.PointCloud()

        try:
            # Combine all fragments
            combined_pcd = o3d.geometry.PointCloud()
            for fragment in self.fragments:
                if hasattr(fragment, 'cpu'):
                    fragment = fragment.cpu()
                combined_pcd += fragment

            # Final cleanup of the combined map
            combined_pcd = combined_pcd.voxel_down_sample(self.voxel_size)

            # Filter out points with zero or invalid values
            points = np.asarray(combined_pcd.points)
            mask = np.all(np.isfinite(points), axis=1)
            combined_pcd.points = o3d.utility.Vector3dVector(points[mask])

            if len(np.asarray(combined_pcd.colors)) > 0:
                colors = np.asarray(combined_pcd.colors)
                combined_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

            return combined_pcd
        except Exception as e:
            print(f"Error creating map: {e}")
            return o3d.geometry.PointCloud()

    def should_update_visualization(self) -> bool:
        """Check if visualization should be updated based on time interval."""
        current_time = time.time()
        if current_time - self.last_vis_update >= self.vis_update_interval:
            self.last_vis_update = current_time
            return True
        return False


if __name__ == "__main__":
    # Initialize camera
    camera = Camera()
    if not camera.start():
        print("Failed to start camera")
        exit(1)

    # Initialize SLAM
    slam = RealSenseSLAM(camera)
    slam.start()

    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window("SLAM Map", width=1280, height=720)

    try:
        while True:
            if slam.process_frame() and slam.should_update_visualization():
                # Update visualization less frequently
                vis.clear_geometries()
                current_map = slam.get_current_map()
                vis.add_geometry(current_map)
                vis.poll_events()
                vis.update_renderer()

            # Brief sleep to prevent CPU overload
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSLAM interrupted by user")

    finally:
        # Clean up
        slam.stop()
        vis.destroy_window()
        camera.stop()

        # Save final point cloud
        try:
            final_map = slam.get_current_map()
            o3d.io.write_point_cloud("slam_map.pcd", final_map)
            print("Final map saved as 'slam_map.pcd'")
        except Exception as e:
            print(f"Error saving map: {e}")