import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
import os
import sys
from scipy.spatial.transform import Rotation as R

from cognitive_bt_framework.src.thread_safe_viz import ThreadSafeVisualizer


XARM_7_PATH = os.environ.get('CUROBO_XARM7_URDF', '')

class PyBulletVisualizer(ThreadSafeVisualizer):
    def __init__(self, robot_urdf_path, camera_params=None):
        """
        Initialize PyBullet visualizer for robot, depth image, and object visualization
        with dynamic camera extrinsics based on robot joint states
        
        Args:
            robot_urdf_path: Path to the robot URDF file
            camera_params: Camera intrinsic parameters
        """
        # Initialize PyBullet in GUI mode
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
        p.setGravity(0, 0, -9.8)
        
        # Import for glob
        import glob
        
        # Create ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot URDF
        print(f"Loading robot from: {robot_urdf_path}")
        self.robot_id = p.loadURDF(
            robot_urdf_path, 
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
        )
        
        # Get number of joints and joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"Robot loaded with {self.num_joints} joints")
        
        # Store joint names and info
        self.joint_info = {}
        self.joint_name_to_id = {}
        self.link_name_to_id = {}
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            self.joint_info[i] = info
            self.joint_name_to_id[joint_name] = i
            self.link_name_to_id[link_name] = i
            print(f"Joint {i}: {joint_name}, Link: {link_name}")
        
        # Find important camera link IDs
        self.camera_depth_optical_frame_id = self.link_name_to_id.get('camera_depth_optical_frame')
        self.camera_color_optical_frame_id = self.link_name_to_id.get('camera_color_optical_frame')
        
        # Initialize thread-safe command queue
        self.initialize_command_queue()

        # Initialize joint_indices attribute - critical for thread-safe robot updates
        self.joint_indices = []
        for i in range(self.num_joints):
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED:
                self.joint_indices.append(i)
        print(f"Found {len(self.joint_indices)} movable joints: {self.joint_indices}")
        
        if not self.camera_depth_optical_frame_id:
            print("Warning: camera_depth_optical_frame not found in URDF")
        if not self.camera_color_optical_frame_id:
            print("Warning: camera_color_optical_frame not found in URDF")
        
        # Set camera parameters (intrinsics)
        self.camera_params = camera_params or {
            'fx': 429.92523193359375, 
            'fy': 429.92523193359375,
            'cx': 431.7160339355469,
            'cy': 233.39739990234375,
            'width': 640,
            'height': 480,
            'depth_scale': 0.001,  # Convert depth values to meters
            'color_format': 'RGB'  # RGB or BGR
        }
        
        # Initialize camera extrinsics (will be updated based on joint states)
        self.camera_to_base_transform = np.eye(4)
        
        # Dictionary to store object visualizations
        self.object_markers = {}
        
        # Point cloud visualization
        self.point_cloud_ids = []
        
        # RGB and depth visualization
        self.rgb_texture_id = None
        self.depth_texture_id = None
        self.rgb_depth_overlay_id = None
        self.current_rgb_image = None
        self.current_depth_image = None
        
        # Create a unique ID for this instance
        self.instance_id = f"{id(self)}_{int(time.time())}"
        
        # Setup temporary directory
        self.temp_dir = os.path.join(os.path.expanduser("~"), ".pybullet_textures")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Add a coordinate system at the robot base for reference
        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], lineWidth=2)  # X-axis (red)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], lineWidth=2)  # Y-axis (green)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], lineWidth=2)  # Z-axis (blue)
        
        # Initialize debug items list
        self.camera_debug_items = []
        
        # Create visualization plane for RGB and depth overlay
        self.create_visualization_plane()
        
        
        print("PyBullet visualizer initialized")
    
    def transform_points_to_base_frame(self, point_cloud):
        """
        Transform point cloud from camera frame to robot base frame
        
        Args:
            point_cloud: Nx3 array of 3D points in camera frame
            
        Returns:
            transformed_cloud: Nx3 array of 3D points in robot base frame
        """
        # Create homogeneous coordinates
        homogeneous_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        
        # Transform to base frame
        transformed_cloud = homogeneous_cloud @ self.camera_to_base_transform.T
        
        # Return 3D coordinates
        return transformed_cloud[:, :3]
    
    def create_visualization_plane(self):
        """Create visualization planes for RGB, depth, and overlay images"""
        # Create a visualization quad in 3D space to display RGB and depth images
        # This will be positioned in the world and oriented to face the camera
        self.visual_shape_rgb = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.225, 0.001],  # Aspect ratio of standard camera
            rgbaColor=[1, 1, 1, 0.9]  # More opaque for better visibility
        )
        
        self.visual_shape_depth = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.225, 0.001],  # Same size as RGB
            rgbaColor=[1, 1, 1, 0.9]
        )
        
        self.visual_shape_overlay = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.225, 0.001],  # Same size as RGB
            rgbaColor=[1, 1, 1, 0.9]
        )
        
        # Position the planes in front of the robot with more separation
        self.rgb_plane = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.visual_shape_rgb,
            basePosition=[0.8, -0.7, 0.8],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.depth_plane = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.visual_shape_depth,
            basePosition=[0.8, 0.0, 0.8],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.overlay_plane = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.visual_shape_overlay,
            basePosition=[0.8, 0.7, 0.8],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Add larger, more visible labels for each visualization
        text_color = [1, 1, 1]
        text_size = 2.0
        
        self.label_rgb = p.addUserDebugText("RGB Image", [0.8, -0.7, 1.1], text_color, textSize=text_size)
        self.label_depth = p.addUserDebugText("Depth Image", [0.8, 0.0, 1.1], text_color, textSize=text_size)
        self.label_overlay = p.addUserDebugText("RGB-Depth Overlay", [0.8, 0.7, 1.1], text_color, textSize=text_size)
        
    def close(self):
        """Close the PyBullet visualizer and clean up resources"""
        # Clean up temporary files
        try:
            import glob
            temp_pattern = os.path.join(self.temp_dir, f"temp_*_{self.instance_id}*.png")
            for temp_file in glob.glob(temp_pattern):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Error removing temporary file {temp_file}: {str(e)}")
        except Exception as e:
            print(f"Error cleaning up temporary files: {str(e)}")
            
        # Disconnect from PyBullet
        p.disconnect(self.physics_client)
        print("PyBullet visualizer closed")
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")
    
    def update_robot_configuration(self, joint_positions):
        """
        Update the robot configuration in PyBullet based on joint positions
        
        Args:
            joint_positions: List/array of joint positions
        """
        # Ensure the number of joint positions matches the expected number
        if len(joint_positions) > self.num_joints:
            print(f"Warning: Received {len(joint_positions)} joint positions, but robot has {self.num_joints} joints")
            joint_positions = joint_positions[:self.num_joints]
        
        # Set each joint position
        for i in range(min(len(joint_positions), self.num_joints)):
            # Skip fixed joints (they can't be moved)
            if self.joint_info[i][2] != p.JOINT_FIXED:
                p.resetJointState(self.robot_id, i, joint_positions[i])
        
        # Update camera extrinsics after changing joint positions
        self.update_camera_extrinsics()
    
    def update_camera_extrinsics(self):
        """Calculate camera extrinsics based on current robot configuration with improved visualization"""
        if self.camera_depth_optical_frame_id is None:
            # Try to find the camera link by searching for common camera link names
            for link_name in self.link_name_to_id:
                if 'camera' in link_name.lower() and ('optical' in link_name.lower() or 'depth' in link_name.lower()):
                    self.camera_depth_optical_frame_id = self.link_name_to_id[link_name]
                    print(f"Found camera link: {link_name}, ID: {self.camera_depth_optical_frame_id}")
                    break
            
            if self.camera_depth_optical_frame_id is None:
                print("Warning: Cannot update camera extrinsics without camera_depth_optical_frame_id")
                # Try the end effector link as fallback
                for link_name in self.link_name_to_id:
                    if 'tool0' in link_name.lower() or 'tcp' in link_name.lower() or 'ee' in link_name.lower():
                        self.camera_depth_optical_frame_id = self.link_name_to_id[link_name]
                        print(f"Using end effector link as fallback: {link_name}")
                        break
                
                if self.camera_depth_optical_frame_id is None:
                    print("Could not find any suitable camera or end effector link")
                    return
        
        # Get the camera link state (position and orientation in world frame)
        link_state = p.getLinkState(self.robot_id, self.camera_depth_optical_frame_id, computeForwardKinematics=True)
        camera_pos = link_state[0]  # Position
        camera_orn = link_state[1]  # Orientation (quaternion)
        
        # Convert quaternion to rotation matrix
        rot = R.from_quat([camera_orn[0], camera_orn[1], camera_orn[2], camera_orn[3]])
        rot_matrix = rot.as_matrix()
        
        # Create 4x4 transformation matrix (world_T_camera)
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = camera_pos
        
        # Store the transformation (base_T_camera)
        self.camera_to_base_transform = transform
        
        # Visualize camera frame with improved visibility
        self.visualize_camera_frame(camera_pos, camera_orn)
    
    def visualize_camera_frame(self, position, orientation, scale=0.1):
        """Visualize the camera coordinate frame with improved visibility"""
        # Create rotation matrix from quaternion
        rot = R.from_quat([orientation[0], orientation[1], orientation[2], orientation[3]])
        rot_matrix = rot.as_matrix()
        
        # Draw coordinate axes - increase scale for better visibility
        origin = np.array(position)
        x_axis = origin + scale * rot_matrix[:, 0]
        y_axis = origin + scale * rot_matrix[:, 1]
        z_axis = origin + scale * rot_matrix[:, 2]
        
        # Remove old debug lines if they exist
        for item in self.camera_debug_items:
            p.removeUserDebugItem(item)
        
        # Draw new coordinate axes with thicker lines
        self.camera_debug_items = [
            p.addUserDebugLine(position, x_axis, [1, 0, 0], lineWidth=3),  # X-axis (red)
            p.addUserDebugLine(position, y_axis, [0, 1, 0], lineWidth=3),  # Y-axis (green)
            p.addUserDebugLine(position, z_axis, [0, 0, 1], lineWidth=3),  # Z-axis (blue)
            p.addUserDebugText("Camera", position, [1, 1, 1], textSize=1.5)
        ]
        
        # Add a visual camera frustum to better visualize the camera's field of view
        frustum_width = 0.08
        frustum_height = 0.06
        frustum_depth = 0.1
        
        # Create frustum corners in camera frame
        frustum_corners = [
            # Near plane corners
            [0, 0, 0],  # Camera origin
            [frustum_width/2, frustum_height/2, frustum_depth],
            [frustum_width/2, -frustum_height/2, frustum_depth],
            [-frustum_width/2, -frustum_height/2, frustum_depth],
            [-frustum_width/2, frustum_height/2, frustum_depth]
        ]
        
        # Transform corners to world frame
        world_corners = []
        for corner in frustum_corners:
            # Transform point from camera frame to world frame
            point = rot_matrix @ np.array(corner) + origin
            world_corners.append(point)
        
        # Draw frustum lines
        camera_origin = world_corners[0]
        frustum_color = [0.8, 0.8, 0]  # Yellow
        
        # Draw lines from origin to corners of frustum
        for i in range(1, 5):
            self.camera_debug_items.append(
                p.addUserDebugLine(camera_origin, world_corners[i], frustum_color, lineWidth=2)
            )
        
        # Draw the frustum near plane
        for i in range(1, 5):
            self.camera_debug_items.append(
                p.addUserDebugLine(world_corners[i], world_corners[1 + (i % 4)], frustum_color, lineWidth=2)
            )
        
        # Add a label for the camera
        self.camera_debug_items.append(
            p.addUserDebugText("Camera View", 
                            (world_corners[1] + world_corners[3]) / 2,  # Center of frustum
                            textColorRGB=[1, 1, 0],
                            textSize=1.2)
        )
    
    def project_depth_to_point_cloud(self, depth_image, mask=None):
        """
        Project depth image to 3D point cloud in camera frame with improved filtering
        
        Args:
            depth_image: Depth image (HxW) with values in camera units
            mask: Optional binary mask to filter points
            
        Returns:
            point_cloud: Nx3 array of 3D points in camera frame
        """
        # Scale depth to meters if needed
        depth_scale = self.camera_params.get('depth_scale', 0.001)
        depth_meters = depth_image.astype(np.float32) * depth_scale
        
        # Get camera parameters
        fx = self.camera_params['fx']
        fy = self.camera_params['fy']
        cx = self.camera_params['cx']
        cy = self.camera_params['cy']
        
        # Create meshgrid for pixel coordinates
        height, width = depth_meters.shape
        v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Convert pixel coordinates to normalized device coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy
        
        # Create 3D points in camera frame (z forward, x right, y down for camera optical frame)
        z = depth_meters
        x = x * z
        y = y * z
        
        # Stack and reshape to Nx3 array
        point_cloud = np.stack((x, y, z), axis=-1)
        point_cloud = point_cloud.reshape(-1, 3)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)
            point_cloud = point_cloud[mask_flat]
        
        # Filter out invalid points - use much less aggressive filtering
        # Lower min threshold to 0.001 (1mm) and higher max to 5.0 (5m)
        valid_mask = (point_cloud[:, 2] > 0.001) & (point_cloud[:, 2] < 5.0)
        point_cloud = point_cloud[valid_mask]
        
        return point_cloud
    
    def transform_point_cloud_to_base(self, point_cloud):
        """
        Transform point cloud from camera frame to robot base frame
        
        Args:
            point_cloud: Nx3 array of 3D points in camera frame
            
        Returns:
            transformed_cloud: Nx3 array of 3D points in robot base frame
        """
        # Create homogeneous coordinates
        homogeneous_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        
        # Transform to base frame
        transformed_cloud = homogeneous_cloud @ self.camera_to_base_transform.T
        
        # Return 3D coordinates
        return transformed_cloud[:, :3]
    
    def visualize_point_cloud(self, point_cloud, color=None, point_size=6.0, subsample=3):
        """
        Visualize point cloud in PyBullet with improved visualization parameters
        
        Args:
            point_cloud: Nx3 array of 3D points in robot base frame
            color: RGB color for the point cloud (default blue for better visibility)
            point_size: Size of the points (increased from 3.0 to 6.0)
            subsample: Factor to subsample points (reduced from 10 to 3)
        """
        # Remove previous point cloud
        for point_id in self.point_cloud_ids:
            p.removeUserDebugItem(point_id)
        self.point_cloud_ids = []
        
        # Return if no points or very few points
        if point_cloud is None or point_cloud.shape[0] < 10:
            return
        
        # Subsample the point cloud for visualization (less aggressive subsampling)
        if point_cloud.shape[0] > 1000:
            # More points for better visualization (up to 10,000 from 2,000)
            max_points = min(point_cloud.shape[0]//subsample, 10000)
            
            # Use structured subsampling instead of random to maintain shape
            step = max(1, point_cloud.shape[0] // max_points)
            indices = np.arange(0, point_cloud.shape[0], step)
            
            # Ensure we don't exceed point limit
            if len(indices) > max_points:
                indices = indices[:max_points]
                
            point_cloud = point_cloud[indices]
        
        # Default color if not provided - use blue for better visibility against most backgrounds
        if color is None:
            color = [0.0, 0.4, 1.0]
        
        # Add points as debug points
        points_id = p.addUserDebugPoints(
            pointPositions=point_cloud.tolist(),
            pointColorsRGB=[color] * len(point_cloud),
            pointSize=point_size
        )
        self.point_cloud_ids.append(points_id)
        
        # Print info about point cloud
        print(f"Visualizing point cloud with {len(point_cloud)} points")
    
    def visualize_object(self, object_name, position, orientation=None, size=None, color=None):
        """
        Visualize an object in PyBullet
        
        Args:
            object_name: Name of the object (for tracking)
            position: 3D position in robot base frame
            orientation: Quaternion orientation [x, y, z, w]
            size: Size of the object (list of 3 floats)
            color: RGB color for the object
        """
        # Default values
        if orientation is None:
            orientation = [0, 0, 0, 1]  # Identity quaternion
        if size is None:
            size = [0.05, 0.05, 0.05]  # Default size
        if color is None:
            color = [1, 0, 0, 0.7]  # Red by default with alpha
        else:
            color = list(color) + [0.7]  # Add alpha if not provided
            
        # Remove existing object if any
        if object_name in self.object_markers:
            p.removeBody(self.object_markers[object_name])
        if f"{object_name}_text" in self.object_markers:
            p.removeUserDebugItem(self.object_markers[f"{object_name}_text"])
        
        # Create new visualization
        visual_id = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[s/2 for s in size], 
            rgbaColor=color
        )
        
        # Create multibody (no collision shape for visualization)
        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            baseOrientation=orientation
        )
        
        # Add text label
        debug_text_id = p.addUserDebugText(
            text=object_name,
            textPosition=[position[0], position[1], position[2] + size[2] + 0.02],
            textColorRGB=[1, 1, 1],
            textSize=1.5
        )
        
        # Store reference to the body and text
        self.object_markers[object_name] = body_id
        self.object_markers[f"{object_name}_text"] = debug_text_id
        
        return body_id
    
    def process_depth_image(self, depth_image):
        """
        Process depth image and visualize it in PyBullet
        
        Args:
            depth_image: Raw depth image
        """
        # Project to point cloud in camera frame
        point_cloud_camera = self.project_depth_to_point_cloud(depth_image)
        
        # Transform to base frame using current camera extrinsics
        point_cloud_base = self.transform_point_cloud_to_base(point_cloud_camera)
        
        # Visualize point cloud
        self.visualize_point_cloud(point_cloud_base, color=[0.7, 0.7, 0.7], subsample=10)
        
        return point_cloud_base
    
    def update_from_perception(self, color_image, depth_image, object_info=None):
        """
        Update visualization based on perception data
        
        Args:
            color_image: RGB image from camera
            depth_image: Depth image from camera
            object_info: Detected object information
        """
        # Store current images
        self.current_rgb_image = color_image
        self.current_depth_image = depth_image
        
        # Process and visualize depth image
        point_cloud_base = self.process_depth_image(depth_image)
        
        # Update RGB, depth, and overlay visualizations
        self.update_image_visualizations(color_image, depth_image)
        
        # Visualize detected object if available
        if object_info is not None:
            # Extract 3D position from object info
            position = None
            orientation = [0, 0, 0, 1]  # Default orientation
            
            # Try different attributes to get 3D position in world coordinates
            if hasattr(object_info, 'world_pose') and object_info.world_pose is not None:
                position = object_info.world_pose[:3]
            elif hasattr(object_info, 'position') and object_info.position is not None:
                # This might be in camera coordinates, transform to world
                pos_camera = np.array(object_info.position)
                pos_homogeneous = np.append(pos_camera, 1.0)
                position = (self.camera_to_base_transform @ pos_homogeneous)[:3]
            
            # If we have a position, visualize the object
            if position is not None:
                # Estimate size from bounding box if available
                size = [0.05, 0.05, 0.05]  # Default
                if hasattr(object_info, 'bbox') and object_info.bbox is not None:
                    x, y, w, h = object_info.bbox
                    aspect_ratio = w / h if h > 0 else 1.0
                    size = [0.05 * aspect_ratio, 0.05, 0.05]
                
                # Get object name
                name = object_info.name if hasattr(object_info, 'name') else "object"
                
                # Visualize the object
                self.visualize_object(
                    object_name=name,
                    position=position,
                    orientation=orientation,
                    size=size,
                    color=[1, 0, 0]  # Red for detected objects
                )
    
    def update_image_visualizations(self, color_image, depth_image):
        """
        Update the RGB, depth, and overlay image visualizations with enhanced processing
        
        Args:
            color_image: RGB image from camera
            depth_image: Depth image from camera
        """
        try:
            if color_image is None or depth_image is None:
                return
            
            # Make a copy to avoid modifying the original
            color_image_copy = color_image.copy()
            
            # Convert BGR to RGB if needed (OpenCV uses BGR by default)
            if color_image_copy.shape[2] == 3:
                color_image_rgb = cv2.cvtColor(color_image_copy, cv2.COLOR_BGR2RGB)
            else:
                color_image_rgb = color_image_copy
                
            # Normalize depth image for visualization with improved parameters
            depth_normalized = self.normalize_depth_for_visualization(depth_image)
            
            # Apply a colormap to the depth image - changed to TURBO for better differentiation
            # TURBO provides better perceptual differentiation of depth values
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
            
            # Create a more visible depth visualization
            # Add brightness and contrast adjustment
            alpha = 1.2  # Contrast control (1.0 means no change)
            beta = 10    # Brightness control (0 means no change)
            depth_colormap = cv2.convertScaleAbs(depth_colormap, alpha=alpha, beta=beta)
            
            # Blend RGB and depth images with better weights
            # 0.65 RGB + 0.35 depth gives a better balance
            overlay_image = cv2.addWeighted(color_image_rgb, 0.65, depth_colormap, 0.35, 0)
            
            # Enhance contrast of the overlay image
            overlay_lab = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(overlay_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_overlay = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Save the images to temporary files and load them as textures
            self.update_texture_from_image(self.rgb_plane, color_image_rgb, "RGB")
            self.update_texture_from_image(self.depth_plane, depth_colormap, "Depth")
            self.update_texture_from_image(self.overlay_plane, enhanced_overlay, "Overlay")
            
        except Exception as e:
            print(f"Error updating image visualizations: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def update_texture_from_image(self, plane_id, image, texture_type="RGB"):
        """
        Update texture on a plane with an image by saving to a temporary file
        
        Args:
            plane_id: PyBullet ID of the plane
            image: Image to use as texture
            texture_type: Type of texture (RGB, Depth, or Overlay)
        """
        try:
            # Ensure image is the right format
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Create a temporary file name
            temp_dir = os.path.join(os.path.expanduser("~"), ".pybullet_textures")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, f"temp_{texture_type.lower()}_{id(self)}.png")
            
            # Save the image to the temporary file
            cv2.imwrite(temp_file, image)
            
            # Load the texture from the file
            texture_attr = f"{texture_type.lower()}_texture_id"
            
            # Load the texture from file (new texture each time to avoid threading issues)
            texture_id = p.loadTexture(temp_file)
            
            # Store the new texture ID
            setattr(self, texture_attr, texture_id)
            
            # Apply texture to the plane
            p.changeVisualShape(
                plane_id,
                -1,  # Link ID (-1 for base)
                textureUniqueId=texture_id
            )
            
        except Exception as e:
            print(f"Error updating {texture_type} texture: {str(e)}")
    
    def close(self):
        """Close PyBullet connection"""
        p.disconnect(self.physics_client)
        
    def normalize_depth_for_visualization(self, depth_image):
        """
        Normalize depth image for visualization with improved contrast
        
        Args:
            depth_image: Raw depth image
            
        Returns:
            Normalized depth image (0-255, uint8)
        """
        # Convert to float and scale to meters if needed
        depth_scale = self.camera_params.get('depth_scale', 0.001)
        depth_meters = depth_image.astype(np.float32) * depth_scale
        
        # Filter out invalid values - use much less aggressive filtering
        min_depth = 0.001  # 1mm (reduced from 1cm)
        max_depth = 5.0    # 5m (increased from 2m)
        
        # Create a mask for valid values
        valid_mask = (depth_meters > min_depth) & (depth_meters < max_depth)
        
        # Create normalized image
        depth_normalized = np.zeros_like(depth_meters)
        if np.any(valid_mask):
            # Use logarithmic scaling for better visualization of near and far objects
            # Log scaling gives more detail to closer objects while still showing far ones
            depth_log = np.log(depth_meters[valid_mask] / min_depth) / np.log(max_depth / min_depth)
            depth_normalized[valid_mask] = np.clip(depth_log, 0, 1)
            
            # Apply histogram equalization to enhance contrast
            depth_normalized = (depth_normalized * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            depth_normalized = clahe.apply(depth_normalized)
        
        return depth_normalized