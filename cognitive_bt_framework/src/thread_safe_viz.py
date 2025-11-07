import queue
import threading
import time

class UpdateCommand:
    """Command pattern for thread-safe visualization updates"""
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        
    def execute(self):
        return self.function(*self.args, **self.kwargs)

class ThreadSafeVisualizer:
    """Mixin class to make visualizer thread-safe"""
    
    def initialize_command_queue(self):
        """Initialize the command queue for thread-safe updates"""
        self.command_queue = queue.Queue()
        self.last_command_time = time.time()
        self.max_queue_size = 10  # Limit queue size to prevent memory issues
        
    def process_command_queue(self):
        """Process pending visualization commands on the main thread"""
        try:
            # Process up to N commands per call to avoid blocking
            max_commands_per_call = 5
            commands_processed = 0
            
            while not self.command_queue.empty() and commands_processed < max_commands_per_call:
                command = self.command_queue.get_nowait()
                command.execute()
                self.command_queue.task_done()
                commands_processed += 1
                
            # Print stats if queue is getting backed up
            if self.command_queue.qsize() > self.max_queue_size // 2:
                print(f"Warning: Command queue size: {self.command_queue.qsize()}")
                # Clear queue if it's too large to prevent memory issues
                if self.command_queue.qsize() > self.max_queue_size:
                    print(f"Warning: Command queue too large, clearing...")
                    while not self.command_queue.empty():
                        try:
                            self.command_queue.get_nowait()
                            self.command_queue.task_done()
                        except queue.Empty:
                            break
                
            return True
        except Exception as e:
            print(f"Error processing command queue: {e}")
            return False
            
    def queue_command(self, function, *args, **kwargs):
        """Add a command to the queue for execution on the main thread"""
        try:
            # Create a command object
            command = UpdateCommand(function, *args, **kwargs)
            
            # Add to queue (with timeout to avoid blocking if queue is full)
            self.command_queue.put(command, timeout=0.1)
            
            # Update timestamp
            self.last_command_time = time.time()
            
            return True
        except Exception as e:
            print(f"Error queueing command: {e}")
            return False
            
    def queue_robot_update(self, joint_positions):
        """Queue a robot configuration update"""
        return self.queue_command(self._safe_update_robot_configuration, joint_positions)
        
    def _safe_update_robot_configuration(self, joint_positions):
        """Thread-safe version of update_robot_configuration"""
        try:
            for i, position in enumerate(joint_positions):
                if i < len(self.joint_indices):
                    joint_index = self.joint_indices[i]
                    p.resetJointState(self.robot_id, joint_index, position)
            return True
        except Exception as e:
            print(f"Error updating robot configuration: {e}")
            return False
            
    def queue_process_depth_image(self, depth_image):
        """Queue depth image processing"""
        return self.queue_command(self._safe_process_depth_image, depth_image)
        
    def _safe_process_depth_image(self, depth_image):
        """Thread-safe version of process_depth_image"""
        try:
            # Project depth image to point cloud in camera frame
            point_cloud_camera = self.project_depth_to_point_cloud(depth_image)
            
            # Skip if no points
            if point_cloud_camera is None or len(point_cloud_camera) == 0:
                return
                
            # Transform point cloud from camera to robot base frame
            point_cloud_base = self.transform_points_to_base_frame(point_cloud_camera)
            
            # Visualize point cloud
            self.visualize_point_cloud(point_cloud_base, subsample=3)
            
            return True
        except Exception as e:
            print(f"Error processing depth image: {e}")
            return False
            
    def queue_update_image_visualizations(self, color_image, depth_image):
        """Queue image visualization update"""
        # Make a copy of the images to avoid threading issues
        if color_image is not None:
            color_image = color_image.copy()
        if depth_image is not None:
            depth_image = depth_image.copy()
            
        return self.queue_command(self._safe_update_image_visualizations, color_image, depth_image)
        
    def _safe_update_image_visualizations(self, color_image, depth_image):
        """Thread-safe version of update_image_visualizations"""
        try:
            if color_image is None or depth_image is None:
                return
            
            # Convert BGR to RGB if needed (OpenCV uses BGR by default)
            if color_image.shape[2] == 3:
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            else:
                color_image_rgb = color_image
                
            # Normalize depth image for visualization with improved parameters
            depth_normalized = self.normalize_depth_for_visualization(depth_image)
            
            # Apply a colormap to the depth image - changed to TURBO for better differentiation
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
            
            # Brightness and contrast adjustment
            alpha = 1.2  # Contrast control
            beta = 10    # Brightness control
            depth_colormap = cv2.convertScaleAbs(depth_colormap, alpha=alpha, beta=beta)
            
            # Blend RGB and depth images
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
            
            return True
        except Exception as e:
            print(f"Error updating image visualizations: {e}")
            return False