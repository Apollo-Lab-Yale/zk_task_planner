    def _adjust_tcp_for_surface_robot_frame(self, target_position_robot, depth_image, object_mask=None, tcp_standoff_m=0.02):
        """
        Adjust TCP position for top-down grasps by finding surface center in robot frame.
        This ensures proper vertical positioning by working in robot coordinates.
        
        Args:
            target_position_robot: Target position already converted to robot frame [x, y, z]
            depth_image: Depth image from camera
            object_mask: Binary mask of object region
            tcp_standoff_m: Standoff distance above surface in meters
            
        Returns:
            Adjusted position in robot frame with proper vertical TCP offset
        """
        try:
            # Get camera intrinsics
            if hasattr(self, 'latest_depth_image') and hasattr(self, 'main_camera'):
                fx = self.main_camera.intrinsics.fx if hasattr(self.main_camera, 'intrinsics') else 525.0
                fy = self.main_camera.intrinsics.fy if hasattr(self.main_camera, 'intrinsics') else 525.0
                ppx = self.main_camera.intrinsics.ppx if hasattr(self.main_camera, 'intrinsics') else depth_image.shape[1] // 2
                ppy = self.main_camera.intrinsics.ppy if hasattr(self.main_camera, 'intrinsics') else depth_image.shape[0] // 2
            else:
                fx = fy = 525.0  # Default values
                ppx, ppy = depth_image.shape[1] // 2, depth_image.shape[0] // 2
                
            h, w = depth_image.shape
            
            # Detect flat horizontal surfaces for top-down grasps using surface normals
            horizontal_surfaces_mask = self._detect_horizontal_surfaces(depth_image, object_mask)
            
            # Create search region around the target
            if horizontal_surfaces_mask is not None and np.any(horizontal_surfaces_mask):
                search_mask = horizontal_surfaces_mask
                print(f"Using detected horizontal surfaces for TCP centering")
            elif object_mask is not None:
                search_mask = object_mask
                print(f"No horizontal surfaces detected, using full object mask")
            else:
                # Create circular search region as fallback
                radius = 30  # pixels
                y_coords, x_coords = np.ogrid[:h, :w]
                search_mask = (x_coords - ppx)**2 + (y_coords - ppy)**2 <= radius**2
                print(f"Using circular search region as fallback")
            
            # Find surface points within the search mask
            valid_depths = depth_image > 100  # Valid depth values
            surface_region = valid_depths & search_mask
            
            if not np.any(surface_region):
                print("No valid surface points found in search region")
                return None
            
            # For top-down grasps, prioritize the center of the horizontal surface region
            surface_coords = np.where(surface_region)
            if len(surface_coords[0]) < 5:
                print("Insufficient surface points for centering")
                return None
                
            # Convert all surface points to robot frame to find highest Z coordinate
            surface_points_robot = []
            surface_y_coords, surface_x_coords = surface_coords
            
            # Convert each surface point to robot frame
            for i in range(len(surface_y_coords)):
                pixel_y, pixel_x = surface_y_coords[i], surface_x_coords[i]
                depth = depth_image[pixel_y, pixel_x] / 1000.0
                
                # Convert to camera 3D coordinates
                cam_x = (pixel_x - ppx) * depth / fx
                cam_y = (pixel_y - ppy) * depth / fy
                cam_z = depth
                
                # Convert to robot frame
                robot_point = self.convert_cam_pose_to_base([cam_x, cam_y, cam_z], [0, 1, 0, 0])
                if robot_point is not None and len(robot_point) > 0:
                    surface_points_robot.append(robot_point[0])
            
            if not surface_points_robot:
                print("Failed to convert any surface points to robot frame")
                return None
            
            # Find the highest Z coordinate and then find the centroid of all points near that height
            surface_points_robot = np.array(surface_points_robot)
            highest_z = np.max(surface_points_robot[:, 2])
            
            # Define a tolerance for "near highest" points (within 5mm of the top)
            z_tolerance = 0.005  # 5mm tolerance
            near_highest_mask = (surface_points_robot[:, 2] >= (highest_z - z_tolerance))
            near_highest_points = surface_points_robot[near_highest_mask]
            
            # Calculate centroid of the highest surface region for stable grasp positioning
            if len(near_highest_points) > 0:
                centroid_x = np.mean(near_highest_points[:, 0])
                centroid_y = np.mean(near_highest_points[:, 1])
                highest_z_coord = highest_z
            else:
                # Fallback to single highest point if no near-highest points found
                highest_z_idx = np.argmax(surface_points_robot[:, 2])
                highest_point = surface_points_robot[highest_z_idx]
                centroid_x, centroid_y, highest_z_coord = highest_point
            
            print(f"Found {len(surface_points_robot)} surface points in robot frame")
            print(f"Near-highest points ({len(near_highest_points)} within {z_tolerance*1000:.1f}mm of top)")
            print(f"Highest Z: {highest_z_coord:.3f}m, Centroid XY: ({centroid_x:.3f}, {centroid_y:.3f})")
            
            # Apply TCP standoff in robot's Z direction (vertical)
            adjusted_position = [
                centroid_x,  # X: use centroid of highest region for stability
                centroid_y,  # Y: use centroid of highest region for stability
                highest_z_coord + tcp_standoff_m  # Z: add standoff above highest surface
            ]
            
            print(f"Surface analysis: centroid_XY=({centroid_x:.3f}, {centroid_y:.3f}), "
                  f"highest_Z={highest_z_coord:.3f}m, adjusted_with_standoff={adjusted_position}")
            
            return adjusted_position
            
        except Exception as e:
            print(f"Error in surface-based TCP adjustment: {e}")
            return None