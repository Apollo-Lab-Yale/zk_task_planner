#!/usr/bin/env python3
"""
ZED Camera Depth Visualization Test Script

This script demonstrates how to capture and visualize depth images from a ZED camera
using the StereoLabs ZED SDK. It provides real-time depth visualization with various
display modes and depth value inspection capabilities.

Requirements:
- ZED SDK installed (https://www.stereolabs.com/developers/release/)
- ZED camera connected
- Python packages: pyzed, opencv-python, numpy, matplotlib

Usage:
    python zed_depth_test.py

Controls:
    - 'q' or ESC: Quit
    - 's': Save current depth map and image
    - 'd': Toggle depth display mode
    - 'c': Toggle colormap
    - Click on image: Print depth value at clicked position
"""

import sys
import cv2
import numpy as np
import pyzed.sl as sl
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ZEDDepthVisualizer:
    def __init__(self):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.runtime_params = sl.RuntimeParameters()
        
        # Matrices to store images and depth data
        self.image = sl.Mat()
        self.depth_map = sl.Mat()
        self.depth_for_display = sl.Mat()
        self.point_cloud = sl.Mat()
        
        # Display settings
        self.display_mode = 0  # 0: normalized depth, 1: raw depth values
        self.colormap = cv2.COLORMAP_JET
        self.colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA]
        self.colormap_index = 0
        
        # Mouse callback variables
        self.mouse_x = 0
        self.mouse_y = 0
        
    def initialize_camera(self):
        """Initialize the ZED camera with optimal settings for depth sensing."""
        # Set configuration parameters
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 30
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        self.init_params.depth_stabilization = 1
        
        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open ZED camera: {err}")
            return False
            
        # Set runtime parameters
        self.runtime_params.confidence_threshold = 50
        self.runtime_params.texture_confidence_threshold = 100
        
        # Get camera information
        camera_info = self.zed.get_camera_information()
        print(f"ZED Camera opened successfully")
        print(f"Resolution: {camera_info.camera_configuration.resolution.width}x{camera_info.camera_configuration.resolution.height}")
        print(f"FPS: {camera_info.camera_configuration.fps}")
        print(f"Depth mode: {self.init_params.depth_mode}")
        
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to get depth values at clicked positions."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x, self.mouse_y = x, y
            # Get depth value at clicked position
            depth_value = self.depth_map.get_value(x, y)
            if not np.isnan(depth_value[1]) and depth_value[1] > 0:
                print(f"Depth at ({x}, {y}): {depth_value[1]:.2f} mm ({depth_value[1]/1000:.3f} m)")
            else:
                print(f"No valid depth data at ({x}, {y})")
    
    
    def display_info_overlay(self, image):
        """Add information overlay to the image."""
        height, width = image.shape[:2]
        
        # Create info text
        colormap_name = {
            cv2.COLORMAP_JET: "JET",
            cv2.COLORMAP_TURBO: "TURBO", 
            cv2.COLORMAP_VIRIDIS: "VIRIDIS",
            cv2.COLORMAP_PLASMA: "PLASMA"
        }[self.colormap]
        
        info_text = [
            f"Mode: {'Raw Depth' if self.display_mode else 'Normalized'}",
            f"Colormap: {colormap_name}",
            f"Mouse: ({self.mouse_x}, {self.mouse_y})",
            "Controls: 'q'=quit, 's'=save, 'd'=mode, 'c'=colormap"
        ]
        
        # Add text overlay
        for i, text in enumerate(info_text):
            cv2.putText(image, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image
    
    def create_depth_colormap(self, depth_image):
        """Create a colorized version of the depth image."""
        # Convert to proper format for OpenCV colormap
        if len(depth_image.shape) == 3:
            # If it's a 3-channel image, convert to grayscale
            if depth_image.shape[2] == 4:  # RGBA
                depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_RGBA2GRAY)
            elif depth_image.shape[2] == 3:  # RGB
                depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
            else:
                depth_gray = depth_image[:, :, 0]  # Take first channel
        else:
            # Already grayscale
            depth_gray = depth_image
        
        # Ensure we have the right data type
        if depth_gray.dtype != np.uint8:
            # Normalize to 0-255 range and convert to uint8
            depth_normalized = cv2.normalize(depth_gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
        else:
            depth_uint8 = depth_gray
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, self.colormap)
        return depth_colored
    
    def save_data(self):
        """Save current depth map and RGB image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        output_dir = "zed_captures"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save RGB image
        rgb_filename = f"{output_dir}/rgb_{timestamp}.png"
        cv2.imwrite(rgb_filename, self.image.get_data())
        
        # Save depth image (8-bit for display)
        depth_display_filename = f"{output_dir}/depth_display_{timestamp}.png"
        cv2.imwrite(depth_display_filename, self.depth_for_display.get_data())
        
        # Save raw depth data as numpy array
        depth_raw_filename = f"{output_dir}/depth_raw_{timestamp}.npy"
        np.save(depth_raw_filename, self.depth_map.get_data())
        
        # Save colorized depth image
        depth_colored = self.create_depth_colormap(self.depth_for_display.get_data())
        depth_colored_filename = f"{output_dir}/depth_colored_{timestamp}.png"
        cv2.imwrite(depth_colored_filename, depth_colored)
        
        print(f"Data saved to {output_dir}/ with timestamp {timestamp}")
    
    def display_info_overlay(self, image):
        """Add information overlay to the image."""
        height, width = image.shape[:2]
        
        # Create info text
        colormap_name = {
            cv2.COLORMAP_JET: "JET",
            cv2.COLORMAP_TURBO: "TURBO", 
            cv2.COLORMAP_VIRIDIS: "VIRIDIS",
            cv2.COLORMAP_PLASMA: "PLASMA"
        }[self.colormap]
        
        info_text = [
            f"Mode: {'Raw Depth' if self.display_mode else 'Normalized'}",
            f"Colormap: {colormap_name}",
            f"Mouse: ({self.mouse_x}, {self.mouse_y})",
            "Controls: 'q'=quit, 's'=save, 'd'=mode, 'c'=colormap"
        ]
        
        # Add text overlay
        for i, text in enumerate(info_text):
            cv2.putText(image, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image
    
    def run(self):
        """Main visualization loop."""
        if not self.initialize_camera():
            return
        
        print("\nStarting depth visualization...")
        print("Controls:")
        print("  'q' or ESC: Quit")
        print("  's': Save current data")
        print("  'd': Toggle depth display mode")
        print("  'c': Cycle through colormaps")
        print("  Click on image: Print depth value")
        print()
        
        # Set up OpenCV windows
        cv2.namedWindow("ZED Depth Visualization", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("ZED Depth Visualization", self.mouse_callback)
        
        try:
            while True:
                # Grab a new frame
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve RGB image
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    
                    # Retrieve depth map (32-bit)
                    self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                    
                    # Retrieve normalized depth image for display (8-bit)
                    self.zed.retrieve_image(self.depth_for_display, sl.VIEW.DEPTH)
                    
                    # Convert to OpenCV format
                    rgb_image = self.image.get_data()
                    depth_display_image = self.depth_for_display.get_data()
                    
                    # Debug: Print image info on first frame
                    if not hasattr(self, '_debug_printed'):
                        print(f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
                        print(f"Depth display image shape: {depth_display_image.shape}, dtype: {depth_display_image.dtype}")
                        self._debug_printed = True
                    
                    # Create visualization
                    try:
                        if self.display_mode == 0:
                            # Use normalized depth image
                            viz_image = self.create_depth_colormap(depth_display_image)
                        else:
                            # Use raw depth values with custom normalization
                            raw_depth = self.depth_map.get_data()
                            # Handle different data types
                            if raw_depth.dtype != np.float32:
                                raw_depth = raw_depth.astype(np.float32)
                            
                            # Clip extreme values and normalize
                            depth_clipped = np.clip(raw_depth, 0, 5000)  # Clip to 5m
                            depth_norm = (depth_clipped / 5000 * 255).astype(np.uint8)
                            viz_image = self.create_depth_colormap(depth_norm)
                    except Exception as e:
                        print(f"Error creating depth visualization: {e}")
                        # Fallback: create a simple grayscale visualization
                        if len(depth_display_image.shape) == 3:
                            viz_image = cv2.cvtColor(depth_display_image, cv2.COLOR_RGB2BGR)
                        else:
                            viz_image = cv2.cvtColor(depth_display_image, cv2.COLOR_GRAY2BGR)
                        continue
                    
                    # Add info overlay
                    viz_image = self.display_info_overlay(viz_image)
                    
                    # Create side-by-side display
                    # Ensure RGB image is in correct format
                    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 4:  # RGBA
                        rgb_display = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)
                    elif len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:  # RGB
                        rgb_display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    else:
                        rgb_display = rgb_image
                    
                    rgb_resized = cv2.resize(rgb_display, (viz_image.shape[1], viz_image.shape[0]))
                    combined = np.hstack([rgb_resized, viz_image])
                    
                    # Display the image
                    cv2.imshow("ZED Depth Visualization", combined)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):
                    self.save_data()
                elif key == ord('d'):
                    self.display_mode = 1 - self.display_mode
                    print(f"Switched to {'raw depth' if self.display_mode else 'normalized'} mode")
                elif key == ord('c'):
                    self.colormap_index = (self.colormap_index + 1) % len(self.colormaps)
                    self.colormap = self.colormaps[self.colormap_index]
                    print(f"Switched to colormap: {self.colormap}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Clean up
            cv2.destroyAllWindows()
            self.zed.close()
            print("ZED camera closed")

def main():
    """Main function to run the depth visualization."""
    print("ZED Camera Depth Visualization Test")
    print("===================================")
    
    # Check if ZED SDK is available
    try:
        import pyzed.sl as sl
    except ImportError:
        print("Error: ZED SDK (pyzed) not found!")
        print("Please install the ZED SDK from: https://www.stereolabs.com/developers/release/")
        return
    
    # Create and run visualizer
    visualizer = ZEDDepthVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()