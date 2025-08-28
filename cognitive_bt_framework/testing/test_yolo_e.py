#!/usr/bin/env python3
"""
Test script for using Intel Realsense camera with Ultralytics YOLO-E model
for real-time efficient object detection.
"""

import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLOE
import pyrealsense2 as rs
import torch


class RealsenseCamera:
    """Class to handle Intel Realsense camera operations."""
    
    def __init__(self, width=640, height=480, fps=30):
        """Initialize the Realsense camera pipeline.
        
        Args:
            width (int): Frame width
            height (int): Frame height
            fps (int): Frames per second
        """
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable color stream
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Enable depth stream
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get stream profile and camera intrinsics
        self.color_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color))
        self.depth_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.depth))
        self.color_intrinsics = self.color_profile.get_intrinsics()
        
        # Create align object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)
        
        # Warming up
        for _ in range(30):
            self.pipeline.wait_for_frames()
            
        print("Realsense camera initialized successfully")

    def get_frame(self):
        """Get the latest color and depth frames from the camera.
        
        Returns:
            tuple: (color_frame, depth_frame, aligned_depth_frame)
        """
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_frame = frames.get_depth_frame()
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_frame, aligned_depth_frame
    
    def release(self):
        """Stop the pipeline and release resources."""
        self.pipeline.stop()
        print("Realsense camera released")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Realsense camera with YOLO-E')
    parser.add_argument('--model', type=str, default='yoloe-11l-seg.pt', 
                        help='YOLO-E model path or name (yoloe-11s-seg.pt, yoloe-11m-seg.pt, yoloe-11l-seg.pt)')
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=[],
                        help='Initial class prompts for detection (can be changed at runtime)')
    parser.add_argument('--conf', type=float, default=0.3, 
                        help='Confidence threshold for detection')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
    args = parser.parse_args()
    
    # Initialize the Realsense camera
    camera = RealsenseCamera(width=args.width, height=args.height, fps=args.fps)
    
    # Initialize YOLO-E model
    try:
        model = YOLOE(args.model)
        print(f"Loaded YOLO-E model: {args.model}")
        
        # Set custom classes if provided
        if args.classes:
            model.set_classes(args.classes, model.get_text_pe(args.classes))
            print(f"Set custom classes: {args.classes}")
            
    except Exception as e:
        print(f"Error loading YOLO-E model: {e}")
        print("Available YOLO-E models: yoloe-11s-seg.pt, yoloe-11m-seg.pt, yoloe-11l-seg.pt")
        camera.release()
        return
    
    # Create windows for display
    cv2.namedWindow('YOLO-E Detection', cv2.WINDOW_AUTOSIZE)
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Variables for dynamic prompt management
    current_classes = args.classes.copy()
    prompt_input = ""
    show_prompt_ui = False
    
    try:
        while True:
            # Get frame from camera
            color_image, _, _ = camera.get_frame()
            
            # Increment frame count for FPS calculation
            frame_count += 1
            
            # Calculate FPS every second
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Run inference with YOLO-E
            results = model.predict(color_image, conf=args.conf, verbose=False)
            
            # Get the annotated frame
            annotated_frame = results[0].plot()
            
            # Add FPS text
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display current classes being detected
            class_text = f"Detecting: {', '.join(current_classes) if current_classes else 'All classes'}"
            cv2.putText(annotated_frame, class_text, (10, annotated_frame.shape[0] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add instruction for prompt edit mode
            cv2.putText(annotated_frame, "Press 'p' to edit prompts", (10, annotated_frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # If prompt edit mode is active, show the input interface
            if show_prompt_ui:
                # Create semi-transparent overlay
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (50, 100), (annotated_frame.shape[1]-50, 250), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Add title
                cv2.putText(annotated_frame, "Enter new detection prompts", (70, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show current input
                cv2.putText(annotated_frame, prompt_input + "|", (70, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show instructions
                cv2.putText(annotated_frame, "Separate multiple classes with commas", (70, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(annotated_frame, "Press Enter to confirm or Esc to cancel", (70, 230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Display the resulting frame
            cv2.imshow('YOLO-E Detection', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('p') and not show_prompt_ui:  # Enter prompt edit mode
                show_prompt_ui = True
                prompt_input = ""
            elif show_prompt_ui:
                if key == 27:  # Esc key - cancel prompt editing
                    show_prompt_ui = False
                elif key == 13:  # Enter key - confirm new prompts
                    if prompt_input.strip():
                        # Parse the input string into a list of prompts
                        new_classes = [cls.strip() for cls in prompt_input.split(',') if cls.strip()]
                        if new_classes:
                            current_classes = new_classes
                            # Update the model with new classes
                            model.set_classes(current_classes, model.get_text_pe(current_classes))
                            print(f"Updated detection classes: {current_classes}")
                        else:
                            # Empty input means detect all classes
                            current_classes = []
                            print("Switched to detecting all classes")
                    else:
                        # Empty input means detect all classes
                        current_classes = []
                        print("Switched to detecting all classes")
                    show_prompt_ui = False
                elif key == 8:  # Backspace
                    prompt_input = prompt_input[:-1] if prompt_input else ""
                elif 32 <= key <= 126:  # Printable ASCII characters
                    prompt_input += chr(key)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        camera.release()
        cv2.destroyAllWindows()
        print("Application terminated")


if __name__ == "__main__":
    main()