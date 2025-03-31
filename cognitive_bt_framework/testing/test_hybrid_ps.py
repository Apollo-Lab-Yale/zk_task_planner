#!/usr/bin/env python3
"""
Improved test script for the Hybrid Perception System.
Features:
- Uses both YOLO-World and FastSAM-CLIP for detection and segmentation
- Interactive object switching and parameter adjustment
- Enhanced visualization of detections and segmentations
- Performance monitoring and profiling
"""

import numpy as np
import cv2
import torch
import time
import argparse
import colorsys
from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMConfig, FastSAMWithCLIP
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision.perception_system import PerceptionSystem

def wait_for_valid_frame(camera, max_attempts=30):
    """Wait for a valid frame from the camera with timeout"""
    print("Waiting for valid frame...")
    for _ in range(max_attempts):
        frames = camera.get_frames()
        if frames is not None:
            color_image, depth_image = frames
            if color_image is not None and depth_image is not None:
                print("Valid frame received")
                return frames
        time.sleep(0.1)
    raise RuntimeError(f"Failed to get valid frame after {max_attempts} attempts")


def create_optimized_configs():
    """Create optimized configurations for YOLO-World and FastSAM models"""
    # FastSAM configuration
    fast_sam_config = FastSAMConfig(
        model_type="FastSAM-x",
        device="cuda" if torch.cuda.is_available() else "cpu",
        clip_threshold=0.3,
        min_area=1.0,
        merge_overlapping=False,
        run_time_profile=False
    )
    
    # YOLO-World configuration
    yolo_model_path = "yolov8x-worldv2.pt"
    
    return yolo_model_path, fast_sam_config


def get_unique_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        # Use HSV color space for more distinct colors
        h = i / n
        s = 0.8
        v = 0.9
        # Convert to RGB and then to BGR (for OpenCV)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def create_segmentation_visualization(image, objects, labeled_masks=None, metadata=None):
    """
    Create a comprehensive visualization showing all objects and segmentations
    
    Args:
        image: Original RGB image
        objects: List of detected ObjectInfo objects
        labeled_masks: Optional labeled mask array from FastSAM
        metadata: Optional metadata dict from FastSAM
        
    Returns:
        Visualization image showing all detections with scores
    """
    # Create a copy of the image for visualization
    seg_vis = image.copy()
    
    # First, draw segmentation masks if available
    if labeled_masks is not None and metadata is not None:
        # Get unique mask IDs (excluding 0 which is background)
        mask_ids = [mid for mid in np.unique(labeled_masks) if mid > 0]
        
        # Generate unique colors for each mask
        colors = get_unique_colors(len(mask_ids))
        color_map = {mask_id: colors[i % len(colors)] for i, mask_id in enumerate(mask_ids)}
        
        # Create a transparent overlay for the masks
        overlay = np.zeros_like(seg_vis, dtype=np.uint8)
        
        # Draw each mask with its color and CLIP score
        for i, mask_id in enumerate(mask_ids):
            mask = labeled_masks == mask_id
            color = color_map[mask_id]
            
            # Add mask to overlay
            overlay[mask] = color
            
            # Find position for the label (center of mass of the mask)
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                # Get CLIP score and area if available
                clip_score = metadata.get(mask_id, {}).get('clip_score', 0)
                area = metadata.get(mask_id, {}).get('area', 0)
                
                # Create compact label
                label = f"{mask_id}:{clip_score:.2f}"
                
                # Use smaller font size for the label
                font_scale = 0.4
                font_thickness = 1
                
                # Get text size for potential overlap detection
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Add a small semi-transparent background for the text
                text_bg = seg_vis.copy()
                cv2.rectangle(
                    text_bg,
                    (center_x - text_width//2 - 2, center_y - text_height - 2),
                    (center_x + text_width//2 + 2, center_y + 2),
                    color,
                    -1  # Filled rectangle
                )
                seg_vis = cv2.addWeighted(seg_vis, 0.7, text_bg, 0.3, 0)
                
                # Add the label
                cv2.putText(
                    seg_vis,
                    label,
                    (center_x - text_width//2, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )
        
        # Blend the mask overlay with the image
        seg_vis = cv2.addWeighted(seg_vis, 0.7, overlay, 0.3, 0)
    
    # Now draw all object bounding boxes
    if objects:
        # Generate colors for each object
        obj_colors = get_unique_colors(len(objects))
        
        for i, obj in enumerate(objects):
            # Draw bounding box
            x, y, w, h = obj.bbox
            color = obj_colors[i % len(obj_colors)]
            cv2.rectangle(seg_vis, (x, y), (x + w, y + h), color, 2)
            
            # Add label with confidence
            label = f"{obj.name} ({obj.confidence:.2f})"
            
            # Add text with background for better visibility
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                seg_vis,
                (x, y - text_size[1] - 5),
                (x + text_size[0], y),
                color,
                -1
            )
            
            cv2.putText(
                seg_vis,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            # Draw component boxes and labels if available
            if obj.components:
                for comp_name, comp in obj.components.items():
                    cx, cy, cw, ch = comp.bbox
                    comp_color = (255, 0, 0)  # Blue for components
                    
                    cv2.rectangle(seg_vis, (cx, cy), (cx + cw, cy + ch), comp_color, 1)
                    
                    # Add component label
                    comp_label = f"{comp_name} ({comp.confidence:.2f})"
                    cv2.putText(
                        seg_vis,
                        comp_label,
                        (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        comp_color,
                        1
                    )
    
    # Add a title
    cv2.putText(
        seg_vis,
        "All Detections and Segmentations",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return seg_vis


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Hybrid Perception System')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--conf', type=float, default=0.5, help='Default confidence threshold')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Initialize camera
    camera = Camera(width=args.width, height=args.height)
    if not camera.start():
        print("Failed to start camera")
        return
    
    try:
        # Wait for valid frame
        try:
            frames = wait_for_valid_frame(camera)
        except RuntimeError as e:
            print(e)
            return
        
        # Get camera matrix from RealSense intrinsics
        camera_matrix = np.array([
            [camera.intrinsics.fx, 0, camera.intrinsics.ppx],
            [0, camera.intrinsics.fy, camera.intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Initialize hybrid perception system
        yolo_model_path, fast_sam_config = create_optimized_configs()
        perception = PerceptionSystem(
            yolo_model_path=yolo_model_path,
            fast_sam_config=fast_sam_config,
            camera_matrix=camera_matrix,
            depth_scale=camera.depth_scale,
            debug=True
        )
        
        # Get object classes from user or use defaults
        user_input = input("Enter objects to detect (comma-separated) or press Enter for defaults: ").strip()
        if user_input:
            target_objects = [obj.strip() for obj in user_input.split(",")]
        else:
            target_objects = ["person", "bottle", "laptop", "cup", "keyboard", "mouse", "book", "cell phone"]
        
        # Initialize component detection
        component_names = ["top", "bottom", "handle", "screen", "keyboard", "button", "cap", "lens", "display", "usb port"]
        
        # Get confidence threshold
        try:
            conf_threshold = float(input("Enter confidence threshold (0.0-1.0) or press Enter for default 0.5: ").strip() or "0.5")
            conf_threshold = max(0.0, min(1.0, conf_threshold))  # Clamp to valid range
        except ValueError:
            conf_threshold = args.conf
            print(f"Invalid input, using default threshold of {conf_threshold}")
        
        # Set initial operating modes
        detection_mode = "objects"  # Options: "objects", "object", "components"
        current_obj_idx = 0
        current_component_idx = 0
        selected_object = None
        
        # Print instructions
        print(f"\nLooking for: {', '.join(target_objects)} with confidence threshold {conf_threshold}")
        print("Detection Modes:")
        print("  - 'objects': Detect all objects in the scene")
        print("  - 'object': Focus on a single object type")
        print("  - 'components': Detect components within a selected object")
        print("\nControls:")
        print("  - 'm': Switch between detection modes")
        print("  - 'n': Next object/component")
        print("  - 'p': Previous object/component")
        print("  - 'c': Detect components in the selected object")
        print("  - '+'/'-': Adjust confidence threshold")
        print("  - 's': Save current frame")
        print("  - 'd': Print detailed performance stats")
        print("  - 'r': Reset performance profiler")
        print("  - 'q': Quit")
        
        # Tracking variables
        save_enabled = False
        show_profiling = False
        frame_count = 0
        fps_history = []
        fps_smoothing = 10  # Number of frames to average for FPS calculation
        profiling_interval = 30  # Print profiling stats every N frames
        
        # Create windows
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
        
        # Main loop
        while True:
            # Get frames from camera
            frames = camera.get_frames()
            if frames is None:
                print("Failed to get frames")
                continue
            
            color_image, depth_image = frames
            frame_count += 1
            
            # Reset the profiler periodically for benchmarking
            if frame_count % 100 == 1:
                perception.reset_profiler()
                if args.debug:
                    print("\nReset profiler - starting new benchmark")
            
            # Start timing
            start_time = time.time()
            raw_segmentation_results = None
            objects = []
            
            # Detection logic based on current mode
            if detection_mode == "objects":
                # Detect all objects in the scene
                objects = perception.detect_objects(
                    image=color_image,
                    classes=target_objects,
                    conf=conf_threshold,
                    segment=True,
                    depth_image=depth_image
                )
                
                # Update the selected object if needed
                if objects and (selected_object is None or current_obj_idx >= len(objects)):
                    selected_object = objects[0]
                    current_obj_idx = 0
                elif objects:
                    selected_object = objects[current_obj_idx]
                
            elif detection_mode == "object":
                # Focus on a single object type
                current_target = target_objects[current_obj_idx]
                
                # First get raw FastSAM results if available
                if perception.segmenter is not None:
                    raw_segmentation_results = perception.segmenter.process_image(
                        color_image,
                        query=f"a {current_target}"
                    )
                
                # Then get the object detection
                object_info = perception.detect_object(
                    target_object=current_target,
                    image=color_image,
                    conf=conf_threshold,
                    segment=True,
                    depth_image=depth_image
                )
                if object_info is not None:
                    objects = [object_info]
                    selected_object = object_info
                else:
                    objects = []
                    selected_object = None
                
            elif detection_mode == "components" and selected_object is not None:
                # Component detection mode - detect a specific component in the selected object
                current_component = component_names[current_component_idx]
                
                # Detect the component
                component = perception.detect_object_component(
                    object_info=selected_object,
                    component_name=current_component,
                    conf_threshold=conf_threshold,
                    depth_image=depth_image
                )
                
                # Always keep showing the parent object
                objects = [selected_object]
                
                # Display component info
                if component is not None:
                    print(f"\nDetected component '{current_component}' with confidence {component.confidence:.3f}")
                    if component.pose is not None:
                        x, y, z = component.pose[:3]
                        print(f"Component position: ({x:.3f}, {y:.3f}, {z:.3f}) meters")
                else:
                    print(f"\rSearching for component '{current_component}'...", end="")
            
            # Calculate process time
            process_time = time.time() - start_time
            
            # Calculate FPS with smoothing
            fps_history.append(1.0/process_time if process_time > 0 else 0)
            if len(fps_history) > fps_smoothing:
                fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history) if fps_history else 0
            
            # Display summary of detected objects
            if objects:
                print("\nDetected objects:")
                for i, obj in enumerate(objects):
                    status = " (selected)" if obj == selected_object else ""
                    print(f"{i}: {obj.name} - Conf: {obj.confidence:.3f}{status}")
                    print(f"Object position: {obj.pose}")
                    # Show component info if available
                    if obj.components:
                        print(f"  Components:")
                        for comp_name, comp in obj.components.items():
                            print(f"  - {comp_name}: Conf: {comp.confidence:.3f}")
            
            # Print performance stats periodically or when manually requested
            if show_profiling or frame_count % profiling_interval == 0:
                # Print top 5 hotspots in compact format
                print("\n--- Performance Hotspots ---")
                perception.print_performance_summary(sort_by="current_total", top_n=5, compact=True)
                show_profiling = False
            
            # Create main detection visualization
            detection_vis = perception.visualize_detections(
                image=color_image,
                objects=objects,
                show_masks=True,
                show_components=True,
                show_poses=True
            )
            
            # Create segmentation visualization
            if raw_segmentation_results is not None:
                labeled_masks, metadata = raw_segmentation_results
                seg_vis = create_segmentation_visualization(
                    color_image, 
                    objects, 
                    labeled_masks, 
                    metadata
                )
            else:
                # Use normal segmentation visualization
                seg_vis = create_segmentation_visualization(color_image, objects)
            # Add mode and status information
            # Add header text to main image
            mode_text = f"Mode: {detection_mode.capitalize()}"
            if detection_mode == "object":
                mode_text += f" - Target: {target_objects[current_obj_idx]}"
            elif detection_mode == "components":
                mode_text += f" - Component: {component_names[current_component_idx]}"
            
            cv2.putText(
                detection_vis,
                mode_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            # Add confidence threshold info
            cv2.putText(
                detection_vis,
                f"Threshold: {conf_threshold:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Add FPS counter
            cv2.putText(
                detection_vis,
                f"FPS: {fps:.1f}",
                (10, detection_vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Show the visualizations
            cv2.imshow("Detection", detection_vis)
            cv2.imshow("Segmentation", seg_vis)
            
            # Save images if requested
            if save_enabled:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                detection_filename = f"detection_{timestamp}.jpg"
                segmentation_filename = f"segmentation_{timestamp}.jpg"
                
                cv2.imwrite(detection_filename, detection_vis)
                cv2.imwrite(segmentation_filename, seg_vis)
                
                print(f"\nSaved images as {detection_filename} and {segmentation_filename}")
                save_enabled = False
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                break
                
            elif key == ord('m'):
                # Switch detection mode
                if detection_mode == "objects":
                    detection_mode = "object"
                    print(f"\nSwitched to single object mode - Target: {target_objects[current_obj_idx]}")
                elif detection_mode == "object":
                    if selected_object is not None:
                        detection_mode = "components"
                        print(f"\nSwitched to component detection mode - Component: {component_names[current_component_idx]}")
                    else:
                        detection_mode = "objects"
                        print("\nSwitched to multi-object detection mode")
                else:  # components mode
                    detection_mode = "objects"
                    print("\nSwitched to multi-object detection mode")
                    
            elif key == ord('n'):
                # Next object/component
                if detection_mode == "object":
                    current_obj_idx = (current_obj_idx + 1) % len(target_objects)
                    print(f"\nSwitched to target: {target_objects[current_obj_idx]}")
                elif detection_mode == "objects" and objects:
                    current_obj_idx = (current_obj_idx + 1) % len(objects)
                    selected_object = objects[current_obj_idx]
                    print(f"\nSelected object {current_obj_idx}: {selected_object.name}")
                elif detection_mode == "components":
                    current_component_idx = (current_component_idx + 1) % len(component_names)
                    print(f"\nSwitched to component: {component_names[current_component_idx]}")
                    
            elif key == ord('p'):
                # Previous object/component
                if detection_mode == "object":
                    current_obj_idx = (current_obj_idx - 1) % len(target_objects)
                    print(f"\nSwitched to target: {target_objects[current_obj_idx]}")
                elif detection_mode == "objects" and objects:
                    current_obj_idx = (current_obj_idx - 1) % len(objects)
                    selected_object = objects[current_obj_idx]
                    print(f"\nSelected object {current_obj_idx}: {selected_object.name}")
                elif detection_mode == "components":
                    current_component_idx = (current_component_idx - 1) % len(component_names)
                    print(f"\nSwitched to component: {component_names[current_component_idx]}")
                    
            elif key == ord('c'):
                # Toggle component detection for selected object
                if selected_object is not None and detection_mode != "components":
                    detection_mode = "components"
                    print(f"\nEntering component detection mode - Component: {component_names[current_component_idx]}")
                elif detection_mode == "components":
                    detection_mode = "objects" if len(objects) > 1 else "object"
                    print(f"\nExiting component detection mode")
                else:
                    print("\nNo object selected for component detection")
                    
            elif key == ord('+') or key == ord('='):
                # Increase confidence threshold
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"\nIncreased confidence threshold to {conf_threshold:.2f}")
                
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"\nDecreased confidence threshold to {conf_threshold:.2f}")
                
            elif key == ord('s'):
                # Enable image saving on next frame
                save_enabled = True
                print("\nSaving next frame...")
                
            elif key == ord('d'):
                # Show detailed performance stats
                show_profiling = True
                print("\n=== Detailed Performance Summary ===")
                perception.print_performance_summary(sort_by="current_total", compact=False)
                
            elif key == ord('r'):
                # Reset profiler
                perception.reset_profiler()
                print("\nPerformance profiler has been reset")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Print final performance summary
        if perception.debug:
            print("\n=== Final Performance Summary ===")
            perception.print_performance_summary(sort_by="total", compact=False)
            
        # Release resources
        camera.stop()
        perception.release()
        cv2.destroyAllWindows()
        print("\nApplication terminated")


if __name__ == "__main__":
    main()