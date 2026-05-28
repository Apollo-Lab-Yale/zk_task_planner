import numpy as np
import cv2
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig
from cognitive_bt_framework.src.vision.realsense import Camera
import time
import torch
import colorsys

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

def create_optimized_config():
    """Create optimized FastSAM configuration"""
    return FastSAMConfig(
        model_type="FastSAM-x",
        device="cuda" if torch.cuda.is_available() else "cpu",
        clip_threshold=0.6,
        min_area=10.0,
        merge_overlapping=False
    )

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

def create_segmentation_visualization(image, labeled_masks, metadata):
    """Create a visualization showing all segmentations with CLIP scores"""
    # Create a copy of the image for segmentation visualization
    seg_vis = image.copy()
    
    # Get unique mask IDs (excluding 0 which is background)
    mask_ids = [mid for mid in np.unique(labeled_masks) if mid > 0]
    
    if not mask_ids:
        # No masks found
        cv2.putText(
            seg_vis,
            "No segmentations found",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        return seg_vis
    
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
            
            # Add a small semi-transparent background for the text to improve readability
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
    result = cv2.addWeighted(seg_vis, 0.7, overlay, 0.3, 0)
    
    # Add a title
    cv2.putText(
        result,
        "All Segmentations with CLIP Scores",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return result

def main():
    # Initialize camera
    camera = Camera(width=640, height=480)
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

        # Initialize perception system
        fast_sam_config = create_optimized_config()
        perception = PerceptionSystem(
            fast_sam_config=fast_sam_config,
            camera_matrix=camera_matrix,
            depth_scale=camera.depth_scale,
            debug=True  # Enable profiling
        )

        # Get query from user or use defaults
        user_input = input("Enter objects to detect (comma-separated) or press Enter for defaults: ").strip()
        if user_input:
            target_objects = [obj.strip() for obj in user_input.split(",")]
        else:
            target_objects = ["pen", "bottle", "phone", "book", "cup"]
        current_target_idx = 0
        
        # Get confidence threshold
        try:
            conf_threshold = float(input("Enter confidence threshold (0.0-1.0) or press Enter for default 0.5: ").strip() or "0.5")
            conf_threshold = max(0.0, min(1.0, conf_threshold))  # Clamp to valid range
        except ValueError:
            conf_threshold = 0.5
            print("Invalid input, using default threshold of 0.5")
        
        print(f"Looking for: {', '.join(target_objects)} with confidence threshold {conf_threshold}")
        print("Press 'n' to switch to next object, 's' to save image, 'p' to print performance, 'r' to reset profiler, 'q' to quit")
        
        save_enabled = False
        show_profiling = False
        frame_count = 0
        profiling_interval = 1  # Print profiling stats every 30 frames
        
        while True:
            # Get frames from camera
            frames = camera.get_frames()
            if frames is None:
                print("Failed to get frames")
                continue

            color_image, depth_image = frames
            target_object = target_objects[current_target_idx]
            frame_count += 1

            # Detect object
            start_time = time.time()
            
            # Reset the profiler for specific benchmark frames
            if frame_count % 100 == 1:
                perception.reset_profiler()
                print("\nReset profiler - starting new benchmark")
            
            # First get raw FastSAM results to display all segmentations
            raw_results = perception.segmenter.process_image(
                color_image,
                query=f"a {target_object}"
            )
            
            # Then get the object detection as before
            object_info = perception.detect_object(
                target_object=target_object,
                image=color_image,
                depth_image=depth_image,
                confidence_threshold=conf_threshold
            )
            process_time = time.time() - start_time
            
            # Display top candidates by CLIP score in console
            if raw_results is not None:
                labeled_masks, metadata = raw_results
                
                # Sort metadata by CLIP score
                sorted_results = []
                for mask_id, meta in metadata.items():
                    if 'clip_score' in meta:
                        sorted_results.append((mask_id, meta['clip_score'], meta.get('area', 0)))
                
                # Sort by confidence score (descending)
                sorted_results.sort(key=lambda x: x[1], reverse=True)
                
                # Display top results
                if len(sorted_results) > 0:
                    print("\nTop detection candidates:")
                    print(f"{'ID':^5} | {'Confidence':^10} | {'Area':^10}")
                    print("-" * 30)
                    
                    # Show top 5 or all if fewer
                    for i, (mask_id, confidence, area) in enumerate(sorted_results[:5]):
                        print(f"{mask_id:^5} | {confidence:.3f}     | {int(area):^10}")

            # Create main visualization
            if object_info is not None:
                # Visualize detection
                vis_image = perception.visualize_detection(
                    image=color_image,
                    object_info=object_info,
                    show_pose=True
                )
                
                # Print detection info
                print(f"\nFound {target_object}!")
                print(f"Confidence: {object_info.confidence:.2f}")
                print(f"Bounding box: {object_info.bbox}")
                
                if object_info.pose is not None:
                    x, y, z, roll, pitch, yaw = object_info.pose
                    print(f"Position: ({x:.3f}, {y:.3f}, {z:.3f}) meters")
                    print(f"Orientation: ({np.rad2deg(roll):.1f}°, {np.rad2deg(pitch):.1f}°, {np.rad2deg(yaw):.1f}°)")
            else:
                print(f"\rSearching for {target_object}...", end="")
                vis_image = color_image.copy()
                
            # Print performance stats periodically or when manually requested
            if show_profiling or frame_count % profiling_interval == 0:
                # Print top 5 hotspots in compact format
                print("\n--- Performance Hotspots ---")
                perception.print_performance_summary(sort_by="current_total", top_n=5, compact=True)
                show_profiling = False
                
            # Create segmentation visualization
            if raw_results is not None:
                labeled_masks, metadata = raw_results
                seg_vis = create_segmentation_visualization(color_image, labeled_masks, metadata)
            else:
                # Fallback to depth visualization if no segmentation results
                seg_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
            
            # Stack images horizontally
            display_image = np.hstack((vis_image, seg_vis))
            
            # Add text overlays
            cv2.putText(
                display_image,
                f"Target: {target_object} (threshold: {conf_threshold:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                display_image,
                f"FPS: {1.0/process_time:.1f}" if process_time > 0 else "FPS: N/A",
                (10, display_image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Show result
            cv2.imshow("Perception Test", display_image)
            
            # Save result if requested
            if save_enabled:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"detection_{target_object}_{timestamp}.jpg"
                cv2.imwrite(filename, display_image)
                print(f"\nSaved result as {filename}")
                save_enabled = False

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Switch to next target object
                current_target_idx = (current_target_idx + 1) % len(target_objects)
                print(f"\nSwitching to: {target_objects[current_target_idx]}")
            elif key == ord('s'):
                # Enable saving on next frame
                save_enabled = True
            elif key == ord('p'):
                # Print full performance summary
                show_profiling = True
                print("\n=== Full Performance Summary ===")
                perception.print_performance_summary(sort_by="current_total", compact=False)
            elif key == ord('r'):
                # Reset profiler
                perception.reset_profiler()
                print("\nPerformance profiler has been reset")

    finally:
        # Print final performance summary
        if perception.debug:
            print("\n=== Final Performance Summary ===")
            perception.print_performance_summary(sort_by="total", compact=False)
            
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()