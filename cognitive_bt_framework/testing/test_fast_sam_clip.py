import numpy as np
import cv2
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMWithCLIP, FastSAMConfig
import time
import torch
import os
from cognitive_bt_framework.utils.time_profiler import IterationTimeProfiler

class ColorPalette:
    def __init__(self):
        self.colors = {}
        self.next_color_idx = 0
        self.palette = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 255, 0),  # Light green
            (0, 128, 255),  # Orange
            (255, 0, 128)   # Purple
        ]
    
    def get_color(self, mask_id):
        if mask_id not in self.colors:
            self.colors[mask_id] = self.palette[self.next_color_idx % len(self.palette)]
            self.next_color_idx += 1
        return self.colors[mask_id]

def create_optimized_config():
    return FastSAMConfig(
        model_type="FastSAM-x",
        device="cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold=0.1,
        iou_threshold=0.25,
        clip_threshold=0.6,
        min_area=1.0,
        merge_overlapping=False,
        # Enable profiling in the FastSAM config
        run_time_profile=True,
        debug=False
    )

def wait_for_valid_frame(camera, max_attempts=30, profiler=None):
    """Wait for a valid frame from the camera with timeout"""
    print("Waiting for valid frame...")
    if profiler:
        profiler.start("wait_for_valid_frame")
    
    for attempt in range(max_attempts):
        if profiler:
            profiler.start("camera_get_frames")
        frames = camera.get_frames()
        if profiler:
            profiler.stop("camera_get_frames")
            
        if frames is not None:
            color_image, depth_image = frames
            if color_image is not None and depth_image is not None:
                print(f"Valid frame received after {attempt+1} attempts")
                if profiler:
                    profiler.stop("wait_for_valid_frame")
                return frames
        time.sleep(0.1)
    
    if profiler:
        profiler.stop("wait_for_valid_frame")
    raise RuntimeError(f"Failed to get valid frame after {max_attempts} attempts")

def process_frame(color_image, depth_image, query, fastsam, color_palette, profiler=None):
    """Process a single frame and create visualization"""
    # Process image with FastSAM
    print(f"Processing frame with query: {query}")
    
    if profiler:
        profiler.start_iteration()
        profiler.start("total_processing")
        profiler.start("fastsam_process")
    
    results = fastsam.process_image(color_image, query)
    
    if profiler:
        profiler.stop("fastsam_process")
        
    if results is None:
        if profiler:
            profiler.stop("total_processing")
            profiler.end_iteration(preserve_current=True)
        return None
        
    labeled_masks, metadata = results
    
    # Create visualization
    if profiler:
        profiler.start("visualization")
        
    vis_image = color_image.copy()
    overlay = np.zeros_like(vis_image, dtype=np.uint8)
    
    # Draw masks and labels
    for mask_id in np.unique(labeled_masks)[1:]:  # Skip 0 (background)
        if profiler:
            profiler.start(f"visualize_mask_{mask_id}")
            
        if mask_id in metadata:
            mask = labeled_masks == mask_id
            color = color_palette.get_color(mask_id)
            overlay[mask] = color
            
            # Add label with score
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                label = f"r{mask_id}"
                if 'clip_score' in metadata[mask_id]:
                    label += f" ({metadata[mask_id]['clip_score']:.2f})"
                
                cv2.putText(
                    vis_image,
                    label,
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
        
        if profiler:
            profiler.stop(f"visualize_mask_{mask_id}")
    
    # Create final blended image
    if profiler:
        profiler.start("blend_overlay")
    vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
    if profiler:
        profiler.stop("blend_overlay")
    
    # Create depth visualization
    if profiler:
        profiler.start("depth_colormap")
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    if profiler:
        profiler.stop("depth_colormap")
    
    # Stack images horizontally
    if profiler:
        profiler.start("stack_images")
    display_image = np.hstack((vis_image, depth_colormap))
    if profiler:
        profiler.stop("stack_images")
    
    # Add query overlay
    cv2.putText(
        display_image,
        f"Query: {query}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    if profiler:
        profiler.stop("visualization")
        
    # Print detection information
    print(f"\nDetection Results:")
    print(f"Found {len(metadata)} masks")
    for mask_id, meta in metadata.items():
        print(f"Mask {mask_id}:")
        print(f"  Area: {meta['area']:.1f} pixels")
        print(f"  CLIP score: {meta.get('clip_score', 'N/A')}")
        print(f"  Bounding box: {meta['bbox']}")
    
    if profiler:
        profiler.stop("total_processing")
        profiler.end_iteration(preserve_current=True)
        
    return display_image, results

def main():
    # Initialize profiler
    profiler = IterationTimeProfiler(enabled=True)
    profiler.start_iteration()
    profiler.start("total_execution")
    
    # Get query from user
    query = input("Enter object to detect: ").strip()
    if not query:
        print("No query provided. Exiting.")
        profiler.stop("total_execution")
        profiler.end_iteration(preserve_current=True)
        profiler.print_summary()
        return

    # Initialize camera
    profiler.start("camera_initialization")
    camera = Camera(width=640, height=480)
    if not camera.start():
        print("Failed to start camera")
        profiler.stop("camera_initialization")
        profiler.stop("total_execution")
        profiler.end_iteration(preserve_current=True)
        profiler.print_summary()
        return
    profiler.stop("camera_initialization")

    try:
        # Initialize FastSAM
        profiler.start("fastsam_initialization")
        config = create_optimized_config()
        fastsam = FastSAMWithCLIP(config)
        profiler.stop("fastsam_initialization")
        
        color_palette = ColorPalette()
        
        # Wait for valid frame
        try:
            frames = wait_for_valid_frame(camera, profiler=profiler)
        except RuntimeError as e:
            print(e)
            profiler.stop("total_execution")
            profiler.end_iteration(preserve_current=True)
            profiler.print_summary()
            return
            
        color_image, depth_image = frames
        
        # Process frame
        result = process_frame(color_image, depth_image, query, fastsam, color_palette, profiler=profiler)
        if result is None:
            print("Failed to process frame")
            profiler.stop("total_execution")
            profiler.end_iteration(preserve_current=True)
            profiler.print_summary()
            return
            
        display_image, (labeled_masks, metadata) = result
        
        # Save result
        profiler.start("save_result")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"fastsam_result_{timestamp}.jpg")
        cv2.imwrite(filename, display_image)
        profiler.stop("save_result")
        
        print(f"\nSaved result as {filename}")
        
        # Also save performance data
        profiler.start("save_performance_data")
        perf_file = os.path.join(output_dir, f"performance_{timestamp}.txt")
        with open(perf_file, 'w') as f:
            f.write(f"Performance data for query: {query}\n")
            f.write(f"Device: {config.device}\n")
            f.write(f"Model: {config.model_type}\n\n")
            
            summary = profiler.get_summary(sort_by="total")
            
            f.write(f"{'Section':<30} {'Calls':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10}\n")
            f.write("-" * 85 + "\n")
            
            for item in summary:
                f.write(f"{item['section']:<30} {item['current_calls']:<8} {item['current_total']:<10.3f} "
                      f"{item['average']:<10.3f} {item['min']:<10.3f} {item['max']:<10.3f}\n")
        profiler.stop("save_performance_data")
        
        print(f"Saved performance data as {perf_file}")
        
        # Show result briefly
        profiler.start("display_result")
        cv2.imshow("Result", display_image)
        cv2.waitKey(3000)  # Show for 3 seconds
        profiler.stop("display_result")
        
        # Run multiple test iterations (optional)
        run_iterations = input("Run performance test iterations? (y/n): ").strip().lower()
        if run_iterations == 'y':
            num_iterations = int(input("Enter number of iterations: ").strip())
            
            print(f"\nRunning {num_iterations} iterations for performance testing...")
            
            profiler.start("iterations_total")
            
            for i in range(num_iterations):
                print(f"\nIteration {i+1}/{num_iterations}")
                
                # Get frame
                profiler.start("iter_get_frame")
                frames = camera.get_frames()
                profiler.stop("iter_get_frame")
                
                if frames is None:
                    print("Failed to get frame, skipping iteration")
                    continue
                    
                color_image, depth_image = frames
                
                # Process frame
                profiler.start(f"iter_{i}_process")
                result = process_frame(color_image, depth_image, query, fastsam, color_palette, profiler=profiler)
                profiler.stop(f"iter_{i}_process")
                
                if result is None:
                    print("Failed to process frame")
                    continue
                
                # Brief display
                cv2.imshow("Result", result[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            profiler.stop("iterations_total")
            
            # Get FastSAM performance metrics directly from its profiler
            print("\nFastSAM internal performance metrics:")
            fastsam.print_performance_summary(sort_by="current_total", top_n=10, compact=True)
            
            # Save detailed performance report
            perf_file = os.path.join(output_dir, f"detailed_performance_{timestamp}.txt")
            with open(perf_file, 'w') as f:
                f.write(f"Detailed Performance Report\n")
                f.write(f"Query: {query}\n")
                f.write(f"Iterations: {num_iterations}\n")
                f.write(f"Device: {config.device}\n")
                f.write(f"Model: {config.model_type}\n\n")
                
                # Compute fastsam_process statistics 
                fastsam_times = [item['current_total'] for item in profiler.get_summary() 
                                if item['section'] == "fastsam_process"]
                avg_time = profiler.get_avg_time("fastsam_process")
                min_time = profiler.get_min_time("fastsam_process") 
                max_time = profiler.get_max_time("fastsam_process")
                
                if fastsam_times:
                    f.write("FastSAM Processing Statistics:\n")
                    f.write(f"  Average: {avg_time*1000:.2f} ms\n")
                    f.write(f"  Min: {min_time*1000:.2f} ms\n")
                    f.write(f"  Max: {max_time*1000:.2f} ms\n")
                    f.write(f"  FPS: {1.0/avg_time:.2f}\n\n")
                
                # Write full timing summary
                f.write("TEST SCRIPT TIMING SUMMARY:\n")
                summary = profiler.get_summary(sort_by="total")
                
                f.write(f"{'Section':<30} {'Calls':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10}\n")
                f.write("-" * 85 + "\n")
                
                for item in summary:
                    f.write(f"{item['section']:<30} {item['current_calls']:<8} {item['current_total']:<10.3f} "
                          f"{item['average']:<10.3f} {item['min']:<10.3f} {item['max']:<10.3f}\n")
                
                # Also include FastSAM's internal profiling data
                f.write("\n\nFASTSAM INTERNAL TIMING SUMMARY:\n")
                fastsam_summary = fastsam.profiler.get_summary(sort_by="total")
                
                f.write(f"{'Section':<30} {'Calls':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10}\n")
                f.write("-" * 85 + "\n")
                
                for item in fastsam_summary:
                    f.write(f"{item['section']:<30} {item['current_calls']:<8} {item['current_total']:<10.3f} "
                          f"{item['average']:<10.3f} {item['min']:<10.3f} {item['max']:<10.3f}\n")
            
            print(f"Saved detailed performance report as {perf_file}")

    finally:
        if 'camera' in locals():
            profiler.start("camera_stop")
            camera.stop()
            profiler.stop("camera_stop")
            
        cv2.destroyAllWindows()
        
        profiler.stop("total_execution")
        profiler.end_iteration(preserve_current=True)
        
        # Print a summary of the test execution
        print("\n=== TEST EXECUTION PERFORMANCE SUMMARY ===")
        profiler.print_summary(sort_by="current_total", top_n=10, compact=True)
        
        # If fastsam was initialized, also print its internal profiling data
        if 'fastsam' in locals():
            print("\n=== FASTSAM INTERNAL PERFORMANCE SUMMARY ===")
            fastsam.print_performance_summary(sort_by="current_total", top_n=10, compact=True)

if __name__ == "__main__":
    main()