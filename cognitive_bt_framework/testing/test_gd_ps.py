import time
import cv2
import numpy as np
import argparse
import os
from datetime import datetime

DINO_CFG = '/home/liam/install/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
DINO_CHKPT = '/home/liam/dev/zk_task_planner/cognitive_bt_framework/weights/groundingdino_swint_ogc.pth'

# Import our modules
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision.ps_enhanced import EnhancedPerception

def get_color_for_score(score, max_detections=10):
    """Generate color based on detection score/rank for better visualization"""
    if score < 0:
        return (128, 128, 128)  # Gray for invalid detections
    
    # Create a color gradient from green (high score) to red (low score)
    # Normalize score to 0-1 range for color mapping
    normalized = max(0, min(1, score))
    
    if normalized > 0.7:
        return (0, 255, 0)      # Green for high confidence
    elif normalized > 0.4:
        return (0, 255, 255)    # Yellow for medium confidence  
    else:
        return (0, 128, 255)    # Orange for lower confidence

def draw_detections_with_smart_positioning(img, ranked_detections, max_display=None, min_score=0.1):
    """Draw detection boxes with smart text positioning to avoid overlap"""
    if max_display is None:
        max_display = len(ranked_detections)
    print(ranked_detections)
    displayed_count = 0
    text_positions = []  # Track text positions to avoid overlap
    
    for i, (score, box, phrase) in enumerate(ranked_detections):
        if displayed_count >= max_display:
            break
            
        if score < min_score:  # Skip very low confidence detections
            continue
            
        box = box.astype(int)
        color = get_color_for_score(score)
        
        # Draw bounding box with thickness based on confidence
        thickness = 3 if score > 0.7 else 2 if score > 0.4 else 1
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
        
        # Prepare text
        text = f"{phrase}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Calculate text position - try above box first, then below if overlap
        text_x = box[0]
        text_y = box[1] - 10
        
        # Check for overlap with existing text
        text_rect = (text_x, text_y - text_height, text_x + text_width, text_y + baseline)
        
        # If text would go above image bounds or overlap, put it below the box
        if text_y - text_height < 0 or any(rectangles_overlap(text_rect, pos) for pos in text_positions):
            text_y = box[3] + text_height + 5
            text_rect = (text_x, text_y - text_height, text_x + text_width, text_y + baseline)
            
            # If still overlapping or going below image, try to the right
            if text_y + baseline > img.shape[0] or any(rectangles_overlap(text_rect, pos) for pos in text_positions):
                text_x = box[2] + 5
                text_y = box[1] + text_height
                text_rect = (text_x, text_y - text_height, text_x + text_width, text_y + baseline)
        
        # Draw text background for better readability
        cv2.rectangle(img, (text_rect[0]-2, text_rect[1]-2), (text_rect[2]+2, text_rect[3]+2), (0, 0, 0), -1)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness)
        
        text_positions.append(text_rect)
        displayed_count += 1
    
    return displayed_count

def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap"""
    return not (rect1[2] < rect2[0] or rect2[2] < rect1[0] or rect1[3] < rect2[1] or rect2[3] < rect1[1])

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Enhanced Perception with RealSense Camera")
    parser.add_argument("--dino-config", default=DINO_CFG, help="Path to Grounding DINO config file")
    parser.add_argument("--dino-checkpoint", default=DINO_CHKPT, help="Path to Grounding DINO checkpoint file")
    parser.add_argument("--box-threshold", type=float, default=0.3, help="Box threshold for Grounding DINO")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text threshold for Grounding DINO")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--max-detections", type=int, default=None, help="Maximum number of detections to display (None for all)")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum score to display detections")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize RealSense camera
    camera = Camera(width=640, height=480, fps=30)
    if not camera.start():
        print("Failed to start camera. Exiting...")
        return
    
    # Initialize Enhanced Perception system
    perception = EnhancedPerception(
        grounding_dino_config_path=args.dino_config,
        grounding_dino_checkpoint_path=args.dino_checkpoint,
        clip_model_name=args.clip_model
    )
    
    print("Starting perception system test. Controls:")
    print(" - 'q': Quit the application")
    print(" - 'c': Capture and analyze the current frame")
    print(" - 's': Save the current frame and analysis results")
    print(" - 'b'/'n': Increase/decrease box threshold")
    print(" - 't'/'y': Increase/decrease text threshold")
    print(" - 'm'/'k': Increase/decrease minimum score threshold")
    print(" - 'p'/'l': Increase/decrease max detections displayed")
    print(" - 'v': Cycle through visualization modes (Color/Depth/Combined)")
    print(" - 'a': Toggle display all detections mode")
    
    # Initialize parameters
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    min_score = args.min_score
    max_detections = args.max_detections
    detailed_mode = False
    visualization_mode = 0  # 0: color, 1: depth, 2: combined
    show_all_detections = max_detections is None
    current_caption = "Press 'c' to analyze the scene"
    current_ranked = []
    save_current_frame = False
    
    # Benchmark data
    timings = {
        "caption": [],
        "detection": [],
        "ranking": [],
        "total": []
    }
    
    try:
        while True:
            loop_start_time = time.time()
            
            # Get frames from camera
            frames = camera.get_frames()
            if not frames:
                print("Failed to get frames. Continuing...")
                time.sleep(0.1)
                continue
            
            color_image, depth_image = frames
            
            # Get closest blob information
            closest_blob = camera.get_closest_blob()
            
            # Create a copy for visualization
            display_img = color_image.copy()
            
            # Convert to RGB for perception system
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # In detailed mode, perform full perception pipeline
            if detailed_mode or save_current_frame:
                # Generate caption for the scene
                caption_start = time.time()
                current_caption = perception.generate_caption(rgb_image)
                caption_time = time.time() - caption_start
                timings["caption"].append(caption_time)
                print(f"Scene caption: {current_caption} (Time: {caption_time:.3f}s)")
                
                # Detect objects using Grounding DINO
                detection_start = time.time()
                boxes, logits, phrases = perception.detect_objects_dino(
                    rgb_image, 
                    current_caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
                detection_time = time.time() - detection_start
                timings["detection"].append(detection_time)
                print(f"Detected {len(boxes)} objects (Time: {detection_time:.3f}s)")
                
                # Rank detections with CLIP
                ranking_start = time.time()
                current_ranked = perception.rank_detections_with_clip(
                    rgb_image,
                    boxes, 
                    phrases, 
                    target_query=current_caption
                )
                ranking_time = time.time() - ranking_start
                timings["ranking"].append(ranking_time)
                print(f"Ranked detections (Time: {ranking_time:.3f}s)")
                
                total_time = caption_time + detection_time + ranking_time
                timings["total"].append(total_time)
                print(f"Total perception time: {total_time:.3f}s")
                
                # Print all detected objects
                print("\nAll detected objects:")
                for i, (score, box, phrase) in enumerate(current_ranked):
                    if score >= min_score:
                        print(f"[{i+1}] {phrase}: {score:.3f}")
                
                detailed_mode = False
            
            # Draw detection boxes with smart positioning
            displayed_count = draw_detections_with_smart_positioning(
                display_img, 
                current_ranked, 
                max_display=max_detections if not show_all_detections else None,
                min_score=min_score
            )
            
            # Add caption to the image
            cv2.putText(display_img, f"Caption: {current_caption}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show closest blob marker if available
            if closest_blob:
                distance, (x, y) = closest_blob
                cv2.circle(display_img, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(display_img, f"{distance:.2f}m", (x+15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add current parameters to the display
            info_y_start = display_img.shape[0] - 120
            cv2.putText(display_img, f"Box Thresh: {box_threshold:.2f}", (10, info_y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"Text Thresh: {text_threshold:.2f}", (10, info_y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"Min Score: {min_score:.2f}", (10, info_y_start + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            max_display_text = "All" if show_all_detections else str(max_detections)
            cv2.putText(display_img, f"Max Display: {max_display_text}", (10, info_y_start + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"Showing: {displayed_count}/{len(current_ranked)}", (10, info_y_start + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Create depth colormap
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Create combined visualization
            combined_img = None
            if visualization_mode == 0:
                # Show the color image with overlays
                cv2.imshow("Perception System Test", display_img)
            elif visualization_mode == 1:
                # Show depth image
                cv2.imshow("Perception System Test", depth_colormap)
            else:  # visualization_mode == 2
                # Create a combined visualization
                alpha = 0.7
                combined_img = cv2.addWeighted(display_img, alpha, depth_colormap, 1-alpha, 0)
                cv2.imshow("Perception System Test", combined_img)
            
            # Save current frame and analysis if requested
            if save_current_frame:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save color image
                color_path = os.path.join(args.output_dir, f"color_{timestamp}.png")
                cv2.imwrite(color_path, color_image)
                
                # Save depth image
                depth_path = os.path.join(args.output_dir, f"depth_{timestamp}.png")
                cv2.imwrite(depth_path, depth_colormap)
                
                # Save visualization
                if combined_img is not None:
                    vis_path = os.path.join(args.output_dir, f"combined_{timestamp}.png")
                    cv2.imwrite(vis_path, combined_img)
                else:
                    vis_path = os.path.join(args.output_dir, f"visualization_{timestamp}.png")
                    cv2.imwrite(vis_path, display_img)
                
                # Save analysis results
                results_path = os.path.join(args.output_dir, f"analysis_{timestamp}.txt")
                with open(results_path, 'w') as f:
                    f.write(f"Caption: {current_caption}\n\n")
                    f.write(f"Total detections: {len(current_ranked)}\n")
                    f.write(f"Displayed detections: {displayed_count}\n")
                    f.write(f"Min score threshold: {min_score}\n\n")
                    f.write("All Detections (ranked by score):\n")
                    for i, (score, box, phrase) in enumerate(current_ranked):
                        status = "DISPLAYED" if score >= min_score and (show_all_detections or i < max_detections) else "HIDDEN"
                        f.write(f"[{i+1}] {phrase} (Score: {score:.3f}, Box: {box}) - {status}\n")
                    
                    if closest_blob:
                        distance, (x, y) = closest_blob
                        f.write(f"\nClosest point: ({x}, {y}) at {distance:.3f}m\n")
                    
                    f.write("\nTimings:\n")
                    f.write(f"Caption generation: {timings['caption'][-1]:.3f}s\n")
                    f.write(f"Object detection: {timings['detection'][-1]:.3f}s\n")
                    f.write(f"Detection ranking: {timings['ranking'][-1]:.3f}s\n")
                    f.write(f"Total processing: {timings['total'][-1]:.3f}s\n")
                
                print(f"Saved analysis results to {args.output_dir}")
                save_current_frame = False
            
            # Calculate FPS
            frame_time = time.time() - loop_start_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture and analyze current frame
                detailed_mode = True
                print("Analyzing current frame...")
            elif key == ord('s'):
                # Save current frame and analysis
                save_current_frame = True
                print("Saving current frame and analysis...")
            elif key == ord('b'):
                # Increase box threshold
                box_threshold = min(1.0, box_threshold + 0.05)
                print(f"Box threshold increased to {box_threshold:.2f}")
            elif key == ord('n'):
                # Decrease box threshold
                box_threshold = max(0.05, box_threshold - 0.05)
                print(f"Box threshold decreased to {box_threshold:.2f}")
            elif key == ord('t'):
                # Increase text threshold
                text_threshold = min(1.0, text_threshold + 0.05)
                print(f"Text threshold increased to {text_threshold:.2f}")
            elif key == ord('y'):
                # Decrease text threshold
                text_threshold = max(0.05, text_threshold - 0.05)
                print(f"Text threshold decreased to {text_threshold:.2f}")
            elif key == ord('m'):
                # Increase minimum score
                min_score = min(1.0, min_score + 0.05)
                print(f"Minimum score increased to {min_score:.2f}")
            elif key == ord('k'):
                # Decrease minimum score
                min_score = max(0.0, min_score - 0.05)
                print(f"Minimum score decreased to {min_score:.2f}")
            elif key == ord('p'):
                # Increase max detections
                if not show_all_detections:
                    max_detections = (max_detections or 5) + 5
                    print(f"Max detections increased to {max_detections}")
            elif key == ord('l'):
                # Decrease max detections
                if not show_all_detections and max_detections and max_detections > 5:
                    max_detections = max_detections - 5
                    print(f"Max detections decreased to {max_detections}")
            elif key == ord('a'):
                # Toggle show all detections
                show_all_detections = not show_all_detections
                if show_all_detections:
                    print("Now showing ALL detections")
                else:
                    max_detections = max_detections or 10
                    print(f"Now showing max {max_detections} detections")
            elif key == ord('v'):
                # Cycle through visualization modes
                visualization_mode = (visualization_mode + 1) % 3
                modes = ["Color", "Depth", "Combined"]
                print(f"Switched to {modes[visualization_mode]} visualization mode")
                
  
        
    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print benchmark statistics
        if timings["total"]:
            print("\nBenchmark Statistics:")
            for key, values in timings.items():
                if values:
                    avg_time = sum(values) / len(values)
                    max_time = max(values)
                    min_time = min(values)
                    print(f"{key.capitalize()} Time - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        
        print("Test completed.")

if __name__ == "__main__":
    main()