import os
import gc
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
import cv2
import open_clip as clip
from PIL import Image
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor

# Import the profiler
from cognitive_bt_framework.utils.time_profiler import IterationTimeProfiler

@dataclass
class FastSAMConfig:
    """Configuration settings for FastSAM with CLIP filtering"""
    model_type: str = "FastSAM-s"  # FastSAM-x has better detection than FastSAM-s
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory management
    max_image_size: int = 640
    enable_memory_efficient_attention: bool = False
    
    # FastSAM parameters - tuned for more detections
    conf_threshold: float = 0.4  # Lowered from 0.4 to get more initial detections
    iou_threshold: float = 0.75   # Lowered from 0.9 to allow more overlapping detections
    retina_masks: bool = True
    
    # CLIP parameters
    clip_model: str = "ViT-H-14-378-quickgelu"
    clip_threshold: float = 0.7  # Lowered from 0.85 to accept more matches
    
    # Post-processing parameters
    remove_small_regions: bool = False
    merge_overlapping: bool = False  # Disabled merging to keep more separate detections
    overlap_threshold: float = 0.5
    min_area: float = 1.0  # Lowered from 10.0 to keep smaller regions
    draw_borders: bool = True
    debug: bool = False
    run_time_profile: bool = True
    profile_mask_processing: bool = False
    save_intermediate_results: bool = False
    show_clip_scores: bool = False

class FastSAMWithCLIP:
    """
    Memory-optimized class for generating masks using FastSAM and filtering with CLIP
    """
    
    def __init__(self, config: FastSAMConfig):
        self.config = config
        
        # Initialize profiler only if profiling is enabled
        self.profiler = IterationTimeProfiler(enabled=config.run_time_profile)
        
        self._setup_memory_config()
        self._setup_models()
        self._next_mask_id = 1
        self.last_masks = []
        self.last_metadata = []

    def _setup_memory_config(self):
        """Configure memory management settings"""
        if self.config.run_time_profile:
            self.profiler.start("_setup_memory_config")
            
        try:
            if self.config.device == "cuda":
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
                
                if self.config.enable_memory_efficient_attention:
                    os.environ["PYTORCH_ENABLE_MEM_EFF_ATTENTION"] = "1"
                    
                if self.config.debug:
                    print("CUDA memory configuration set up")
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("_setup_memory_config")

    def _setup_models(self):
        """Initialize FastSAM and CLIP models"""
        if self.config.run_time_profile:
            self.profiler.start("_setup_models")
            
        try:
            if self.config.device == "cuda":
                if self.config.run_time_profile:
                    self.profiler.start("cuda_memory_clear")
                if self.config.debug:
                    print("Clearing CUDA cache before model initialization")
                torch.cuda.empty_cache()
                gc.collect()
                if self.config.run_time_profile:
                    self.profiler.stop("cuda_memory_clear")

            # Initialize FastSAM
            if self.config.run_time_profile:
                self.profiler.start("init_fastsam")
            if self.config.debug:
                print(f"Initializing FastSAM with model type: {self.config.model_type}")
                
            predictor_overrides = {
                "conf": self.config.conf_threshold,
                "task": "segment",
                "mode": "predict",
                "model": f"{self.config.model_type}.pt",
                "save": False,
                "imgsz": self.config.max_image_size
            }
            self.predictor = FastSAMPredictor(overrides=predictor_overrides)
            if self.config.run_time_profile:
                self.profiler.stop("init_fastsam")

            # Initialize CLIP
            if self.config.run_time_profile:
                self.profiler.start("init_clip")
            if self.config.debug:
                print(f"Initializing CLIP with model: {self.config.clip_model}")
                
            self.clip_model, _, self.clip_preprocess = clip.create_model_and_transforms(
                self.config.clip_model, 
                device=self.config.device
            )
            self.clip_tokenizer = clip.get_tokenizer(self.config.clip_model)
            if self.config.run_time_profile:
                self.profiler.stop("init_clip")

            if self.config.debug:
                print("Model initialization complete")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(
                    "GPU out of memory. Try reducing input image size "
                    "or adjusting thresholds."
                )
            raise e
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("_setup_models")

    @torch.no_grad()
    def get_clip_score(self, crop: Image.Image, query: str) -> float:
        """Calculate CLIP similarity score between image crop and text query"""
        if self.config.run_time_profile:
            self.profiler.start("get_clip_score")
            
        try:
            # Prepare text prompts
            if self.config.run_time_profile:
                self.profiler.start("prepare_text")
            text_inputs = self.clip_tokenizer(
                [f"{query}", "a photo of background"]
            ).to(self.config.device)
            if self.config.run_time_profile:
                self.profiler.stop("prepare_text")
            
            # Prepare image
            if self.config.run_time_profile:
                self.profiler.start("prepare_image")
            image_input = self.clip_preprocess(crop).unsqueeze(0).to(self.config.device)
            if self.config.run_time_profile:
                self.profiler.stop("prepare_image")

            # Get similarity scores
            if self.config.run_time_profile:
                self.profiler.start("clip_inference")
            image_features, text_features, logits_per_image = self.clip_model(image_input, text_inputs)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            if self.config.run_time_profile:
                self.profiler.stop("clip_inference")
                
            score = float(similarity[0][0].item())
            
            if self.config.debug and self.config.show_clip_scores:
                print(f"CLIP score for query '{query}': {score:.3f}")
                
            return score
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("get_clip_score")

    def process_image(
        self,
        image: Union[str, np.ndarray],
        query: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process image with FastSAM and optionally filter with CLIP query
        
        Args:
            image: Input image (path or numpy array)
            query: Optional text query for CLIP filtering
            
        Returns:
            Tuple[np.ndarray, Dict]: Labeled mask array and metadata dictionary
        """
        # Start a new profiling iteration
        if self.config.run_time_profile:
            self.profiler.start_iteration()
            self.profiler.start("process_image")
            
        try:
            if self.config.debug:
                if isinstance(image, str):
                    print(f"Processing image from file: {image}")
                else:
                    print(f"Processing image array with shape: {image.shape}")
                if query:
                    print(f"Using query: '{query}'")
            
            # Generate masks with FastSAM
            if self.config.run_time_profile:
                self.profiler.start("fastsam_inference")
            start = time.time()
            results = self.predictor(image)
            fastsam_time = time.time() - start
            
            # Only print the timing information if time profiling or debug is enabled
            if self.config.run_time_profile or self.config.debug:
                print(f"FastSAM inference time: {fastsam_time:.2f}s")
                
            if self.config.run_time_profile:
                self.profiler.stop("fastsam_inference")

            # Convert results to labeled masks and metadata
            if self.config.run_time_profile:
                self.profiler.start("process_results")
            labeled_masks, metadata = self._process_results(results, image, query)
            if self.config.run_time_profile:
                self.profiler.stop("process_results")
                
            self.last_masks = labeled_masks
            self.last_metadata = metadata
            
            if self.config.debug:
                print(f"Found {len(metadata)} masks after processing")
                
            return labeled_masks, metadata

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                error_msg = "Out of memory during processing. Try:\n" \
                          "1. Reducing input image size\n" \
                          "2. Adjusting confidence threshold\n" \
                          "3. Adjusting IoU threshold"
                if self.config.debug:
                    print(error_msg)
                raise RuntimeError(error_msg)
            raise e
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("process_image")
                # End the iteration but preserve the timing data for reporting
                self.profiler.end_iteration(preserve_current=True)
                
                # Print timing summary if explicitly enabled for profiling
                if self.config.run_time_profile:
                    self.print_performance_summary()

    def _process_results(
        self, 
        results, 
        original_image: Union[str, np.ndarray],
        query: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Process FastSAM results with fixed mask dimensions"""
        if self.config.run_time_profile:
            self.profiler.start("_process_results")
            
        try:
            # Convert image to numpy if needed
            if isinstance(original_image, str):
                if self.config.run_time_profile:
                    self.profiler.start("load_image")
                original_image = cv2.imread(original_image)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                if self.config.run_time_profile:
                    self.profiler.stop("load_image")

            height, width = original_image.shape[:2]
            labeled_masks = np.zeros((height, width), dtype=np.int32)
            metadata = {}

            # Print dimensions for debugging
            if self.config.debug:
                print(f"Original image dimensions: {original_image.shape}")
                if len(results) > 0 and hasattr(results[0], 'masks') and len(results[0].masks) > 0:
                    print(f"First mask dimensions: {results[0].masks[0].data.shape}")

            # Process each mask
            if self.config.run_time_profile:
                self.profiler.start("process_masks")
                
            mask_count = len(results[0].masks) if results and len(results) > 0 else 0
            processed_count = 0
            filtered_count = 0
            
            if self.config.debug:
                print(f"Processing {mask_count} masks from FastSAM")
            
            for i, mask_data in enumerate(results[0].masks, 1):
                # Only profile individual masks if specifically enabled
                if self.config.run_time_profile and self.config.profile_mask_processing:
                    self.profiler.start(f"mask_{i}")
                    
                try:
                    # Get mask and convert to numpy
                    if self.config.run_time_profile:
                        self.profiler.start("mask_to_numpy")
                    mask = mask_data.data.cpu().numpy()[0]
                    mask = (mask > 0.5).astype(bool)
                    if self.config.run_time_profile:
                        self.profiler.stop("mask_to_numpy")
                        
                    if self.config.debug:
                        print(f"Processing mask {i}, original shape: {mask.shape}")
                    
                    # Resize mask to match original image dimensions
                    if mask.shape != (height, width):
                        if self.config.run_time_profile:
                            self.profiler.start("resize_mask")
                        mask = cv2.resize(
                            mask.astype(np.uint8), 
                            (width, height), 
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        if self.config.run_time_profile:
                            self.profiler.stop("resize_mask")
                    
                    if self.config.debug:
                        print(f"Mask {i} resized shape: {mask.shape}")
                    
                    if not mask.any():
                        if self.config.debug:
                            print(f"Mask {i} is empty after resize")
                        filtered_count += 1
                        continue
                        
                    area = float(mask.sum())
                    if area < self.config.min_area and self.config.remove_small_regions:
                        if self.config.debug:
                            print(f"Mask {i} too small: {area} < {self.config.min_area}")
                        filtered_count += 1
                        continue

                    # Get bounding box
                    if self.config.run_time_profile:
                        self.profiler.start("compute_bbox")
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) == 0 or len(x_indices) == 0:
                        if self.config.debug:
                            print(f"Mask {i} has no valid indices")
                        filtered_count += 1
                        continue
                        
                    x1, y1 = int(x_indices.min()), int(y_indices.min())
                    x2, y2 = int(x_indices.max()), int(y_indices.max())
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    if self.config.run_time_profile:
                        self.profiler.stop("compute_bbox")

                    # Apply CLIP filtering if query provided
                    clip_score = 1.0
                    if query:
                        if self.config.run_time_profile:
                            self.profiler.start("clip_filtering")
                            
                        # Ensure crop coordinates are within bounds
                        x1 = max(0, min(x1, width-1))
                        y1 = max(0, min(y1, height-1))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        
                        if x2 <= x1 or y2 <= y1:
                            if self.config.debug:
                                print(f"Invalid crop dimensions for mask {i}: ({x1},{y1}) to ({x2},{y2})")
                            filtered_count += 1
                            continue
                        
                        # Crop and convert to PIL
                        crop = original_image[y1:y2, x1:x2]
                        if crop.size == 0:
                            if self.config.debug:
                                print(f"Empty crop for mask {i}")
                            filtered_count += 1
                            continue
                            
                        crop_pil = Image.fromarray(crop)
                        
                        # Make crop square with padding
                        max_dim = max(crop.shape[0], crop.shape[1])
                        crop_square = Image.new('RGB', (max_dim, max_dim), (0,0,0))
                        paste_x = (max_dim - crop.shape[1]) // 2
                        paste_y = (max_dim - crop.shape[0]) // 2
                        crop_square.paste(crop_pil, (paste_x, paste_y))
                        
                        clip_score = self.get_clip_score(crop_square, query)
                        
                        if self.config.debug and self.config.show_clip_scores:
                            print(f"Mask {i} CLIP score: {clip_score:.3f}")
                        
                        if clip_score < self.config.clip_threshold:
                            if self.config.debug:
                                print(f"Mask {i} filtered by CLIP score: {clip_score:.3f} < {self.config.clip_threshold}")
                            filtered_count += 1
                            continue
                            
                        if self.config.run_time_profile:
                            self.profiler.stop("clip_filtering")
                    
                    # Add mask to the result
                    mask_id = self._next_mask_id
                    self._next_mask_id += 1
                    
                    labeled_masks[mask] = mask_id
                    metadata[mask_id] = {
                        'area': area,
                        'bbox': bbox,
                        'clip_score': clip_score
                    }
                    processed_count += 1
                    
                    if self.config.debug:    
                        print(f"Successfully processed mask {i} with ID {mask_id}")

                except Exception as e:
                    if self.config.debug:
                        print(f"Error processing mask {i}: {str(e)}")
                    continue
                finally:
                    if self.config.run_time_profile and self.config.profile_mask_processing:
                        self.profiler.stop(f"mask_{i}")
                        
            if self.config.run_time_profile:
                self.profiler.stop("process_masks")
                
            if self.config.debug:
                print(f"Total masks: {mask_count}, Processed: {processed_count}, Filtered: {filtered_count}")
            
            # Optionally merge overlapping masks
            if self.config.merge_overlapping and len(metadata) > 1:
                if self.config.run_time_profile:
                    self.profiler.start("merge_masks")
                if self.config.debug:
                    print("Merging overlapping masks...")
                labeled_masks, metadata = self._merge_overlapping_masks(labeled_masks, metadata)
                if self.config.debug:
                    print(f"After merging: {len(metadata)} masks remaining")
                if self.config.run_time_profile:
                    self.profiler.stop("merge_masks")
                
            # Save intermediate results if enabled
            if self.config.save_intermediate_results:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                if isinstance(original_image, str):
                    base_name = os.path.basename(original_image).split('.')[0]
                else:
                    base_name = "image"
                    
                # Save labeled masks
                np.save(f"{base_name}_masks_{timestamp}.npy", labeled_masks)
                
                # Save metadata
                import json
                # Convert metadata to serializable format
                serializable_metadata = {}
                for k, v in metadata.items():
                    serializable_metadata[str(k)] = {
                        'area': float(v['area']),
                        'bbox': [int(x) for x in v['bbox']],
                        'clip_score': float(v['clip_score']) if 'clip_score' in v else 1.0
                    }
                with open(f"{base_name}_metadata_{timestamp}.json", 'w') as f:
                    json.dump(serializable_metadata, f, indent=2)
                
                if self.config.debug:
                    print(f"Saved intermediate results with prefix {base_name}_{timestamp}")
                
            return labeled_masks, metadata
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("_process_results")

    # Rest of the class methods (visualize_masks, etc.) can be updated in a similar way
    # by separating debug print statements from profiling operations

    # Add profiler utility functions
    def print_performance_summary(self, sort_by="total", top_n=None, compact=True):
        """Print summary of performance timing"""
        if self.config.run_time_profile:
            print("\n=== FastSAM+CLIP Performance Summary ===")
            self.profiler.print_summary(sort_by=sort_by, top_n=top_n, compact=compact)
        else:
            print("Performance profiling is disabled. Create FastSAMWithCLIP with run_time_profile=True to enable.")
            
    def reset_profiler(self):
        """Reset the profiler timings"""
        if self.config.run_time_profile:
            self.profiler.reset()
            if self.config.debug:
                print("Performance profiler has been reset")
            
    def get_performance_hotspots(self, top_n=5):
        """
        Get the top N hotspots in performance
        
        Returns a dictionary with the top N slowest sections and their times
        """
        if not self.config.run_time_profile:
            return {"profiling_enabled": False}
            
        summary = self.profiler.get_summary(sort_by="current_total", top_n=top_n)
        
        hotspots = {
            "profiling_enabled": True,
            "current_iteration": self.profiler.iteration_count,
            "sections": []
        }
        
        for item in summary:
            hotspots["sections"].append({
                "name": item['section'],
                "current_time": item['current_total'],
                "avg_time": item['average'],
                "calls": item['current_calls']
            })
            
        return hotspots


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastSAM with CLIP filtering")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--query", default=None, help="Text query for CLIP filtering")
    parser.add_argument("--output", default="output.jpg", help="Output path")
    parser.add_argument("--model-type", default="FastSAM-x", 
                        choices=["FastSAM-s", "FastSAM-x"],
                        help="FastSAM model type")
    parser.add_argument("--conf-threshold", type=float, default=0.4,
                        help="Confidence threshold for FastSAM")
    parser.add_argument("--clip-threshold", type=float, default=0.95,
                        help="Similarity threshold for CLIP filtering")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = FastSAMConfig(
        model_type=args.model_type,
        conf_threshold=args.conf_threshold,
        clip_threshold=args.clip_threshold,
        run_time_profile=args.profile,
        debug=args.profile  # Enable debug logging when profiling
    )
    
    # Initialize processor
    processor = FastSAMWithCLIP(config)
    
    # Load image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    labeled_masks, metadata = processor.process_image(image, args.query)
    
    # Visualize results
    result = processor.visualize_masks(image, labeled_masks, metadata)
    
    # Save output
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_bgr)
    
    # Print performance summary if profiling is enabled
    if args.profile:
        processor.print_performance_summary(sort_by="current_total", compact=False)
    
    print(f"Processed image saved to {args.output}")
    if args.query:
        print(f"Found {len(metadata)} matches for query '{args.query}'")