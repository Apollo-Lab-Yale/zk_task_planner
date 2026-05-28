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
    conf_threshold: float = 0.2  # Lowered from 0.4 to get more initial detections
    iou_threshold: float = 0.4   # Lowered from 0.9 to allow more overlapping detections
    retina_masks: bool = True
    
    # CLIP parameters
    clip_model: str = "ViT-H-14-378-quickgelu"
    clip_threshold: float = 0.3  # Lowered from 0.85 to accept more matches
    
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

    def _run_segmentation(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float], Dict]:
        """
        Run FastSAM segmentation on the input image
        
        Args:
            image: RGB image to segment
            
        Returns:
            Tuple containing:
            - List of binary masks
            - List of confidence scores
            - Dictionary with additional metadata
        """
        if self.config.run_time_profile:
            self.profiler.start("_run_segmentation")
        
        try:
            if self.config.debug:
                print(f"Running FastSAM segmentation on image of shape {image.shape}")
            
            # Reset mask ID counter
            self._next_mask_id = 1
            
            # Check for empty or invalid image
            if image.size == 0 or image is None:
                if self.config.debug:
                    print("Empty or invalid image provided")
                return [], [], {}
                
            # Make sure image is RGB
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            
            # Run prediction with FastSAM
            if self.config.run_time_profile:
                self.profiler.start("predictor_predict")
                
            # The predictor is expecting BGR format for OpenCV compatibility
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get everything as masks without filtering
            # This sets everything_as_mask=True for the FastSAMPredictor
            results = self.predictor(
                bgr_image,
                device=self.config.device,
                retina_masks=self.config.retina_masks,
                imgsz=self.config.max_image_size,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=self.config.debug
            )
                
            if self.config.run_time_profile:
                self.profiler.stop("predictor_predict")
            
            # Extract masks and scores
            if self.config.run_time_profile:
                self.profiler.start("extract_masks")
                
            masks = []
            scores = []
            metadata = {}
            
            if len(results) > 0:
                # Check if there are segmentation masks
                if hasattr(results[0], 'masks') and results[0].masks is not None and len(results[0].masks) > 0:
                    if self.config.debug:
                        print(f"Found {len(results[0].masks)} masks in results")
                        
                    # Process each mask
                    for i, mask_data in enumerate(results[0].masks):
                        try:
                            # Convert tensor to numpy
                            if hasattr(mask_data, 'data'):
                                # Handle case where mask is a tensor
                                mask_np = mask_data.data.cpu().numpy()
                                if mask_np.shape[0] > 0:  # Check if mask is not empty
                                    mask_np = mask_np[0]  # Get first mask if multiple
                            else:
                                # Handle case where mask is already numpy
                                mask_np = mask_data
                                
                            # Convert to boolean mask
                            binary_mask = mask_np > 0.5
                            
                            # Skip empty masks
                            if not np.any(binary_mask):
                                continue
                                
                            # Resize mask to original image size if needed
                            if binary_mask.shape != image.shape[:2]:
                                binary_mask = cv2.resize(
                                    binary_mask.astype(np.uint8), 
                                    (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
                                
                            # Get confidence score if available
                            conf = 1.0
                            if hasattr(results[0], 'boxes') and len(results[0].boxes) > i:
                                if hasattr(results[0].boxes, 'conf') and len(results[0].boxes.conf) > i:
                                    conf = float(results[0].boxes.conf[i].cpu().numpy())
                            
                            # Add mask and score
                            masks.append(binary_mask)
                            scores.append(conf)
                            
                            # Add metadata
                            metadata[i+1] = {
                                'confidence': conf,
                                'area': float(np.sum(binary_mask))
                            }
                            
                            # Extract contours for visualization
                            if self.config.draw_borders:
                                contours, _ = cv2.findContours(
                                    binary_mask.astype(np.uint8), 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE
                                )
                                metadata[i+1]['contours'] = [c.reshape(-1, 2).tolist() for c in contours]
                                
                        except Exception as e:
                            if self.config.debug:
                                print(f"Error processing mask {i}: {str(e)}")
                            continue
                else:
                    # There are results but no masks, try to generate masks from bounding boxes
                    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                        if self.config.debug:
                            print(f"No masks found, generating from {len(results[0].boxes)} boxes")
                            
                        for i, box in enumerate(results[0].boxes.data):
                            try:
                                # Get box coordinates
                                x1, y1, x2, y2, conf, _ = box.cpu().numpy()
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                
                                # Create mask from box
                                mask = np.zeros(image.shape[:2], dtype=bool)
                                mask[y1:y2, x1:x2] = True
                                
                                masks.append(mask)
                                scores.append(float(conf))
                                
                                metadata[i+1] = {
                                    'confidence': float(conf),
                                    'area': float(np.sum(mask))
                                }
                                
                                if self.config.draw_borders:
                                    contours = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])]
                                    metadata[i+1]['contours'] = [c.reshape(-1, 2).tolist() for c in contours]
                                    
                            except Exception as e:
                                if self.config.debug:
                                    print(f"Error processing box {i}: {str(e)}")
                                continue
            
            if self.config.run_time_profile:
                self.profiler.stop("extract_masks")
            
            if self.config.debug:
                print(f"FastSAM returned {len(masks)} valid masks")
                
            # If no masks were found, create a default one covering the whole image
            if len(masks) == 0 and self.config.debug:
                print("No masks found - segmentation failed")
                
            return masks, scores, metadata
            
        except Exception as e:
            if self.config.debug:
                print(f"Error in _run_segmentation: {str(e)}")
                import traceback
                traceback.print_exc()
            return [], [], {}
            
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("_run_segmentation")
                
    def _filter_with_clip(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        scores: List[float],
        query: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        Filter segmentation masks using CLIP text-image matching
        
        Args:
            image: Original RGB image
            masks: List of binary masks from FastSAM
            scores: List of confidence scores from FastSAM
            query: Text query for CLIP filtering
            
        Returns:
            Tuple containing:
            - Labeled mask array with each region assigned a unique ID
            - Metadata dictionary with clip scores and other information
        """
        if self.config.run_time_profile:
            self.profiler.start("_filter_with_clip")
        
        try:
            height, width = image.shape[:2]
            labeled_masks = np.zeros((height, width), dtype=np.int32)
            metadata = {}
            
            if self.config.debug:
                print(f"Filtering {len(masks)} masks with CLIP query: '{query}'")
            
            # Process each mask
            for i, mask in enumerate(masks):
                if self.config.run_time_profile:
                    self.profiler.start(f"process_mask_{i}")
                
                try:
                    # Skip empty masks
                    if not np.any(mask):
                        if self.config.debug:
                            print(f"Mask {i} is empty, skipping")
                        continue
                    
                    # Check mask area
                    area = float(np.sum(mask))
                    if area < self.config.min_area:
                        if self.config.debug:
                            print(f"Mask {i} too small ({area} px), skipping")
                        continue
                    
                    # Get bounding box for cropping
                    y_indices, x_indices = np.where(mask)
                    x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))
                    x2, y2 = int(np.max(x_indices)), int(np.max(y_indices))
                    
                    # Ensure bbox is valid
                    if x2 <= x1 or y2 <= y1 or x1 >= width or y1 >= height:
                        if self.config.debug:
                            print(f"Mask {i} has invalid bbox, skipping")
                        continue
                    
                    # Crop image to mask bbox
                    crop = image[y1:y2, x1:x2].copy()
                    
                    # Create mask for the cropped region
                    crop_mask = mask[y1:y2, x1:x2]
                    
                    # Apply mask to image (optional - focus CLIP on the object)
                    # Set background to black or gray to help CLIP focus on the foreground
                    if self.config.run_time_profile:
                        self.profiler.start(f"apply_mask_{i}")
                    masked_crop = crop.copy()
                    # Set background (non-mask) pixels to gray (128, 128, 128)
                    masked_crop[~crop_mask] = 128
                    if self.config.run_time_profile:
                        self.profiler.stop(f"apply_mask_{i}")
                    
                    # Convert to PIL for CLIP
                    crop_pil = Image.fromarray(masked_crop)
                    
                    # Get CLIP score
                    if self.config.run_time_profile:
                        self.profiler.start(f"clip_score_{i}")
                    clip_score = self.get_clip_score(crop_pil, query)
                    if self.config.run_time_profile:
                        self.profiler.stop(f"clip_score_{i}")
                    
                    if self.config.debug and self.config.show_clip_scores:
                        print(f"Mask {i}: CLIP score = {clip_score:.4f}")
                    
                    # Filter by CLIP threshold
                    if clip_score < self.config.clip_threshold:
                        if self.config.debug:
                            print(f"Mask {i} rejected: CLIP score {clip_score:.4f} < threshold {self.config.clip_threshold}")
                        continue
                    
                    # Add to labeled masks
                    mask_id = self._next_mask_id
                    self._next_mask_id += 1
                    
                    labeled_masks[mask] = mask_id
                    
                    # Create metadata
                    meta_entry = {
                        'clip_score': clip_score,
                        'confidence': scores[i] if i < len(scores) else 1.0,
                        'area': area,
                        'bbox': [x1, y1, x2-x1, y2-y1]
                    }
                    
                    # Add contours for visualization if enabled
                    if self.config.draw_borders:
                        if self.config.run_time_profile:
                            self.profiler.start(f"find_contours_{i}")
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        # Convert contours to lists for easier serialization
                        contour_lists = []
                        for contour in contours:
                            contour_lists.append(contour.reshape(-1, 2).tolist())
                        
                        meta_entry['contours'] = contour_lists
                        if self.config.run_time_profile:
                            self.profiler.stop(f"find_contours_{i}")
                    
                    metadata[mask_id] = meta_entry
                    
                    if self.config.debug:
                        print(f"Mask {i} accepted with ID {mask_id}: CLIP score {clip_score:.4f}")
                
                except Exception as e:
                    if self.config.debug:
                        print(f"Error processing mask {i}: {str(e)}")
                    continue
                finally:
                    if self.config.run_time_profile:
                        self.profiler.stop(f"process_mask_{i}")
            
            # Optionally merge overlapping masks
            if self.config.merge_overlapping and len(metadata) > 1:
                if self.config.debug:
                    print(f"Merging overlapping masks (before: {len(metadata)})")
                labeled_masks, metadata = self._merge_overlapping_masks(labeled_masks, metadata)
                if self.config.debug:
                    print(f"After merging: {len(metadata)} masks")
            
            return labeled_masks, metadata
        
        except Exception as e:
            if self.config.debug:
                print(f"Error in _filter_with_clip: {str(e)}")
                import traceback
                traceback.print_exc()
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.int32), {}
        
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("_filter_with_clip")
    
    def process_image(
        self,
        image: np.ndarray,
        query: str,
        roi_coords: Tuple[int, int, int, int] = None,
        store_results: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process an image using FastSAM and CLIP for semantic filtering
        
        Args:
            image: RGB image to process
            query: CLIP text query for filtering results
            roi_coords: Optional (x, y, w, h) coordinates if image is an ROI
            store_results: Whether to store results for later visualization
            
        Returns:
            Tuple containing:
            - Labeled mask array with each region assigned a unique ID
            - Metadata dictionary with clip scores and other information
        """
        if self.config.run_time_profile:
            self.profiler.start("process_image")
        
        try:
            # Run segmentation
            if self.config.run_time_profile:
                self.profiler.start("run_segmentation")
            
            masks, scores, metadata = self._run_segmentation(image)
            
            if self.config.run_time_profile:
                self.profiler.stop("run_segmentation")
            
            # Filter with CLIP if a query is provided
            if query:
                if self.config.run_time_profile:
                    self.profiler.start("filter_with_clip")
                
                labeled_masks, clip_metadata = self._filter_with_clip(image, masks, scores, query)
                
                if self.config.run_time_profile:
                    self.profiler.stop("filter_with_clip")
            else:
                # Without CLIP filtering, just use the raw masks
                if self.config.run_time_profile:
                    self.profiler.start("prepare_without_clip")
                
                labeled_masks = np.zeros(image.shape[:2], dtype=np.int32)
                clip_metadata = {}
                
                for i, mask in enumerate(masks):
                    mask_id = i + 1
                    labeled_masks[mask] = mask_id
                    clip_metadata[mask_id] = {
                        'confidence': scores[i] if i < len(scores) else 0.0,
                        'area': np.sum(mask)
                    }
                
                if self.config.run_time_profile:
                    self.profiler.stop("prepare_without_clip")
            
            # Store the results if requested
            if store_results:
                self.store_last_results(image, labeled_masks, clip_metadata, roi_coords)
            
            return labeled_masks, clip_metadata
            
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("process_image")
                    
    def visualize_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Visualize masks with labels and arrows"""
        if self.config.run_time_profile:
            self.profiler.start("visualize_masks")
        try:
            result = self._visualize_with_arrows(image, masks, metadata, alpha)
            return result
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("visualize_masks")
                
    def visualize_roi_masks(
        self,
        original_image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        roi_coords: Tuple[int, int, int, int] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize masks that were created from an ROI on the original full-sized image.
        
        Args:
            original_image: The original full-sized image
            masks: Labeled mask array from FastSAM
            metadata: Metadata dictionary
            roi_coords: (x, y, w, h) coordinates of the ROI in the original image
            alpha: Transparency of the overlay
            
        Returns:
            Visualization of the masks on the original image
        """
        if self.config.run_time_profile:
            self.profiler.start("visualize_roi_masks")
        
        try:
            # If roi_coords is not provided, use default visualization
            if roi_coords is None:
                return self.visualize_masks(original_image, masks, metadata, alpha)
            
            # Extract ROI coordinates
            x, y, w, h = roi_coords
            
            # Create a full-sized mask array initialized to zeros (background)
            full_masks = np.zeros(original_image.shape[:2], dtype=np.int32)
            
            # Check if the mask needs to be resized to match the ROI
            roi_shape = (h, w)
            mask_shape = masks.shape
            
            if roi_shape != mask_shape:
                # Resize the mask to match the ROI
                resized_masks = cv2.resize(
                    masks.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
                
                # Place the resized mask in the correct position in the full mask
                try:
                    full_masks[y:y+h, x:x+w] = resized_masks
                except ValueError as e:
                    # Handle potential dimension mismatch more gracefully
                    print(f"Error placing mask in full image: {e}")
                    print(f"ROI shape: {roi_shape}, Mask shape: {mask_shape}")
                    print(f"Resized mask shape: {resized_masks.shape}")
                    print(f"Full mask shape: {full_masks.shape}")
                    print(f"ROI coordinates: x={x}, y={y}, w={w}, h={h}")
                    
                    # Adjust dimensions if needed
                    h_to_use = min(h, resized_masks.shape[0], full_masks.shape[0] - y)
                    w_to_use = min(w, resized_masks.shape[1], full_masks.shape[1] - x)
                    
                    if h_to_use <= 0 or w_to_use <= 0:
                        return original_image.copy()  # Return original image if we can't place the mask
                    
                    full_masks[y:y+h_to_use, x:x+w_to_use] = resized_masks[:h_to_use, :w_to_use]
            else:
                # If mask already matches ROI size, place it directly
                full_masks[y:y+h, x:x+w] = masks
            
            # Update metadata to include ROI information
            roi_metadata = {}
            for mask_id, meta in metadata.items():
                roi_metadata[mask_id] = meta.copy()
                
                # If we have contours, adjust them to full image coordinates
                if 'contours' in meta:
                    adjusted_contours = []
                    for contour in meta['contours']:
                        adjusted_contour = [(cx + x, cy + y) for cx, cy in contour]
                        adjusted_contours.append(adjusted_contour)
                    roi_metadata[mask_id]['contours'] = adjusted_contours
            
            # Visualize the masks using the standard method
            return self._visualize_with_arrows(original_image, full_masks, roi_metadata, alpha)
        
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("visualize_roi_masks")
                
    def store_last_results(self, image: np.ndarray, masks: np.ndarray, metadata: Dict, roi_coords: Tuple[int, int, int, int] = None):
        """
        Store the results of the last segmentation for later visualization
        
        Args:
            image: The image that was segmented
            masks: Labeled mask array from FastSAM
            metadata: Metadata dictionary
            roi_coords: (x, y, w, h) coordinates if processing an ROI
        """
        self.last_image = image.copy()
        self.last_masks = masks.copy()
        self.last_metadata = metadata.copy()
        self.last_roi_coords = roi_coords
        
    def get_last_visualization(self, original_image: np.ndarray = None, alpha: float = 0.5):
        """
        Get visualization of the last segmentation results with robust error handling
        
        Args:
            original_image: Optional full image to use instead of the stored one
            alpha: Transparency for the visualization
            
        Returns:
            Visualization image
        """
        if self.config.run_time_profile:
            self.profiler.start("get_last_visualization")
            
        try:
            # Check if we have last masks and metadata
            if not hasattr(self, 'last_masks') or self.last_masks is None:
                if self.config.debug:
                    print("No last masks available for visualization")
                    
                # Return original image with a text overlay
                if original_image is not None:
                    result = original_image.copy()
                    cv2.putText(
                        result,
                        "No segmentation available",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )
                    return result
                return None
            
            # Get image to use
            img = original_image if original_image is not None else self.last_image
            
            if img is None:
                if self.config.debug:
                    print("No image available for visualization")
                return None
            
            # Check if masks are valid
            if isinstance(self.last_masks, np.ndarray) and self.last_masks.size == 0:
                if self.config.debug:
                    print("Empty mask array")
                    
                # Return original image with a text overlay
                result = img.copy()
                cv2.putText(
                    result,
                    "No valid segmentation",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
                return result
            
            # Check if metadata is valid
            if not self.last_metadata or len(self.last_metadata) == 0:
                if self.config.debug:
                    print("No metadata available, creating simple overlay")
                    
                # Create a simple overlay without metadata
                result = img.copy()
                
                # Try to extract any masks
                try:
                    if isinstance(self.last_masks, np.ndarray):
                        unique_ids = np.unique(self.last_masks)
                        unique_ids = unique_ids[unique_ids > 0]  # Remove background
                        
                        if len(unique_ids) > 0:
                            overlay = np.zeros_like(img)
                            for mask_id in unique_ids:
                                mask = self.last_masks == mask_id
                                color = np.random.random(3) * 255
                                overlay[mask] = color
                                
                            result = cv2.addWeighted(
                                result,
                                1 - alpha,
                                overlay.astype(np.uint8),
                                alpha,
                                0
                            )
                            
                            cv2.putText(
                                result,
                                f"Found {len(unique_ids)} regions (no metadata)",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2
                            )
                        else:
                            cv2.putText(
                                result,
                                "No valid segmentation regions",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2
                            )
                except Exception as e:
                    if self.config.debug:
                        print(f"Error creating simple overlay: {e}")
                    
                    cv2.putText(
                        result,
                        "Error creating segmentation visualization",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )
                    
                return result
            
            # Use appropriate visualization method
            try:
                if hasattr(self, 'last_roi_coords') and self.last_roi_coords is not None:
                    return self.visualize_roi_masks(img, self.last_masks, self.last_metadata, self.last_roi_coords, alpha)
                else:
                    return self.visualize_masks(img, self.last_masks, self.last_metadata, alpha)
            except Exception as e:
                if self.config.debug:
                    print(f"Error in visualization: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # Create a simple fallback visualization
                result = img.copy()
                cv2.putText(
                    result,
                    f"Visualization error: {str(e)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
                return result
                
        except Exception as e:
            if self.config.debug:
                print(f"Error in get_last_visualization: {e}")
                import traceback
                traceback.print_exc()
                
            # Return original image if available
            if original_image is not None:
                result = original_image.copy()
                cv2.putText(
                    result,
                    "Visualization failed",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
                return result
            return None
            
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("get_last_visualization")

    def _visualize_with_arrows(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Helper method for mask visualization with arrows and labels"""
        if self.config.run_time_profile:
            self.profiler.start("_visualize_with_arrows")
            
        try:
            vis_image = image.copy()
            unique_masks = np.unique(masks)[1:]
            overlay = np.zeros_like(vis_image, dtype=np.float32)
            placed_boxes = []
            
            for mask_id in unique_masks:
                if self.config.run_time_profile:
                    self.profiler.start(f"visualize_mask_{mask_id}")
                try:
                    if mask_id not in metadata:
                        continue
                        
                    mask = masks == mask_id
                    color = np.random.random(3) * 255
                    overlay[mask] = color
                    
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        x_center = int(np.mean(x_coords))
                        y_center = int(np.mean(y_coords))
                        
                        label = f"r{mask_id}"
                        if 'clip_score' in metadata[mask_id]:
                            label += f" ({metadata[mask_id]['clip_score']:.2f})"
                        
                        text_size = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        text_x = x_center - text_size[0]//2
                        text_y = y_center + text_size[1]//2
                        
                        text_box = (
                            text_x - 5, 
                            text_y - text_size[1] - 5,
                            text_x + text_size[0] + 5, 
                            text_y + 5
                        )
                        
                        # Find non-overlapping position
                        offset_y = text_size[1] + 10
                        while any(self._overlaps(text_box, box) for box in placed_boxes):
                            text_y += offset_y
                            text_box = (
                                text_x - 5,
                                text_y - text_size[1] - 5,
                                text_x + text_size[0] + 5,
                                text_y + 5
                            )
                        
                        placed_boxes.append(text_box)
                        
                        # Draw arrow and text
                        cv2.arrowedLine(
                            vis_image,
                            (x_center, y_center),
                            (text_x + text_size[0]//2, text_y - text_size[1]//2),
                            (0, 0, 0),
                            2,
                            tipLength=0.2
                        )
                        
                        # Draw text with outline
                        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                            cv2.putText(
                                vis_image, 
                                label,
                                (text_x+dx, text_y+dy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,0,0),
                                2
                            )
                        cv2.putText(
                            vis_image,
                            label,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255,255,255),
                            2
                        )
                    
                    if self.config.draw_borders and 'contours' in metadata[mask_id]:
                        contours = metadata[mask_id]['contours']
                        cv2.drawContours(
                            vis_image,
                            [np.array(c) for c in contours],
                            -1,
                            (0, 0, 0),
                            thickness=2
                        )
                finally:
                    if self.config.run_time_profile:
                        self.profiler.stop(f"visualize_mask_{mask_id}")
            
            if self.config.run_time_profile:
                self.profiler.start("blend_overlay")
            vis_image = cv2.addWeighted(
                vis_image,
                1 - alpha,
                overlay.astype(np.uint8),
                alpha,
                0
            )
            if self.config.run_time_profile:
                self.profiler.stop("blend_overlay")
                
            return vis_image.astype(np.uint8)
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("_visualize_with_arrows")

    def _overlaps(self, box1, box2):
        """Check if two bounding boxes overlap"""
        return not (
            box1[2] < box2[0] or
            box1[0] > box2[2] or
            box1[3] < box2[1] or
            box1[1] > box2[3]
        )

    def _merge_overlapping_masks(
        self,
        masks: np.ndarray,
        metadata: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Merge masks with significant overlap"""
        if self.config.run_time_profile:
            self.profiler.start("_merge_overlapping_masks")
            
        try:
            new_masks = masks.copy()
            new_metadata = metadata.copy()
            
            mask_ids = list(metadata.keys())
            for i in range(len(mask_ids)):
                for j in range(i + 1, len(mask_ids)):
                    id1, id2 = mask_ids[i], mask_ids[j]
                    
                    if id1 not in new_metadata or id2 not in new_metadata:
                        continue
                    
                    if self.config.run_time_profile:
                        self.profiler.start(f"compare_masks_{id1}_{id2}")
                        
                    mask1 = masks == id1
                    mask2 = masks == id2
                    
                    intersection = np.logical_and(mask1, mask2).sum()
                    union = np.logical_or(mask1, mask2).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        
                        if iou > self.config.overlap_threshold:
                            # Keep mask with higher CLIP score if available
                            score1 = new_metadata[id1].get('clip_score', 0)
                            score2 = new_metadata[id2].get('clip_score', 0)
                            
                            if score1 >= score2:
                                new_masks[mask2] = id1
                                del new_metadata[id2]
                            else:
                                new_masks[mask1] = id2
                                del new_metadata[id1]
                                
                    if self.config.run_time_profile:
                        self.profiler.stop(f"compare_masks_{id1}_{id2}")
                        
            return new_masks, new_metadata
        finally:
            if self.config.run_time_profile:
                self.profiler.stop("_merge_overlapping_masks")

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