import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2
import gc
import os
import time

@dataclass
class SAM2MaskConfig:
    """Configuration settings for SAM2-based mask generation"""
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint_path: str = "sam2.1_hiera_large.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory management
    max_image_size: Optional[int] = 1024  # Resize large images
    points_per_batch: int = 64  # Reduced from 128 for memory
    enable_memory_efficient_attention: bool = False
    enable_torch_compile: bool = True  # Enable for potential speedup if enough memory
    
    # Mask generation parameters
    points_per_side: int = 32  # Reduced from 64 for memory
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.92
    stability_score_offset: float = 0.7
    crop_n_layers: int = 1
    box_nms_thresh: float = 0.7
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: float = 25.0
    use_m2m: bool = True
    
    # Post-processing parameters
    remove_small_regions: bool = False
    merge_overlapping: bool = False
    overlap_threshold: float = 0.5
    draw_borders: bool = True

class SAM2MaskGenerator:
    """
    Memory-optimized class for generating and managing masks using SAM2.
    Includes memory management strategies and batch processing.
    """
    
    def __init__(self, config: SAM2MaskConfig):
        """Initialize the SAM2-based mask generator"""
        self.config = config
        self._setup_memory_config()
        self._setup_model()
        self._mask_registry = {}
        self._next_mask_id = 1

    def _setup_memory_config(self):
        """Configure memory management settings"""
        # Set PyTorch memory allocator settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        if self.config.device == "cuda":
            # Enable gradient checkpointing for memory efficiency
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] += ",max_split_size_mb:512"
            
            # Enable memory efficient attention if configured
            if self.config.enable_memory_efficient_attention:
                os.environ["PYTORCH_ENABLE_MEM_EFF_ATTENTION"] = "1"

    def _setup_model(self):
        """Initialize and configure the SAM2 model with memory optimizations"""
        try:
            # Clear CUDA cache before model initialization
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            # Configure device-specific settings
            if self.config.device == "cuda":
                torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            # Build SAM2 model
            self.sam2 = build_sam2(
                self.config.model_cfg,
                self.config.checkpoint_path,
                device=self.config.device,
                apply_postprocessing=False
            )

            # Optional: Use torch.compile for speedup if enabled
            if self.config.enable_torch_compile and self.config.device == "cuda":
                self.sam2 = torch.compile(self.sam2)

            # Initialize mask generator with memory-optimized settings
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2,)
            #     points_per_side=self.config.points_per_side,
            #     points_per_batch=self.config.points_per_batch,
            #     pred_iou_thresh=self.config.pred_iou_thresh,
            #     stability_score_thresh=self.config.stability_score_thresh,
            #     stability_score_offset=self.config.stability_score_offset,
            #     crop_n_layers=self.config.crop_n_layers,
            #     box_nms_thresh=self.config.box_nms_thresh,
            #     crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
            #     min_mask_region_area=self.config.min_mask_region_area,
            #     use_m2m=self.config.use_m2m
            # )

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(
                    "GPU out of memory. Try reducing points_per_batch, "
                    "points_per_side, or max_image_size in the configuration."
                )
            raise e

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image with resizing if needed
        
        Args:
            image (np.ndarray): Input RGB image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if self.config.max_image_size is None:
            return image
            
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.config.max_image_size:
            scale = self.config.max_image_size / max_dim
            new_h = int(h * scale)
            new_w = int(w * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        return image

    def generate_masks(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Generate masks for the input image using SAM2 with memory optimization
        
        Args:
            image (np.ndarray): Input RGB image
            
        Returns:
            Tuple[np.ndarray, Dict]: Labeled mask array and metadata dictionary
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            original_size = image.shape[:2]
            
            # Clear cache before generation
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            start_time = time.time()
            # Generate masks using SAM2
            sam_masks = self.mask_generator.generate(processed_image)
            print(f"Time taken for mask generation: {time.time() - start_time} seconds")
            # Create labeled mask array
            height, width = processed_image.shape[:2]
            labeled_masks = np.zeros((height, width), dtype=np.int32)
            metadata = {}
            
            # Process masks in batches to manage memory
            batch_size = 10
            for i in range(0, len(sam_masks), batch_size):
                batch_masks = sam_masks[i:i + batch_size]
                
                for mask_data in batch_masks:
                    mask_id = self._next_mask_id
                    self._next_mask_id += 1
                    
                    labeled_masks[mask_data['segmentation']] = mask_id
                    
                    # Store metadata
                    metadata[mask_id] = {
                        'area': float(mask_data['area']),
                        'bbox': mask_data['bbox'],
                        'predicted_iou': float(mask_data['predicted_iou']),
                        'stability_score': float(mask_data['stability_score']),
                        'point_coords': mask_data['point_coords'],
                        'crop_box': mask_data['crop_box']
                    }
                    
                    # Optional: Calculate contours
                    if self.config.draw_borders:
                        contours, _ = cv2.findContours(
                            mask_data['segmentation'].astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        metadata[mask_id]['contours'] = [
                            cv2.approxPolyDP(c, epsilon=0.01, closed=True).tolist()
                            for c in contours
                        ]

            # Post-process if configured
            if self.config.remove_small_regions or self.config.merge_overlapping:
                labeled_masks, metadata = self._post_process_masks(
                    labeled_masks, 
                    metadata
                )
            
            # Resize back to original size if needed
            if processed_image.shape[:2] != original_size:
                labeled_masks = cv2.resize(
                    labeled_masks,
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            return labeled_masks, metadata

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(
                    "Out of memory during mask generation. Try:\n"
                    "1. Reducing points_per_batch (currently {})\n"
                    "2. Reducing points_per_side (currently {})\n"
                    "3. Reducing max_image_size (currently {})\n"
                    "4. Using enable_memory_efficient_attention=True"
                    .format(
                        self.config.points_per_batch,
                        self.config.points_per_side,
                        self.config.max_image_size
                    )
                )
            raise e

    def _post_process_masks(
        self, 
        masks: np.ndarray, 
        metadata: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Post-process masks by removing small regions and merging overlapping masks"""
        if self.config.remove_small_regions:
            masks, metadata = self._remove_small_regions(masks, metadata)
            
        if self.config.merge_overlapping:
            masks, metadata = self._merge_overlapping_masks(masks, metadata)
            
        return masks, metadata
    
    def _remove_small_regions(
        self, 
        masks: np.ndarray, 
        metadata: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Remove masks smaller than min_mask_region_area"""
        new_masks = masks.copy()
        new_metadata = metadata.copy()
        
        for mask_id, meta in list(metadata.items()):
            if meta['area'] < self.config.min_mask_region_area:
                new_masks[masks == mask_id] = 0
                del new_metadata[mask_id]
                
        return new_masks, new_metadata
    
    def _merge_overlapping_masks(
        self, 
        masks: np.ndarray, 
        metadata: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Merge masks that have significant overlap"""
        new_masks = masks.copy()
        new_metadata = metadata.copy()
        
        mask_ids = list(metadata.keys())
        for i in range(len(mask_ids)):
            for j in range(i + 1, len(mask_ids)):
                id1, id2 = mask_ids[i], mask_ids[j]
                
                if id1 not in new_metadata or id2 not in new_metadata:
                    continue
                    
                mask1 = masks == id1
                mask2 = masks == id2
                
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                # print('--------')
                # print(union)
                # print(intersection)
                # print('---------')
                iou = intersection / union
                
                if iou > self.config.overlap_threshold:
                    score1 = metadata[id1]['stability_score']
                    score2 = metadata[id2]['stability_score']
                    
                    if score1 >= score2:
                        new_masks[mask2] = id1
                        del new_metadata[id2]
                    else:
                        new_masks[mask1] = id2
                        del new_metadata[id1]
                        
        return new_masks, new_metadata

    def visualize_masks(
        self, 
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        alpha: float = 0.5,
        font_scale: float = 0.5,
        font_thickness: int = 1
    ) -> np.ndarray:
        """
        Create a visualization of the masks overlaid on the image with ID labels
        
        Args:
            image (np.ndarray): Original RGB image
            masks (np.ndarray): Labeled mask array
            metadata (Dict): Mask metadata dictionary
            alpha (float): Transparency of the masks
            font_scale (float): Scale of the font for ID labels
            font_thickness (int): Thickness of the font for ID labels
            
        Returns:
            np.ndarray: Visualization image with overlaid masks and labels
        """
        vis_image = image.copy()
        unique_masks = np.unique(masks)[1:]  # Skip 0 (background)
        
        # Create blank overlay for all masks
        overlay = np.zeros_like(vis_image, dtype=np.float32)
        
        for mask_id in unique_masks:
            if mask_id not in metadata:
                continue
                
            mask = masks == mask_id
            color = np.random.random(3) * 255
            
            # Add colored mask to overlay
            overlay[mask] = color
            
            # Calculate centroid for text placement
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                # Fallback to bbox center if moments fail
                bbox = metadata[mask_id]['bbox']  # [x, y, w, h]
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2
            
            # Draw ID label
            text = str(mask_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # Create text background
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            text_bg_x1 = text_x - 2
            text_bg_y1 = text_y - text_size[1] - 2
            text_bg_x2 = text_x + text_size[0] + 2
            text_bg_y2 = text_y + 2
            
            # Draw text background
            cv2.rectangle(
                vis_image,
                (text_bg_x1, text_bg_y1),
                (text_bg_x2, text_bg_y2),
                (255, 255, 255),
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                cv2.LINE_AA
            )
            
            # Draw borders if enabled
            if self.config.draw_borders:
                contours = metadata[mask_id].get('contours')
                if contours:
                    cv2.drawContours(
                        vis_image,
                        [np.array(c) for c in contours],
                        -1,
                        (0, 0, 255),
                        thickness=1
                    )
        
        # Blend overlay with original image
        vis_image = cv2.addWeighted(
            vis_image,
            1 - alpha,
            overlay.astype(np.uint8),
            alpha,
            0
        )
        
        return vis_image.astype(np.uint8)

    def show_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        figsize: Tuple[int, int] = (20, 20),
        font_size: int = 10
    ) -> None:
        """
        Display masks using matplotlib with ID labels
        
        Args:
            image (np.ndarray): Original RGB image
            masks (np.ndarray): Labeled mask array
            metadata (Dict): Mask metadata dictionary
            figsize (Tuple[int, int]): Figure size for matplotlib
            font_size (int): Size of the font for ID labels
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Show original image
        plt.imshow(image)
        
        # Create mask overlay
        unique_masks = np.unique(masks)[1:]  # Skip 0 (background)
        mask_img = np.zeros((*image.shape[:2], 4))
        
        for mask_id in unique_masks:
            if mask_id not in metadata:
                continue
                
            mask = masks == mask_id
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            mask_img[mask] = color_mask
            
            # Calculate centroid
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                # Fallback to bbox center
                bbox = metadata[mask_id]['bbox']
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
            
            # Add ID label
            plt.text(
                cx, cy,
                str(mask_id),
                color='white',
                fontsize=font_size,
                bbox=dict(
                    facecolor='black',
                    alpha=0.7,
                    edgecolor='none',
                    pad=1
                ),
                ha='center',
                va='center'
            )
            
            # Draw borders if enabled
            if self.config.draw_borders:
                contours = metadata[mask_id].get('contours')
                if contours:
                    plt.contour(
                        mask,
                        colors=['blue'],
                        alpha=0.4,
                        linewidths=1
                    )
        
        plt.imshow(mask_img)
        plt.axis('off')
        plt.show()

    def get_mask_metadata(self, mask_id: int) -> Optional[Dict]:
        """Retrieve metadata for a specific mask ID"""
        return self._mask_registry.get(mask_id)