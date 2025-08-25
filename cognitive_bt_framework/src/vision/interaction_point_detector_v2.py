import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import os
import time
from collections import defaultdict

# Try to import PCL; provide fallback if not available
try:
    import pcl
    import pcl.pcl_visualization
    HAS_PCL = True
except ImportError:
    print("Warning: python-pcl not found. Using fallback geometric processing.")
    HAS_PCL = False

# Try to import Open3D as an alternative to PCL
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

class InteractionType(Enum):
    GRASP_EDGE = "grasp_edge"          # Corners, edges suitable for grasping
    GRASP_SURFACE = "grasp_surface"    # Flat surfaces for palm/finger contact
    PUSH_POINT = "push_point"          # Stable points for pushing
    HANDLE = "handle"                  # Detected handles or graspable features
    PIVOT = "pivot"                    # Points suitable for rotation/pivoting
    CONTACT = "contact"                # General stable contact points


@dataclass
class InteractionPoint:
    x: int
    y: int
    score: float
    interaction_type: InteractionType
    confidence: float
    approach_angle: Optional[float] = None
    grasp_width: Optional[float] = None
    stability: float = 0.5
    accessibility: float = 0.5


# Simplified DFormer-Tiny model for RGB-D fusion
class DFormerTiny(nn.Module):
    def __init__(self, num_interaction_types: int = 6):
        super().__init__()
        # Basic implementation of the DFormer-Tiny architecture
        # In a real implementation, you would load this from a pre-trained model
        
        # RGB encoder
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Depth encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        
        # Interaction type predictors - one for each interaction type
        self.interaction_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1)
            ) for _ in range(num_interaction_types)
        ])
    
    def forward(self, rgb, depth):
        # Extract features
        rgb_feats = self.rgb_encoder(rgb)
        depth_feats = self.depth_encoder(depth)
        
        # Reshape for attention
        b, c, h, w = rgb_feats.shape
        rgb_feats_flat = rgb_feats.flatten(2).permute(0, 2, 1)
        depth_feats_flat = depth_feats.flatten(2).permute(0, 2, 1)
        
        # Cross-modal attention
        attn_output, _ = self.cross_attn(
            query=rgb_feats_flat,
            key=depth_feats_flat,
            value=depth_feats_flat
        )
        
        # Reshape back
        fused_feats = attn_output.permute(0, 2, 1).reshape(b, c, h, w)
        
        # Generate attention maps for each interaction type
        attention_maps = []
        for predictor in self.interaction_predictors:
            # Each predictor outputs a heatmap indicating regions of interest
            attn_map = predictor(fused_feats)
            attn_map = F.interpolate(attn_map, scale_factor=4, mode='bilinear', align_corners=False)
            attention_maps.append(attn_map.squeeze(1))
        
        return {
            "attention_maps": attention_maps,
            "feature_maps": fused_feats
        }


class RobustInteractionDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_cuda: bool = True,
        quantize: bool = True,
        min_distance: int = 40,  # Increased from 20 to 40 for better filtering
        debug: bool = True,  # Enable debug by default to track issues
        cascade_enabled: bool = True,
        resolution_scale: float = 1.0,
    ):
        """
        Enhanced Interaction Point Detector combining geometric processing with lightweight ML.
        
        Args:
            model_path: Path to DFormer-Tiny weights (optional)
            use_cuda: Whether to use GPU acceleration
            quantize: Whether to use INT8 quantization to reduce memory usage
            min_distance: Minimum distance between interaction points
            debug: Enable debug outputs
            cascade_enabled: Enable cascade processing for early termination
            resolution_scale: Scale factor for input resolution (0.5-1.0)
        """
        self.debug = debug
        self.min_distance = min_distance
        self.cascade_enabled = cascade_enabled
        self.resolution_scale = np.clip(resolution_scale, 0.25, 1.0)
        
        # Initialize CUDA settings
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            
        # Initialize the model
        self.model = self._initialize_model(model_path, quantize)
        
        # Cascade processing configuration
        self.cascade_config = {
            "geometric_confidence_threshold": 0.75,  # Skip RGB processing if geometry is confident
            "max_texture_complexity": 0.6,  # Avoid points in high-texture regions
            "min_edge_strength": 0.02,  # Minimum depth edge strength
            "max_points": 30,  # Maximum number of candidate points from each method
        }
        
        # Processing timers for performance analysis
        self.timers = defaultdict(float)
        self.timer_counts = defaultdict(int)
        
        if self.debug:
            print(f"Enhanced Interaction Detector initialized:")
            print(f"  - CUDA: {'Available' if self.use_cuda else 'Not available'}")
            print(f"  - PCL: {'Available' if HAS_PCL else 'Not available'}")
            print(f"  - Open3D: {'Available' if HAS_OPEN3D else 'Not available'}")
            print(f"  - Resolution scale: {self.resolution_scale}")
            print(f"  - Cascade processing: {'Enabled' if self.cascade_enabled else 'Disabled'}")

    def _initialize_model(self, model_path: Optional[str], quantize: bool) -> nn.Module:
        """Initialize and load the DFormer-Tiny model."""
        model = DFormerTiny(num_interaction_types=len(InteractionType))
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            if self.debug:
                print(f"Loading model weights from {model_path}")
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Using randomly initialized weights instead.")
        else:
            if self.debug and model_path:
                print(f"Model weights file {model_path} not found. Using random initialization.")
                
        # Apply quantization if requested
        if quantize:
            if self.debug:
                print("Applying INT8 quantization to reduce memory usage")
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            
        # Move model to GPU if available
        if self.use_cuda:
            model = model.cuda()
            
        # Set to evaluation mode
        model.eval()
        return model

    def detect_interaction_points(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        min_distance: int = None,
        obj_info: Any = None,
        max_points: int = 15,
        depth_data: np.ndarray = None,
        apply_center_shift: bool = True,
        edge_threshold: float = 10.0,
        shift_factor: float = 0.4,
        fast_mode: bool = True,
        depth_plane_only: bool = False
    ) -> List[InteractionPoint]:
        """
        Main entry point to detect interaction points for manipulation tasks.
        
        Args:
            image: RGB image of the object
            mask: Binary mask of the object
            min_distance: Minimum distance between points (pixels)
            obj_info: Optional object metadata
            max_points: Maximum number of points to return
            depth_data: Depth image of the scene
            apply_center_shift: Whether to shift points toward center
            edge_threshold: Threshold for edge detection
            shift_factor: Factor for center shifting
            fast_mode: Enable optimizations
            depth_plane_only: Use only depth plane detection
            
        Returns:
            List of interaction points for manipulation
        """
        if not np.any(mask):
            return []
        
        # Use provided min_distance or default
        if min_distance is not None:
            effective_min_distance = min_distance
        else:
            effective_min_distance = self.min_distance
            
        # Start timing
        start_time = time.time()
        
        # Use depth_data parameter
        depth = depth_data
        
        if self.debug:
            mask_area = np.sum(mask)
            has_depth = depth is not None
            print(f"\n[DEBUG V2] STARTING INTERACTION POINT DETECTION")
            print(f"[DEBUG V2] Object mask area: {mask_area} pixels")
            print(f"[DEBUG V2] Has depth data: {has_depth}")
            print(f"[DEBUG V2] Max points: {max_points}, Min distance: {effective_min_distance}")
            print(f"[DEBUG V2] Depth plane only: {depth_plane_only}")
            if has_depth:
                valid_depths = depth[mask & (depth > 0)]
                if len(valid_depths) > 0:
                    depth_min, depth_max = np.min(valid_depths), np.max(valid_depths)
                    depth_range = depth_max - depth_min
                    print(f"[DEBUG V2] Depth range: {depth_min:.3f}m to {depth_max:.3f}m (range: {depth_range:.3f}m)")
                else:
                    print(f"[DEBUG V2] WARNING: No valid depth values in mask!")
        
        # Resize inputs if scaling is applied
        if self.resolution_scale < 1.0:
            h, w = image.shape[:2]
            new_h, new_w = int(h * self.resolution_scale), int(w * self.resolution_scale)
            image_resized = cv2.resize(image, (new_w, new_h))
            if depth is not None:
                depth_resized = cv2.resize(depth, (new_w, new_h))
            else:
                depth_resized = None
            mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h)) > 0
            
            # Save the scale factor for converting back to original coordinates
            self.scale_factor = 1.0 / self.resolution_scale
        else:
            image_resized = image
            depth_resized = depth
            mask_resized = mask
            self.scale_factor = 1.0
            
        # TIER 1: Fast geometric processing
        t1 = time.time()
        if self.debug:
            print("DEBUG: Starting geometric feature detection...")
        geometric_points = self._detect_geometric_features(depth_resized, mask_resized)
        self.timers["geometric"] += time.time() - t1
        self.timer_counts["geometric"] += 1
        
        if self.debug:
            print(f"[DEBUG V2] Geometric detection found {len(geometric_points)} points")
            if len(geometric_points) == 0:
                print(f"[DEBUG V2] WARNING: No geometric points found - check protrusion/indentation detection!")
        
        # Early termination if confident geometric features found and cascade is enabled
        if self.cascade_enabled and not depth_plane_only:
            high_confidence_points = [p for p in geometric_points 
                                     if p.confidence > self.cascade_config["geometric_confidence_threshold"]]
            if len(high_confidence_points) >= max_points:
                # Scale back to original coordinates
                if self.scale_factor != 1.0:
                    for p in high_confidence_points:
                        p.x = int(p.x * self.scale_factor)
                        p.y = int(p.y * self.scale_factor)
                
                final_points = self._apply_nms(high_confidence_points, effective_min_distance)[:max_points]
                
                if self.debug:
                    print(f"Early termination with {len(final_points)} high-confidence geometric points")
                    total_time = time.time() - start_time
                    print(f"Total processing time: {total_time:.3f}s")
                
                return final_points
        
        # TIER 2: RGB-D fusion for challenging regions  
        if not depth_plane_only and depth_resized is not None:
            t2 = time.time()
            # Temporarily disable RGB-guided features to avoid CUDA memory issues
            if self.debug:
                print("DEBUG: Skipping RGB-guided feature detection (disabled to avoid CUDA memory issues)...")
            fusion_points = []  # self._detect_rgb_guided_features(image_resized, depth_resized, mask_resized)
            self.timers["fusion"] += time.time() - t2
            self.timer_counts["fusion"] += 1
            
            all_points = geometric_points + fusion_points
        else:
            all_points = geometric_points
            
        # Score and prioritize features
        t3 = time.time()
        scored_points = self._score_and_rank_points(all_points, image_resized, depth_resized, mask_resized)
        self.timers["scoring"] += time.time() - t3
        self.timer_counts["scoring"] += 1
        
        # Scale back to original coordinates
        if self.scale_factor != 1.0:
            for p in scored_points:
                p.x = int(p.x * self.scale_factor)
                p.y = int(p.y * self.scale_factor)
        
        # Apply non-maximum suppression and return top points
        if self.debug:
            print(f"[DEBUG V2] Before NMS: {len(scored_points)} points, min_distance: {effective_min_distance}")
            
        final_points = self._apply_nms(scored_points, effective_min_distance)[:max_points]
        
        if self.debug:
            total_time = time.time() - start_time
            points_filtered_by_nms = len(scored_points) - len(final_points)
            print(f"[DEBUG V2] NMS FILTERING: {points_filtered_by_nms} points filtered out")
            print(f"[DEBUG V2] After NMS: {len(final_points)} points (from {len(all_points)} total candidates)")
            if len(final_points) > 0:
                print(f"[DEBUG V2] Final point locations: {[(p.x, p.y) for p in final_points[:5]]}")
            else:
                print(f"[DEBUG V2] WARNING: NO FINAL POINTS! Check filtering parameters.")
            print(f"[DEBUG V2] Total processing time: {total_time:.3f}s\n")
            
            if self.timer_counts["geometric"] > 0:
                avg_geometric = self.timers["geometric"] / self.timer_counts["geometric"]
                print(f"Average geometric processing time: {avg_geometric:.3f}s")
            
            if self.timer_counts["fusion"] > 0:
                avg_fusion = self.timers["fusion"] / self.timer_counts["fusion"]
                print(f"Average fusion processing time: {avg_fusion:.3f}s")
            
            if self.timer_counts["scoring"] > 0:
                avg_scoring = self.timers["scoring"] / self.timer_counts["scoring"]
                print(f"Average scoring time: {avg_scoring:.3f}s")
                
        return final_points
    
    # ------------------- Geometric Feature Detection Methods -------------------
    
    def _detect_geometric_features(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Extract features using purely geometric methods."""
        points = []
        
        # Make sure depth is valid and normalized
        if depth is None or np.all(depth == 0):
            if self.debug:
                print("[DEBUG V2] FILTERED OUT: No valid depth data for geometric features")
            return []
        
        if self.debug:
            valid_depth_pixels = np.sum((depth > 0) & mask)
            total_mask_pixels = np.sum(mask)
            print(f"[DEBUG V2] Geometric features - Valid depth pixels: {valid_depth_pixels}/{total_mask_pixels}")
            
        # Apply median filter to reduce noise
        depth_filtered = cv2.medianBlur(depth.astype(np.float32), 5)
        depth_filtered[~mask] = 0
        
        # 1. Detect protrusions (handles, knobs, buttons)
        protrusion_points = self._detect_protrusions(depth_filtered, mask)
        points.extend(protrusion_points)
        if self.debug:
            print(f"DEBUG: Protrusion detection: {len(protrusion_points)} points")
        
        # 2. Detect indentations (buttons, recessed features)
        indentation_points = self._detect_indentations(depth_filtered, mask)
        points.extend(indentation_points)
        if self.debug:
            print(f"DEBUG: Indentation detection: {len(indentation_points)} points")
        
        # 3. Detect surface normals for flat surfaces (like bottle caps)
        if self.debug:
            print("DEBUG: Starting surface normal detection...")
        normal_points = self._detect_surface_normals(depth_filtered, mask)
        points.extend(normal_points)
        if self.debug:
            print(f"DEBUG: Surface normal detection: {len(normal_points)} points")
        
        # 4. Detect depth edges with adaptive thresholding
        edge_points = self._detect_depth_edges(depth_filtered, mask)
        points.extend(edge_points)
        if self.debug:
            print(f"DEBUG: Edge detection: {len(edge_points)} points")
        
        # 5. If PCL or Open3D is available, use more advanced geometric processing
        if HAS_PCL or HAS_OPEN3D:
            advanced_points = self._detect_advanced_geometric_features(depth_filtered, mask)
            points.extend(advanced_points)
        
        return points
    
    def _detect_protrusions(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect local protrusions that could be handles or knobs."""
        points = []
        
        if self.debug:
            print(f"[DEBUG V2] Starting protrusion detection...")
        
        # Multi-scale analysis
        kernel_sizes = [7, 11, 15]
        
        for kernel_size in kernel_sizes:
            # Local mean depth
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            local_mean = cv2.filter2D(depth, -1, kernel)
            
            # Find protrusions (where depth is greater than local mean)
            protrusion_map = depth - local_mean
            protrusion_map[~mask] = 0
            
            # Calculate local statistics for adaptive thresholding
            local_sq_mean = cv2.filter2D(depth**2, -1, kernel)
            local_var = local_sq_mean - local_mean**2
            local_std = np.sqrt(np.maximum(local_var, 0))
            
            # Adaptive threshold: protrusions more than 1.5 std + 5mm
            threshold = 1.5 * local_std + 0.005
            significant_protrusions = (protrusion_map > threshold) & mask
            
            if self.debug:
                protrusion_pixels = np.sum(significant_protrusions)
                if protrusion_pixels > 0:
                    max_protrusion = np.max(protrusion_map[significant_protrusions])
                    print(f"[DEBUG V2] Kernel {kernel_size}: {protrusion_pixels} protrusion pixels, max value: {max_protrusion:.4f}m")
                else:
                    max_overall = np.max(protrusion_map[mask]) if np.any(mask) else 0
                    avg_threshold = np.mean(threshold[mask]) if np.any(mask) else 0
                    print(f"[DEBUG V2] Kernel {kernel_size}: No protrusions found (max: {max_overall:.4f}m, avg threshold: {avg_threshold:.4f}m)")
            
            # Find local maxima
            local_maxima = self._find_local_maxima(protrusion_map, significant_protrusions, kernel_size//2)
            
            for y, x in zip(*np.where(local_maxima)):
                if self._is_stable_feature(depth, x, y, mask):
                    prominence = protrusion_map[y, x]
                    points.append(InteractionPoint(
                        x=int(x), y=int(y),
                        score=min(1.0, prominence * 20),
                        interaction_type=InteractionType.HANDLE,
                        confidence=0.9,
                        grasp_width=kernel_size * 2
                    ))
        
        return points[:self.cascade_config["max_points"]]
    
    def _detect_indentations(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect local indentations that could be buttons or recessed features."""
        points = []
        
        if self.debug:
            print(f"[DEBUG V2] Starting indentation detection...")
        
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        local_mean = cv2.filter2D(depth, -1, kernel)
        
        # Find depressions (where local mean is greater than depth)
        depression_map = local_mean - depth
        depression_map[~mask] = 0
        
        # Find significant depressions
        valid_depressions = depression_map[mask & (depression_map > 0)]
        
        if self.debug:
            total_depression_pixels = len(valid_depressions)
            if total_depression_pixels > 0:
                max_depression = np.max(valid_depressions)
                mean_depression = np.mean(valid_depressions)
                print(f"[DEBUG V2] Found {total_depression_pixels} depression pixels, max: {max_depression:.4f}m, mean: {mean_depression:.4f}m")
            else:
                print(f"[DEBUG V2] No depression pixels found!")
        
        if len(valid_depressions) > 0:
            threshold = np.percentile(valid_depressions, 85)
            significant_depressions = (depression_map > threshold) & mask
            
            if self.debug:
                significant_pixels = np.sum(significant_depressions)
                print(f"[DEBUG V2] Depression threshold: {threshold:.4f}m, significant pixels: {significant_pixels}")
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(significant_depressions.astype(np.uint8))
            
            if self.debug:
                print(f"[DEBUG V2] Found {num_labels-1} depression components")
            
            for label in range(1, num_labels):
                component = (labels == label)
                area = np.sum(component)
                
                if 20 < area < 5000:  # Button to handle-sized features
                    moments = cv2.moments(component.astype(np.uint8))
                    if moments["m00"] > 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        
                        depth_diff = depression_map[cy, cx]
                        
                        if self.debug:
                            print(f"[DEBUG V2]   Added indentation point at ({cx}, {cy}) with depth_diff: {depth_diff:.4f}m")
                        
                        points.append(InteractionPoint(
                            x=cx, y=cy,
                            score=min(1.0, depth_diff * 30),
                            interaction_type=InteractionType.PUSH_POINT,
                            confidence=0.85
                        ))
                else:
                    if self.debug:
                        print(f"[DEBUG V2]   Component {label} FILTERED OUT: area {area} not in range [21, 499]")
        
        if self.debug:
            original_count = len(points)
            max_allowed = self.cascade_config["max_points"]
            print(f"[DEBUG V2] Indentation detection complete: {original_count} points found")
            if original_count > max_allowed:
                print(f"[DEBUG V2] CASCADE FILTERING: Limiting to {max_allowed} points (filtered {original_count - max_allowed})")
        
        return points[:self.cascade_config["max_points"]]
    
    def _detect_surface_normals(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect distinct surface normal sections (like bottle caps, flat surfaces)."""
        points = []
        
        if np.all(depth == 0) or not np.any(mask):
            return []
        
        # Parameters for surface normal detection
        patch_size = 9  # Size of local patch for normal computation
        min_patch_area = 50  # Minimum pixels for a valid surface patch
        normal_threshold = 0.1  # Very low threshold to catch slightly angled surfaces
        
        # Compute surface normals for the entire depth image
        normals = self._compute_surface_normals(depth, mask, patch_size)
        if normals is None:
            if self.debug:
                print("DEBUG: Surface normal computation failed")
            return []
        
        if self.debug:
            valid_normals = np.any(normals != 0, axis=2)
            print(f"DEBUG: Computed {np.sum(valid_normals)} valid surface normals")
            
        # Find regions with consistent surface normals
        # Focus on normals pointing toward camera (good for top surfaces like caps)
        camera_normal = np.array([0, 0, -1])  # Points toward camera
        
        # Calculate dot product with camera normal
        # normals is (H, W, 3), we need to compute dot product for each pixel
        normal_alignment = np.sum(normals * camera_normal.reshape(1, 1, 3), axis=2)
        
        # Find regions well-aligned with camera (horizontal surfaces)
        horizontal_surfaces = (normal_alignment > normal_threshold) & mask
        
        if self.debug:
            print(f"DEBUG: Found {np.sum(horizontal_surfaces)} horizontal surface pixels (threshold: {normal_threshold})")
            if np.any(mask):
                valid_alignments = normal_alignment[mask]
                print(f"DEBUG: Normal alignment range: {np.min(valid_alignments):.3f} to {np.max(valid_alignments):.3f}")
                print(f"DEBUG: Mean alignment: {np.mean(valid_alignments):.3f}, Std: {np.std(valid_alignments):.3f}")
                print(f"DEBUG: Above threshold count: {np.sum(valid_alignments > normal_threshold)}")
        
        if not np.any(horizontal_surfaces):
            if self.debug:
                print("DEBUG: No horizontal surfaces found")
            return []
            
        # Find connected components of horizontal surfaces
        num_labels, labels = cv2.connectedComponents(horizontal_surfaces.astype(np.uint8))
        
        if self.debug:
            print(f"DEBUG: Found {num_labels-1} connected components of horizontal surfaces")
        
        for label in range(1, num_labels):
            component = (labels == label)
            area = np.sum(component)
            
            if self.debug:
                print(f"DEBUG: Component {label} has area {area} pixels")
            
            # Only consider reasonably sized flat surfaces
            if min_patch_area < area < 2000:  # Cap-sized surfaces
                # Find center of the surface patch
                moments = cv2.moments(component.astype(np.uint8))
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    # Verify this is a stable flat surface
                    if self.debug:
                        print(f"DEBUG: Verifying surface at ({cx}, {cy})...")
                    
                    surface_is_valid = self._verify_flat_surface(depth, cx, cy, mask, patch_size)
                    if self.debug:
                        print(f"DEBUG: Surface verification result: {surface_is_valid}")
                    
                    if surface_is_valid:
                        # Calculate confidence based on normal consistency and area
                        patch_normals = normals[component]  # This gives us (N, 3) array
                        if len(patch_normals) > 0:
                            normal_consistency = np.mean(np.sum(patch_normals * camera_normal, axis=1))
                        else:
                            normal_consistency = 0.0
                        area_score = min(1.0, area / 500.0)  # Normalize by typical cap area
                        confidence = 0.7 * normal_consistency + 0.3 * area_score
                        
                        points.append(InteractionPoint(
                            x=cx, y=cy,
                            score=min(1.0, confidence * 1.2),
                            interaction_type=InteractionType.GRASP_SURFACE,
                            confidence=confidence,
                            grasp_width=np.sqrt(area)  # Estimate based on surface area
                        ))
        
        # Also detect side surfaces (vertical walls) for grasping
        side_normal = np.array([1, 0, 0])  # Side-facing normal
        side_alignment = np.abs(np.sum(normals * side_normal.reshape(1, 1, 3), axis=2))
        vertical_surfaces = (side_alignment > normal_threshold) & mask
        
        if np.any(vertical_surfaces):
            num_labels, labels = cv2.connectedComponents(vertical_surfaces.astype(np.uint8))
            
            for label in range(1, num_labels):
                component = (labels == label)
                area = np.sum(component)
                
                if 30 < area < 800:  # Smaller patches for side grasping
                    moments = cv2.moments(component.astype(np.uint8))
                    if moments["m00"] > 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        
                        # Verify this is a graspable side surface
                        if self._verify_side_surface(depth, cx, cy, mask, patch_size):
                            points.append(InteractionPoint(
                                x=cx, y=cy,
                                score=0.8,
                                interaction_type=InteractionType.GRASP_EDGE,
                                confidence=0.75,
                                approach_angle=90.0  # Side approach
                            ))
        
        return points[:self.cascade_config["max_points"]]
    
    def _compute_surface_normals(self, depth: np.ndarray, mask: np.ndarray, patch_size: int) -> np.ndarray:
        """Compute surface normals using depth gradients - more robust than 3D fitting."""
        h, w = depth.shape
        normals = np.zeros((h, w, 3), dtype=np.float32)
        
        # Convert depth to float for gradient computation
        depth_float = depth.astype(np.float32)
        
        # Compute depth gradients using Sobel operator
        grad_x = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
        
        # Only compute normals for masked pixels
        for y in range(1, h-1):
            for x in range(1, w-1):
                if not mask[y, x] or depth[y, x] == 0:
                    continue
                    
                # Get local depth gradients
                dx = grad_x[y, x]
                dy = grad_y[y, x]
                
                # Compute normal using cross product method
                # Two tangent vectors: (1, 0, dx) and (0, 1, dy)
                # Normal = (1, 0, dx) × (0, 1, dy) = (-dx, -dy, 1)
                normal = np.array([-dx, -dy, 1.0])
                
                # Normalize the normal vector
                norm_length = np.linalg.norm(normal)
                if norm_length > 1e-6:  # Avoid division by zero
                    normal = normal / norm_length
                    
                    # Ensure normal points toward camera (negative z)
                    if normal[2] > 0:
                        normal = -normal
                    
                    normals[y, x] = normal
        
        return normals
    
    def _verify_flat_surface(self, depth: np.ndarray, x: int, y: int, mask: np.ndarray, window_size: int) -> bool:
        """Verify that a point represents a stable flat surface."""
        half_win = window_size // 2
        y_min = max(0, y - half_win)
        y_max = min(depth.shape[0], y + half_win + 1)
        x_min = max(0, x - half_win)
        x_max = min(depth.shape[1], x + half_win + 1)
        
        local_depth = depth[y_min:y_max, x_min:x_max]
        local_mask = mask[y_min:y_max, x_min:x_max]
        
        if np.sum(local_mask) < 0.5 * local_mask.size:
            return False
        
        # Convert depth from mm to meters for consistency
        valid_depths = local_depth[local_mask & (local_depth > 0)] / 1000.0
        if len(valid_depths) < 5:
            return False
        
        # Check depth consistency (flat surface should have low variation)
        depth_std = np.std(valid_depths)
        depth_range = np.ptp(valid_depths)
        
        # Relaxed thresholds for real-world depth data noise
        max_std = 0.05   # 5cm std (was 1cm) 
        max_range = 0.10  # 10cm range (was 2cm)
        
        if self.debug:
            print(f"DEBUG: Surface validation at ({x},{y}) - std: {depth_std:.3f} (< {max_std}), range: {depth_range:.3f} (< {max_range})")
            print(f"DEBUG: Valid depths count: {len(valid_depths)}, mask coverage: {np.sum(local_mask)}/{local_mask.size} = {np.sum(local_mask)/local_mask.size:.2f}")
            print(f"DEBUG: Min depth: {np.min(valid_depths):.3f}m, Max depth: {np.max(valid_depths):.3f}m")
            print(f"DEBUG: Window size: {window_size}, actual window: {x_max-x_min}x{y_max-y_min}")
        
        result = depth_std < max_std and depth_range < max_range
        if self.debug:
            print(f"DEBUG: Surface verification result: {result}")
        
        return result
    
    def _verify_side_surface(self, depth: np.ndarray, x: int, y: int, mask: np.ndarray, window_size: int) -> bool:
        """Verify that a point represents a graspable side surface."""
        half_win = window_size // 2
        y_min = max(0, y - half_win)
        y_max = min(depth.shape[0], y + half_win + 1)
        x_min = max(0, x - half_win)
        x_max = min(depth.shape[1], x + half_win + 1)
        
        local_depth = depth[y_min:y_max, x_min:x_max]
        local_mask = mask[y_min:y_max, x_min:x_max]
        
        if np.sum(local_mask) < 0.3 * local_mask.size:
            return False
        
        valid_depths = local_depth[local_mask & (local_depth > 0)]
        if len(valid_depths) < 5:
            return False
        
        # Side surfaces should have moderate depth variation
        depth_std = np.std(valid_depths)
        depth_range = np.ptp(valid_depths)
        
        # More variation than flat surfaces but still structured
        return 0.005 < depth_std < 0.05 and 0.01 < depth_range < 0.1
    
    def _detect_depth_edges(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect edge features in depth that indicate graspable elements."""
        points = []
        
        # Use Scharr operator for better edge detection
        grad_x = cv2.Scharr(depth, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(depth, cv2.CV_64F, 0, 1)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag[~mask] = 0
        
        # Find strong edges
        valid_gradients = grad_mag[mask & (grad_mag > 0)]
        if len(valid_gradients) > 0:
            edge_threshold = np.percentile(valid_gradients, 90)
            strong_edges = (grad_mag > edge_threshold) & mask
            
            # Apply non-maximum suppression to thin edges
            grad_dir = np.arctan2(grad_y, grad_x)
            thin_edges = self._non_max_suppression_edges(grad_mag, grad_dir, strong_edges)
            
            # Sample points along edges
            edge_coords = np.where(thin_edges)
            step = max(1, len(edge_coords[0]) // 20)
            
            for i in range(0, len(edge_coords[0]), step):
                y, x = edge_coords[0][i], edge_coords[1][i]
                
                # Check if this edge is part of a graspable feature
                if self._is_graspable_edge(depth, x, y, mask):
                    edge_strength = grad_mag[y, x]
                    
                    # Calculate approach angle from gradient direction
                    angle = np.rad2deg(grad_dir[y, x]) % 180
                    
                    points.append(InteractionPoint(
                        x=int(x), y=int(y),
                        score=min(1.0, edge_strength / np.max(grad_mag) * 1.2),
                        interaction_type=InteractionType.GRASP_EDGE,
                        confidence=0.8,
                        approach_angle=angle
                    ))
        
        return points[:self.cascade_config["max_points"]]
    
    def _detect_advanced_geometric_features(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Use PCL or Open3D for advanced geometric feature extraction."""
        points = []
        
        if HAS_PCL:
            # Convert depth to point cloud
            cloud = self._depth_to_pointcloud_pcl(depth, mask)
            if cloud is None or cloud.size == 0:
                return []
                
            # Compute normals
            normals = cloud.make_NormalEstimation()
            normals.set_KSearch(10)
            normals_cloud = normals.compute()
            
            # Convert back to numpy for processing
            normal_array = np.asarray(normals_cloud)
            
            # Find points with distinctive normal patterns
            if normal_array.size > 0:
                # Find points with normals perpendicular to viewing direction
                # These are often good for grasping
                z_normals = normal_array[:, 2]  # Z component of normal
                grasp_candidates = np.where(np.abs(z_normals) < 0.3)[0]  # Nearly perpendicular to view
                
                # Convert back to image coordinates
                for idx in grasp_candidates[:min(10, len(grasp_candidates))]:
                    pt = cloud[idx]
                    x, y, z = pt[0], pt[1], pt[2]
                    
                    # Skip if outside mask
                    img_y, img_x = int(y), int(x)
                    if 0 <= img_y < mask.shape[0] and 0 <= img_x < mask.shape[1] and mask[img_y, img_x]:
                        points.append(InteractionPoint(
                            x=img_x, y=img_y,
                            score=0.75,
                            interaction_type=InteractionType.GRASP_EDGE,
                            confidence=0.7,
                            approach_angle=np.rad2deg(np.arctan2(normal_array[idx, 1], normal_array[idx, 0]))
                        ))
                    
        elif HAS_OPEN3D:
            # Create point cloud using Open3D
            cloud = self._depth_to_pointcloud_open3d(depth, mask)
            if cloud is None or len(cloud.points) == 0:
                return []
                
            # Estimate normals
            cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Access point and normal data
            points_array = np.asarray(cloud.points)
            normals_array = np.asarray(cloud.normals)
            
            if len(points_array) > 0:
                # Find points with normals perpendicular to viewing direction
                z_normals = normals_array[:, 2]  # Z component of normal
                grasp_candidates = np.where(np.abs(z_normals) < 0.3)[0]
                
                # Convert to image coordinates and add to points
                for idx in grasp_candidates[:min(10, len(grasp_candidates))]:
                    pt = points_array[idx]
                    x, y, z = pt[0], pt[1], pt[2]
                    
                    # Skip if outside mask
                    img_y, img_x = int(y), int(x)
                    if 0 <= img_y < mask.shape[0] and 0 <= img_x < mask.shape[1] and mask[img_y, img_x]:
                        points.append(InteractionPoint(
                            x=img_x, y=img_y,
                            score=0.75,
                            interaction_type=InteractionType.GRASP_EDGE,
                            confidence=0.7,
                            approach_angle=np.rad2deg(np.arctan2(normals_array[idx, 1], normals_array[idx, 0]))
                        ))
        
        return points[:self.cascade_config["max_points"]]
    
    # ------------------- RGB-D Fusion Methods -------------------
    
    def _detect_rgb_guided_features(self, image: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Extract features using RGB-D fusion with the DFormer-Tiny model."""
        points = []
        
        # Skip if the model isn't initialized
        if self.model is None:
            return points
            
        # Preprocess inputs for the model
        rgb_tensor = self._preprocess_image(image)
        depth_tensor = self._preprocess_depth(depth)
        
        # Get cross-modal attention maps from model
        with torch.no_grad():
            outputs = self.model(rgb_tensor, depth_tensor)
            attention_maps = outputs["attention_maps"]
        
        # Convert attention maps to numpy
        attention_np = []
        for attn_map in attention_maps:
            if self.use_cuda:
                attn_map = attn_map.cpu()
            attention_np.append(attn_map.numpy())
        
        # Process attention maps to find interaction points
        for interaction_type_idx, attention_map in enumerate(attention_np):
            # Filter attention map by object mask
            filtered_map = attention_map * mask
            
            # Find local maxima in attention map
            local_maxima = self._find_local_maxima(filtered_map, window_size=7)
            
            # Convert maxima to interaction points
            for y, x in zip(*np.where(local_maxima)):
                score = filtered_map[y, x]
                if score > 0.3:  # Minimum confidence threshold
                    # Determine interaction type from index
                    interaction_type = list(InteractionType)[interaction_type_idx % len(InteractionType)]
                    
                    # Compute approach angle based on depth gradients
                    approach_angle = self._compute_approach_angle(x, y, depth)
                    
                    points.append(InteractionPoint(
                        x=int(x), y=int(y),
                        score=float(score),
                        interaction_type=interaction_type,
                        confidence=float(score),
                        approach_angle=approach_angle
                    ))
        
        # Additional RGB-specific detection for handles on textured backgrounds
        # This targets specifically dark handles on wood grain
        dark_handle_points = self._detect_dark_handles_on_texture(image, depth, mask)
        points.extend(dark_handle_points)
        
        return points[:self.cascade_config["max_points"]]
    
    def _detect_dark_handles_on_texture(self, image: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Special detection for dark handles on textured backgrounds like wood grain."""
        points = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 1. Calculate texture complexity
        texture_map = cv2.Laplacian(gray, cv2.CV_64F)
        texture_map = cv2.GaussianBlur(np.abs(texture_map), (5, 5), 0)
        texture_map = texture_map / np.max(texture_map) if np.max(texture_map) > 0 else texture_map
        
        # 2. Detect dark regions
        # Adaptive threshold to find darker regions
        dark_threshold = np.percentile(gray[mask], 30)  # Bottom 30% darkness
        dark_regions = (gray < dark_threshold) & mask
        
        # 3. Find connected dark regions
        num_labels, labels = cv2.connectedComponents(dark_regions.astype(np.uint8))
        
        for label in range(1, num_labels):
            component = (labels == label)
            area = np.sum(component)
            
            # Only consider reasonably sized regions
            if 100 < area < 5000:
                # Check texture inside vs outside the region
                inside_texture = np.mean(texture_map[component])
                
                # Dilated region for context
                kernel = np.ones((15, 15), np.uint8)
                dilated = cv2.dilate(component.astype(np.uint8), kernel) > 0
                context_region = dilated & ~component & mask
                
                if np.sum(context_region) > 0:
                    outside_texture = np.mean(texture_map[context_region])
                    
                    # Dark handle on textured background will have lower texture inside
                    # and higher texture outside (wood grain)
                    if inside_texture < outside_texture * 0.7:
                        # Find center of the region
                        moments = cv2.moments(component.astype(np.uint8))
                        if moments["m00"] > 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            # Check if there's a depth feature supporting this
                            if self._verify_depth_feature(depth, component):
                                points.append(InteractionPoint(
                                    x=cx, y=cy,
                                    score=0.85,  # High score since this is a specialized detector
                                    interaction_type=InteractionType.HANDLE,
                                    confidence=0.9,
                                    grasp_width=np.sqrt(area)  # Approximate width based on area
                                ))
        
        return points
    
    def _verify_depth_feature(self, depth: np.ndarray, region: np.ndarray) -> bool:
        """Verify if a region has distinctive depth characteristics."""
        if np.sum(region) == 0:
            return False
            
        # Check depth variation in the region
        region_depth = depth[region]
        if len(region_depth) == 0 or np.all(region_depth == 0):
            return False
            
        region_depth = region_depth[region_depth > 0]  # Remove zero values
        if len(region_depth) == 0:
            return False
            
        # Dilated region for context
        kernel = np.ones((11, 11), np.uint8)
        dilated = cv2.dilate(region.astype(np.uint8), kernel) > 0
        context_region = dilated & ~region & (depth > 0)
        
        if np.sum(context_region) == 0:
            return False
            
        context_depth = depth[context_region]
        
        # Calculate statistics
        region_mean = np.mean(region_depth)
        context_mean = np.mean(context_depth)
        depth_diff = abs(region_mean - context_mean)
        
        # Check if there's a significant depth difference
        # A handle might be slightly protruding or recessed
        return depth_diff > 0.005  # 5mm difference threshold
    
    # ------------------- Scoring and Ranking Methods -------------------
    
    def _score_and_rank_points(self, points: List[InteractionPoint], 
                              image: np.ndarray, depth: np.ndarray, 
                              mask: np.ndarray) -> List[InteractionPoint]:
        """Score points based on multiple criteria to handle challenging cases."""
        if not points:
            return []
            
        # Calculate texture complexity map from RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        texture_map = cv2.Laplacian(gray, cv2.CV_64F)
        texture_map = cv2.GaussianBlur(np.abs(texture_map), (5, 5), 0)
        max_texture = np.max(texture_map) if np.max(texture_map) > 0 else 1.0
        texture_map = texture_map / max_texture  # Normalize
        
        # Calculate depth reliability map
        depth_edges = cv2.Sobel(depth, cv2.CV_64F, 1, 1)
        max_edge = np.max(np.abs(depth_edges)) if np.max(np.abs(depth_edges)) > 0 else 1.0
        depth_reliability = 1.0 - np.clip(np.abs(depth_edges) / max_edge, 0, 1)
        
        # Calculate distance transform for boundary distance
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        for point in points:
            # Base score from detection method
            total_score = point.score * 0.4
            
            if 0 <= point.y < mask.shape[0] and 0 <= point.x < mask.shape[1]:
                # TEXTURE PENALTY: Reduce score in highly textured regions
                # This helps avoid false positives in wood grain areas
                texture_score = 1.0 - texture_map[point.y, point.x]
                total_score += texture_score * 0.25
                
                # DEPTH RELIABILITY: Prioritize reliable depth regions
                reliability = depth_reliability[point.y, point.x]
                total_score += reliability * 0.2
                
                # BOUNDARY DISTANCE: Optimal distance from object edge
                dist = dist_transform[point.y, point.x]
                optimal_dist = 15.0  # pixels
                boundary_score = np.exp(-abs(dist - optimal_dist) / optimal_dist)
                total_score += boundary_score * 0.15
            
            # INTERACTION TYPE BONUS: Certain types get priority
            type_bonus = {
                InteractionType.HANDLE: 0.4,
                InteractionType.PUSH_POINT: 0.3,
                InteractionType.GRASP_EDGE: 0.2,
                InteractionType.GRASP_SURFACE: 0.1,
                InteractionType.PIVOT: 0.1,
                InteractionType.CONTACT: 0.0
            }
            bonus = type_bonus.get(point.interaction_type, 0.0)
            total_score += bonus * 0.1
            
            point.score = min(1.0, total_score)
        
        # Sort by score (highest first)
        return sorted(points, key=lambda p: p.score, reverse=True)
    
    # ------------------- Utility Methods -------------------
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image for the model."""
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
            
        # Normalize to [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_chw = np.transpose(img_float, (2, 0, 1))
        
        # Add batch dimension
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        
        if self.use_cuda:
            img_tensor = img_tensor.cuda()
            
        return img_tensor
    
    def _preprocess_depth(self, depth: np.ndarray) -> torch.Tensor:
        """Preprocess depth image for the model."""
        # Normalize depth
        depth_min = np.min(depth[depth > 0]) if np.any(depth > 0) else 0
        depth_max = np.max(depth) if np.max(depth) > 0 else 1.0
        
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)
            
        # Replace zeros with mean to avoid artifacts
        if np.any(depth_norm > 0):
            mean_depth = np.mean(depth_norm[depth_norm > 0])
            depth_norm[depth_norm == 0] = mean_depth
            
        # Add channel dimension
        depth_c = depth_norm.reshape(1, depth_norm.shape[0], depth_norm.shape[1])
        
        # Add batch dimension
        depth_tensor = torch.from_numpy(depth_c.astype(np.float32)).unsqueeze(0)
        
        if self.use_cuda:
            depth_tensor = depth_tensor.cuda()
            
        return depth_tensor
    
    def _compute_approach_angle(self, x: int, y: int, depth: np.ndarray) -> Optional[float]:
        """Compute approach angle based on depth gradients."""
        if not (0 < x < depth.shape[1]-1 and 0 < y < depth.shape[0]-1):
            return None
            
        # Compute gradient
        grad_x = (depth[y, x+1] - depth[y, x-1]) / 2.0
        grad_y = (depth[y+1, x] - depth[y-1, x]) / 2.0
        
        # Calculate angle (in degrees)
        angle = np.rad2deg(np.arctan2(grad_y, grad_x)) % 180
        
        return float(angle)
    
    def _find_local_maxima(self, data: np.ndarray, mask: np.ndarray = None, window_size: int = 5) -> np.ndarray:
        """Find local maxima in the data."""
        # Dilate to find local maxima
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size*2+1, window_size*2+1))
        dilated = cv2.dilate(data, kernel)
        
        # Local maxima are points equal to dilated version
        local_maxima = (data == dilated) & (data > 0)
        
        # Apply mask if provided
        if mask is not None:
            local_maxima = local_maxima & mask
            
        return local_maxima
    
    def _is_stable_feature(self, depth: np.ndarray, x: int, y: int, 
                          mask: np.ndarray, window_size: int = 7) -> bool:
        """Check if a feature is stable and well-defined."""
        half_win = window_size // 2
        y_min = max(0, y - half_win)
        y_max = min(depth.shape[0], y + half_win + 1)
        x_min = max(0, x - half_win)
        x_max = min(depth.shape[1], x + half_win + 1)
        
        local_region = depth[y_min:y_max, x_min:x_max]
        local_mask = mask[y_min:y_max, x_min:x_max]
        
        if np.sum(local_mask) < 0.5 * local_mask.size:
            return False
        
        valid_depths = local_region[local_mask & (local_region > 0)]
        if len(valid_depths) < 5:
            return False
        
        # Check depth variation is reasonable
        depth_range = np.ptp(valid_depths)
        depth_std = np.std(valid_depths)
        
        return 0.003 < depth_range < 0.1 and depth_std > 0.001  # 3mm to 10cm range with some variation
    
    def _non_max_suppression_edges(self, grad_mag: np.ndarray, grad_dir: np.ndarray,
                                  strong_edges: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression to edge pixels."""
        h, w = grad_mag.shape
        suppressed = np.zeros_like(grad_mag, dtype=bool)
        
        # Quantize gradient directions to 8 directions
        angle = np.rad2deg(grad_dir) % 180
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if not strong_edges[y, x]:
                    continue
                
                # Get gradient direction
                a = angle[y, x]
                
                # Check neighbors perpendicular to gradient
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    # Horizontal edge
                    if (grad_mag[y, x] >= grad_mag[y, x-1] and 
                        grad_mag[y, x] >= grad_mag[y, x+1]):
                        suppressed[y, x] = True
                elif 22.5 <= a < 67.5:
                    # Diagonal /
                    if (grad_mag[y, x] >= grad_mag[y-1, x+1] and 
                        grad_mag[y, x] >= grad_mag[y+1, x-1]):
                        suppressed[y, x] = True
                elif 67.5 <= a < 112.5:
                    # Vertical edge
                    if (grad_mag[y, x] >= grad_mag[y-1, x] and 
                        grad_mag[y, x] >= grad_mag[y+1, x]):
                        suppressed[y, x] = True
                else:
                    # Diagonal \
                    if (grad_mag[y, x] >= grad_mag[y-1, x-1] and 
                        grad_mag[y, x] >= grad_mag[y+1, x+1]):
                        suppressed[y, x] = True
        
        return suppressed
    
    def _is_graspable_edge(self, depth: np.ndarray, x: int, y: int, 
                          mask: np.ndarray, window_size: int = 9) -> bool:
        """Check if an edge point represents a graspable feature."""
        half_win = window_size // 2
        y_min = max(0, y - half_win)
        y_max = min(depth.shape[0], y + half_win + 1)
        x_min = max(0, x - half_win)
        x_max = min(depth.shape[1], x + half_win + 1)
        
        local_depth = depth[y_min:y_max, x_min:x_max]
        local_mask = mask[y_min:y_max, x_min:x_max]
        
        if np.sum(local_mask) < 0.3 * local_mask.size:
            return False
        
        # Check for depth variation indicating 3D structure
        valid_depths = local_depth[local_mask & (local_depth > 0)]
        if len(valid_depths) < 5:
            return False
        
        depth_std = np.std(valid_depths)
        depth_range = np.ptp(valid_depths)
        
        # Good graspable edges have moderate depth variation
        return 0.002 < depth_std < 0.05 and depth_range > 0.005
    
    def _apply_nms(self, points: List[InteractionPoint], min_distance: int) -> List[InteractionPoint]:
        """Apply non-maximum suppression to remove nearby duplicate points."""
        if not points:
            if self.debug:
                print(f"[DEBUG V2] NMS: No points to filter")
            return []
        
        if self.debug:
            print(f"[DEBUG V2] NMS: Starting with {len(points)} points, min_distance={min_distance}")
        
        # Sort by score (highest first)
        sorted_points = sorted(points, key=lambda p: p.score, reverse=True)
        
        keep = []
        filtered_count = 0
        for point in sorted_points:
            # Check if too close to any kept point
            too_close = False
            for kept in keep:
                dist = np.sqrt((point.x - kept.x)**2 + (point.y - kept.y)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                keep.append(point)
            else:
                filtered_count += 1
        
        if self.debug:
            print(f"[DEBUG V2] NMS: Kept {len(keep)} points, filtered {filtered_count} points")
        
        return keep
    
    # Point cloud conversion methods
    
    def _depth_to_pointcloud_pcl(self, depth: np.ndarray, mask: np.ndarray):
        """Convert depth image to PCL point cloud."""
        if not HAS_PCL:
            return None
            
        # Create empty point cloud
        cloud = pcl.PointCloud()
        
        # Get valid depth points
        valid_mask = (depth > 0) & mask
        if not np.any(valid_mask):
            return cloud
            
        y_indices, x_indices = np.where(valid_mask)
        points = np.zeros((len(y_indices), 3), dtype=np.float32)
        
        # Simple conversion assuming normalized coordinates
        points[:, 0] = x_indices  # X
        points[:, 1] = y_indices  # Y
        points[:, 2] = depth[valid_mask]  # Z
        
        # Set points in cloud
        cloud.from_array(points.astype(np.float32))
        return cloud
    
    def _depth_to_pointcloud_open3d(self, depth: np.ndarray, mask: np.ndarray):
        """Convert depth image to Open3D point cloud."""
        if not HAS_OPEN3D:
            return None
            
        # Create empty point cloud
        cloud = o3d.geometry.PointCloud()
        
        # Get valid depth points
        valid_mask = (depth > 0) & mask
        if not np.any(valid_mask):
            return cloud
            
        y_indices, x_indices = np.where(valid_mask)
        points = np.zeros((len(y_indices), 3), dtype=np.float32)
        
        # Simple conversion assuming normalized coordinates
        points[:, 0] = x_indices  # X
        points[:, 1] = y_indices  # Y
        points[:, 2] = depth[valid_mask]  # Z
        
        # Set points in cloud
        cloud.points = o3d.utility.Vector3dVector(points)
        return cloud
    
    # ------------------- Visualization Methods -------------------
    
    def visualize_points(self, image: np.ndarray, points: List[InteractionPoint], 
                        show_scores: bool = True, show_types: bool = True) -> np.ndarray:
        """Visualize detected interaction points on the image."""
        # Create a copy for visualization
        vis_img = image.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
            
        # Color map for different interaction types
        color_map = {
            InteractionType.GRASP_EDGE: (0, 255, 0),      # Green
            InteractionType.GRASP_SURFACE: (255, 0, 0),   # Blue
            InteractionType.PUSH_POINT: (0, 0, 255),      # Red
            InteractionType.HANDLE: (255, 255, 0),        # Cyan
            InteractionType.PIVOT: (255, 0, 255),         # Magenta
            InteractionType.CONTACT: (0, 255, 255)        # Yellow
        }
        
        for i, point in enumerate(points):
            # Get color based on interaction type
            color = color_map.get(point.interaction_type, (255, 255, 255))
            
            # Scale point size by confidence
            radius = int(5 + point.confidence * 5)
            
            # Draw point
            cv2.circle(vis_img, (point.x, point.y), radius, color, -1)
            
            # Draw point index
            cv2.putText(vis_img, str(i+1), (point.x + radius, point.y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show score if requested
            if show_scores:
                score_text = f"{point.score:.2f}"
                cv2.putText(vis_img, score_text, (point.x, point.y - radius - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show type if requested
            if show_types:
                type_text = point.interaction_type.value
                cv2.putText(vis_img, type_text, (point.x, point.y + radius + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw approach angle if available
            if point.approach_angle is not None:
                angle_rad = np.deg2rad(point.approach_angle)
                end_x = int(point.x + np.cos(angle_rad) * radius * 2)
                end_y = int(point.y + np.sin(angle_rad) * radius * 2)
                cv2.line(vis_img, (point.x, point.y), (end_x, end_y), (0, 255, 255), 2)
        
        return vis_img