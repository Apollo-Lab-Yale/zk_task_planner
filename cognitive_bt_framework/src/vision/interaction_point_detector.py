import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

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
    approach_angle: Optional[float] = None  # Recommended approach direction
    grasp_width: Optional[float] = None     # Estimated required gripper width
    stability: float = 0.5                  # How stable this contact point is
    accessibility: float = 0.5              # How accessible for robot arm
    surface_area: Optional[float] = None    # Surface area for surface points
    surface_stability: Optional[float] = None  # Surface stability metric
    surface_planarity: Optional[float] = None  # How planar the surface is

class RobustInteractionDetector:
    def __init__(self, debug: bool = False, save_debug_images: bool = False, debug_output_dir: str = "debug_output"):
        self.debug = debug
        self.save_debug_images = save_debug_images
        self.debug_output_dir = debug_output_dir
        self.debug_data = {}  # Store intermediate results for visualization
        
        if self.save_debug_images:
            import os
            os.makedirs(self.debug_output_dir, exist_ok=True)
        
    def detect_interaction_points(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        obj_info: Any = None,
        max_points: int = 15,
        depth_data: Optional[np.ndarray] = None,
        apply_center_shift: bool = True,
        edge_threshold: float = 15.0,
        shift_factor: float = 0.4,
        min_distance: int = 20,  # Added parameter for minimum distance between points
        fast_mode: bool = True,   # Enable fast mode by default
        depth_plane_only: bool = True  # NEW: Use only depth plane detection method
    ) -> List[InteractionPoint]:
        """
        Detect robust interaction points with optional center-shift for edge points.
        
        Args:
            image: RGB image of the object
            mask: Binary mask of the object
            obj_info: Optional object metadata
            max_points: Maximum number of points to return
            depth_data: Optional depth data for 3D analysis
            apply_center_shift: Whether to shift edge points toward center
            edge_threshold: Distance from edge to consider "edge point" (pixels)
            shift_factor: How much to shift toward center (0.0-1.0)
            min_distance: Minimum distance between points in pixels (for non-maximum suppression)
            fast_mode: Whether to use fast mode (ignored when depth_plane_only=True)
            depth_plane_only: If True, use only depth plane detection method (requires depth_data)
            
        Returns:
            List of interaction points
        """
        if not np.any(mask):
            return []
        
        # Check if we should use only depth plane detection
        if depth_plane_only:
            if depth_data is None:
                if self.debug:
                    print("WARNING: depth_plane_only=True but no depth_data provided. Falling back to standard methods.")
                depth_plane_only = False
            else:
                if self.debug:
                    print("Using depth plane detection method exclusively")
                # Use only the new depth plane detection method
                candidates = self._detect_depth_planes_with_distribution(depth_data, mask)
                if self.debug:
                    print(f"[DEBUG] Initial candidates from depth plane detection: {len(candidates)}")
                
                # Always apply distance filtering for spatial distribution
                if self.debug:
                    print(f"[DEBUG] Applying NMS filtering with min_distance={min_distance} pixels")
                final_points = self._apply_nms(candidates, min_distance=min_distance, depth_data=depth_data)
                if self.debug:
                    print(f"[DEBUG] Points after NMS filtering: {len(final_points)}")
                
                # Limit to max_points after distance filtering if needed
                if len(final_points) > max_points:
                    if self.debug:
                        print(f"[DEBUG] Limiting from {len(final_points)} to {max_points} points (max_points constraint)")
                    final_points = final_points[:max_points]
                
                # Apply center shift if requested (though depth plane points are already well-positioned)
                if apply_center_shift:
                    if self.debug:
                        print(f"[DEBUG] Applying center-shift for {len(final_points)} depth plane points")
                    final_points = self.shift_edge_points_to_center(
                        final_points, mask, edge_threshold, shift_factor
                    )
                    if self.debug:
                        print(f"[DEBUG] Points after center-shift: {len(final_points)}")
                
                if self.debug:
                    print(f"[DEBUG] Final point count returned: {len(final_points[:max_points])}")
                return final_points[:max_points]
        
        # Original multi-method detection (when depth_plane_only=False)
        # Convert image to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        masked_gray = np.zeros_like(gray)
        masked_gray[mask] = gray[mask]
        
        # Collect candidate points from multiple methods
        candidates = []
        
        if fast_mode:
            # Fast mode: Use only the most effective methods
            # Method 1: Geometric feature detection (fastest and most reliable)
            candidates.extend(self._detect_geometric_features(masked_gray, mask))
            
            # Method 2: Contour-based analysis (fast and effective)
            candidates.extend(self._detect_contour_features(mask))
            
            # Method 3: Corner detection only (skip expensive edge detection)
            candidates.extend(self._detect_corners_edges(masked_gray, mask, corners_only=True))
        else:
            # Full mode: Use all detection methods
            # Method 1: Geometric feature detection
            candidates.extend(self._detect_geometric_features(masked_gray, mask))
            
            # Method 2: Contour-based analysis
            candidates.extend(self._detect_contour_features(mask))
            
            # Method 3: Corner and edge detection
            candidates.extend(self._detect_corners_edges(masked_gray, mask))
            
            # Method 4: Surface analysis
            candidates.extend(self._detect_surface_features(masked_gray, mask))
            
            # Method 5: Handle/affordance detection
            candidates.extend(self._detect_affordances(image, mask))
            
            # Method 6: 3D analysis if depth available
            if depth_data is not None:
                candidates.extend(self._detect_3d_features(depth_data, mask))
        
        # Score and rank all candidates
        scored_points = self._score_and_rank_points(candidates, mask, image.shape)
        
        # Apply non-maximum suppression with configurable min_distance
        if self.debug:
            print(f"Applying NMS with minimum distance of {min_distance} pixels")
        final_points = self._apply_nms(scored_points, min_distance=min_distance, depth_data=depth_data)
        
        # Apply center shift if requested
        if apply_center_shift:
            if self.debug:
                print("\n--- Applying Center-Shift for Edge Points ---")
            final_points = self.shift_edge_points_to_center(
                final_points, mask, edge_threshold, shift_factor
            )
        
        # Debug data storage disabled for performance
        # if self.save_debug_images:
        #     self._store_debug_data(image, mask, depth_data, candidates, scored_points, final_points)
        #     self._create_debug_visualizations()
        #     self._write_debug_explanation(final_points[:max_points])
        
        return final_points[:max_points]

    
    
    def shift_edge_points_to_center(
        self,
        points: List[InteractionPoint],
        mask: np.ndarray,
        edge_threshold: float = 15.0,
        shift_factor: float = 0.4,
        min_shift: int = 5,
        max_shift: int = 20
    ) -> List[InteractionPoint]:
        """
        Shift points near object edges toward the object center for better depth reliability.
        
        Args:
            points: List of detected interaction points
            mask: Object mask
            edge_threshold: Distance from edge to consider "edge point" (pixels)
            shift_factor: How much to shift toward center (0.0-1.0)
            min_shift: Minimum shift distance (pixels)
            max_shift: Maximum shift distance (pixels)
        
        Returns:
            List of points with edge points shifted toward center
        """
        if not points or not np.any(mask):
            return points
        
        # Calculate distance from edges for each pixel
        distance_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Find object center
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            return points  # Can't find center
        
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]
        
        shifted_points = []
        
        for i, point in enumerate(points):
            x, y = point.x, point.y
            
            # Check if point is in bounds
            if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
                shifted_points.append(point)
                continue
            
            # Get distance from edge
            edge_distance = distance_transform[y, x]
            
            # Check if this is an edge point
            if edge_distance >= edge_threshold:
                # Interior point - no shift needed
                shifted_points.append(point)
                continue
            
            # Calculate vector to center
            dx = center_x - x
            dy = center_y - y
            distance_to_center = np.sqrt(dx**2 + dy**2)
            
            if distance_to_center < 1:
                shifted_points.append(point)
                continue
            
            # Calculate shift amount based on how close to edge
            edge_ratio = (edge_threshold - edge_distance) / edge_threshold
            shift_distance = max(min_shift, min(max_shift, distance_to_center * shift_factor * edge_ratio))
            
            # Calculate new position
            shift_x = (dx / distance_to_center) * shift_distance
            shift_y = (dy / distance_to_center) * shift_distance
            
            new_x = int(round(x + shift_x))
            new_y = int(round(y + shift_y))
            
            # Ensure new position is on the mask
            if (0 <= new_y < mask.shape[0] and 0 <= new_x < mask.shape[1] and mask[new_y, new_x]):
                # Create shifted point
                shifted_point = InteractionPoint(
                    x=new_x,
                    y=new_y,
                    score=point.score,
                    interaction_type=point.interaction_type,
                    confidence=point.confidence,
                    approach_angle=point.approach_angle,
                    grasp_width=point.grasp_width,
                    stability=point.stability,
                    accessibility=point.accessibility
                )
                
                # Add shift metadata
                shifted_point.was_shifted = True
                shifted_point.original_position = (x, y)
                shifted_point.shift_distance = shift_distance
                
                # DEBUG: Show center-shift impact
                print(f"DEBUG Center-shift: Point moved from ({x}, {y}) to ({new_x}, {new_y}), distance: {shift_distance:.1f}px")
                
                shifted_points.append(shifted_point)
            else:
                # Couldn't find valid shift position, keep original
                shifted_points.append(point)
        
        return shifted_points
    
    def _detect_geometric_features(self, gray: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect points based on geometric properties like curvature."""
        points = []
        
        # Find contours for geometric analysis
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            if len(contour) < 5:  # Reduced from 10 to 5
                continue
                
            # Detect high curvature points
            curvature_points = self._calculate_curvature_points(contour)
            for (x, y), curvature in curvature_points:
                if curvature > 0.2:  # Reduced from 0.3 to 0.2 for more points
                    points.append(InteractionPoint(
                        x=int(x), y=int(y),
                        score=curvature,
                        interaction_type=InteractionType.GRASP_EDGE,
                        confidence=min(curvature, 0.9)
                    ))
        
        return points
    
    def _detect_contour_features(self, mask: np.ndarray) -> List[InteractionPoint]:
        """Analyze contour properties for interaction points."""
        points = []
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Reduced from 100 to 50
                continue
                
            # Polygon approximation for corner detection
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Add polygon corners as grasp points
            for point in approx:
                x, y = point[0]
                points.append(InteractionPoint(
                    x=int(x), y=int(y),
                    score=0.8,
                    interaction_type=InteractionType.GRASP_EDGE,
                    confidence=0.8
                ))
            
            # Detect convex hull defects (potential handles)
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 3000:  # Reduced from 5000 to 3000 for more handles
                            far = tuple(contour[f][0])
                            points.append(InteractionPoint(
                                x=far[0], y=far[1],
                                score=min(d/10000.0, 1.0),
                                interaction_type=InteractionType.HANDLE,
                                confidence=0.7
                            ))
        
        return points
    
    def _detect_corners_edges(self, gray: np.ndarray, mask: np.ndarray, corners_only: bool = False) -> List[InteractionPoint]:
        """Detect corners and edges using multiple methods."""
        points = []
        
        # Harris corner detection
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Threshold and extract corner points
        corner_threshold = 0.01 * corners.max() if corners.max() > 0 else 0
        corner_locations = np.where(corners > corner_threshold)
        
        for y, x in zip(corner_locations[0], corner_locations[1]):
            if mask[y, x]:  # Ensure point is within object
                points.append(InteractionPoint(
                    x=int(x), y=int(y),
                    score=float(corners[y, x] / corners.max()),
                    interaction_type=InteractionType.GRASP_EDGE,
                    confidence=0.8
                ))
        
        # FAST corner detection for additional candidates
        fast = cv2.FastFeatureDetector_create(threshold=10)
        keypoints = fast.detect(gray)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                points.append(InteractionPoint(
                    x=x, y=y,
                    score=kp.response,
                    interaction_type=InteractionType.CONTACT,
                    confidence=0.6
                ))
        
        return points
    
    def _detect_surface_features(self, gray: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect stable surface contact points."""
        points = []
        
        # Find regions with low gradient (flat surfaces)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smooth gradient to find stable regions
        gradient_smooth = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0)
        
        # Find local minima in gradient (flat areas)
        flat_regions = gradient_smooth < np.percentile(gradient_smooth[mask], 25)
        flat_regions = flat_regions & mask
        
        # Sample points from flat regions
        y_coords, x_coords = np.where(flat_regions)
        if len(y_coords) > 0:
            # Subsample to avoid too many points
            step = max(1, len(y_coords) // 15)  # Increased sampling from //10 to //15
            for i in range(0, len(y_coords), step):
                x, y = x_coords[i], y_coords[i]
                points.append(InteractionPoint(
                    x=int(x), y=int(y),
                    score=0.6,
                    interaction_type=InteractionType.GRASP_SURFACE,
                    confidence=0.7
                ))
        
        return points
    
    def _detect_affordances(self, image: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect semantic affordances like handles, knobs, etc."""
        points = []
        
        # Look for circular features (knobs, caps, etc.)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Hough circle detection
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    points.append(InteractionPoint(
                        x=int(x), y=int(y),
                        score=0.9,
                        interaction_type=InteractionType.HANDLE,
                        confidence=0.8,
                        grasp_width=float(r * 2)
                    ))
        
        return points
    
    def _detect_3d_features(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Use depth information to find 3D interaction features."""
        points = []
        
        if depth is None:
            return points
            
        # Preprocess depth data
        depth_masked = depth.copy()
        depth_masked[~mask] = 0
        
        # Smooth depth to reduce noise
        depth_smooth = cv2.GaussianBlur(depth_masked, (5, 5), 0)
        depth_smooth[~mask] = 0
        
        # 1. Detect ridges and valleys (high curvature features)
        ridge_valley_points = self._detect_ridges_valleys(depth_smooth, mask)
        points.extend(ridge_valley_points)
        
        # 2. Detect peaks and local maxima (protruding features)
        peak_points = self._detect_depth_peaks(depth_smooth, mask)
        points.extend(peak_points)
        
        # 3. Detect stable planar regions (good for surface grasps)
        planar_points = self._detect_stable_planes(depth_smooth, mask)
        points.extend(planar_points)
        
        # 4. Detect depth discontinuities (edges in 3D)
        edge_points = self._detect_depth_discontinuities(depth_smooth, mask)
        points.extend(edge_points)
        
        # 5. Detect depth planes with distributed points (NEW METHOD)
        plane_points = self._detect_depth_planes_with_distribution(depth_smooth, mask)
        points.extend(plane_points)
        
        # 6. Calculate surface normals and approach angles for all points
        points = self._add_surface_normals(points, depth_smooth, mask)
        
        return points
    
    def _calculate_curvature_points(self, contour: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
        """Calculate curvature at contour points."""
        points_with_curvature = []
        
        if len(contour) < 5:
            return points_with_curvature
            
        # Smooth the contour first
        epsilon = 0.005 * cv2.arcLength(contour, True)
        smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(smooth_contour) < 3:
            return points_with_curvature
            
        points = [point[0] for point in smooth_contour]
        
        for i in range(len(points)):
            if len(points) < 3:
                continue
                
            # Get neighboring points (circular)
            prev_idx = (i - 1) % len(points)
            next_idx = (i + 1) % len(points)
            
            prev_pt = np.array(points[prev_idx], dtype=float)
            curr_pt = np.array(points[i], dtype=float)
            next_pt = np.array(points[next_idx], dtype=float)
            
            # Calculate vectors
            v1 = prev_pt - curr_pt
            v2 = next_pt - curr_pt
            
            # Avoid division by zero
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                v1_normalized = v1 / norm1
                v2_normalized = v2 / norm2
                
                # Calculate angle between vectors
                cos_angle = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Convert to curvature measure (0 = straight, 1 = sharp corner)
                curvature = 1.0 - (angle / np.pi)
                
                points_with_curvature.append(((curr_pt[0], curr_pt[1]), curvature))
        
        return points_with_curvature
    
    def _score_and_rank_points(
        self, 
        candidates: List[InteractionPoint], 
        mask: np.ndarray, 
        image_shape: Tuple[int, ...]
    ) -> List[InteractionPoint]:
        """Score points based on multiple criteria and rank them."""
        
        for point in candidates:
            # Base score from detection method
            total_score = point.score
            
            # Factor 1: Distance from object boundary (prefer points not too close to edge)
            boundary_score = self._calculate_boundary_score(point, mask)
            
            # Factor 2: Local stability (consistent neighborhood)
            stability_score = self._calculate_stability_score(point, mask)
            
            # Factor 3: Accessibility (how reachable for robot)
            accessibility_score = self._calculate_accessibility_score(point, mask)
            
            # Factor 4: Interaction type bonus
            type_bonus = self._get_interaction_type_bonus(point.interaction_type)
            
            # Factor 5: 3D geometric properties (if available)
            geometry_score = self._calculate_3d_geometry_score(point)
            
            # Combine scores with weights
            final_score = (
                total_score * 0.25 +
                boundary_score * 0.15 +
                stability_score * 0.15 +
                accessibility_score * 0.15 +
                type_bonus * 0.1 +
                geometry_score * 0.2
            )
            
            point.score = final_score
            point.stability = stability_score
            point.accessibility = accessibility_score
        
        # Sort by score (highest first)
        return sorted(candidates, key=lambda p: p.score, reverse=True)
    
    def _calculate_boundary_score(self, point: InteractionPoint, mask: np.ndarray) -> float:
        """Calculate how good the point is relative to object boundaries."""
        # Use distance transform to find distance from boundary
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        if (0 <= point.y < mask.shape[0] and 0 <= point.x < mask.shape[1]):
            distance_from_edge = dist_transform[point.y, point.x]
            # Optimal distance is neither too close to edge nor too far inside
            optimal_distance = 10.0
            score = 1.0 - abs(distance_from_edge - optimal_distance) / optimal_distance
            return max(0.0, min(1.0, score))
        return 0.0
    
    def _calculate_stability_score(self, point: InteractionPoint, mask: np.ndarray) -> float:
        """Calculate local stability around the point."""
        # Check how much of the neighborhood is part of the object
        neighborhood_size = 15
        y, x = point.y, point.x
        
        y_min = max(0, y - neighborhood_size)
        y_max = min(mask.shape[0], y + neighborhood_size + 1)
        x_min = max(0, x - neighborhood_size)
        x_max = min(mask.shape[1], x + neighborhood_size + 1)
        
        neighborhood = mask[y_min:y_max, x_min:x_max]
        if neighborhood.size > 0:
            return float(np.sum(neighborhood)) / neighborhood.size
        return 0.0
    
    def _calculate_accessibility_score(self, point: InteractionPoint, mask: np.ndarray) -> float:
        """Calculate how accessible the point is for robot manipulation."""
        # Simple heuristic: points closer to the boundary are more accessible
        # In a real system, this would consider robot kinematics and approach angles
        
        # Distance from mask centroid (center points are less accessible)
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] > 0:
            centroid_x = M["m10"] / M["m00"]
            centroid_y = M["m01"] / M["m00"]
            
            distance_from_center = np.sqrt((point.x - centroid_x)**2 + (point.y - centroid_y)**2)
            max_distance = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2) / 2
            
            return min(1.0, distance_from_center / max_distance)
        
        return 0.5
    
    def _get_interaction_type_bonus(self, interaction_type: InteractionType) -> float:
        """Assign bonus scores based on interaction type usefulness."""
        type_scores = {
            InteractionType.GRASP_EDGE: 0.85,
            InteractionType.HANDLE: 0.95,
            InteractionType.GRASP_SURFACE: 0.95,  # Increased priority for surface centers
            InteractionType.CONTACT: 0.6,
            InteractionType.PUSH_POINT: 0.8,
            InteractionType.PIVOT: 0.75
        }
        return type_scores.get(interaction_type, 0.5)
    
    def _apply_nms(self, points: List[InteractionPoint], min_distance: int = 30, depth_data: Optional[np.ndarray] = None) -> List[InteractionPoint]:
        """
        Apply two-stage filtering: distance-based coverage + depth consistency prioritization.
        
        Stage 1: Ensure spatial coverage of all valid regions using distance filtering
        Stage 2: Within each spatial region, prioritize depth consistency over other metrics
        
        Args:
            points: List of candidate interaction points
            min_distance: Minimum allowed distance between points in pixels
            depth_data: Optional depth data for consistency checking
            
        Returns:
            Filtered list ensuring both coverage and depth consistency prioritization
        """
        if not points:
            if self.debug:
                print("[DEBUG] NMS: No input points to filter")
            return []
        
        if self.debug:
            print(f"[DEBUG] NMS: Starting with {len(points)} candidate points")
        
        # STAGE 1: SPATIAL COVERAGE - Group points into spatial regions
        spatial_regions = self._group_points_by_spatial_regions(points, min_distance)
        
        if self.debug:
            print(f"[DEBUG] NMS: Grouped into {len(spatial_regions)} spatial regions (min_distance={min_distance})")
            for i, region in enumerate(spatial_regions):
                print(f"[DEBUG] NMS:   Region {i}: {len(region)} points")
        
        # STAGE 2: DEPTH CONSISTENCY - Select best point from each region
        final_points = []
        points_filtered_in_regions = 0
        for i, region_points in enumerate(spatial_regions):
            points_filtered_in_regions += len(region_points) - 1  # All but the best point get filtered
            
            if depth_data is not None:
                # Within each region, prioritize depth consistency
                best_point = self._select_best_point_by_depth_consistency(region_points, depth_data)
            else:
                # Fallback to score-based selection
                best_point = max(region_points, key=lambda p: p.score)
            
            final_points.append(best_point)
        
        if self.debug:
            print(f"[DEBUG] NMS: Selected 1 best point from each region")
            print(f"[DEBUG] NMS: FILTERED OUT {points_filtered_in_regions} points due to proximity/redundancy")
            print(f"[DEBUG] NMS: Final result: {len(final_points)} points")
        
        return final_points
    
    def _group_points_by_spatial_regions(self, points: List[InteractionPoint], min_distance: int) -> List[List[InteractionPoint]]:
        """
        Group points into spatial regions based on minimum distance threshold.
        Ensures coverage of all distinct spatial areas.
        
        Args:
            points: List of interaction points
            min_distance: Minimum distance between regions
            
        Returns:
            List of point groups, each representing a spatial region
        """
        if not points:
            return []
        
        # Sort points by score (highest first) to ensure best points seed regions
        sorted_points = sorted(points, key=lambda p: p.score, reverse=True)
        
        regions = []
        assigned = [False] * len(sorted_points)
        
        for i, point in enumerate(sorted_points):
            if assigned[i]:
                continue
                
            # Start a new region with this point
            current_region = [point]
            assigned[i] = True
            
            # Find all points within min_distance of any point in this region
            for j, other_point in enumerate(sorted_points):
                if assigned[j]:
                    continue
                    
                # Check if this point is close to any point in the current region
                for region_point in current_region:
                    distance = np.sqrt(
                        (region_point.x - other_point.x)**2 + 
                        (region_point.y - other_point.y)**2
                    )
                    
                    if distance < min_distance:
                        current_region.append(other_point)
                        assigned[j] = True
                        break
            
            regions.append(current_region)
        
        return regions
    
    def _select_best_point_by_depth_consistency(self, region_points: List[InteractionPoint], depth_data: np.ndarray) -> InteractionPoint:
        """
        Select the best point from a spatial region prioritizing depth consistency.
        
        Args:
            region_points: Points within the same spatial region
            depth_data: Depth image for consistency analysis
            
        Returns:
            Best point based on depth consistency and score combination
        """
        if len(region_points) == 1:
            return region_points[0]
        
        # Calculate combined score: depth consistency (weighted higher) + original score
        best_point = None
        best_combined_score = -1
        
        for point in region_points:
            depth_consistency = self._calculate_depth_consistency(point, depth_data)
            
            # Weighted combination: 70% depth consistency, 30% original score
            combined_score = 0.7 * depth_consistency + 0.3 * (point.score / 100.0)  # Normalize score
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_point = point
        
        return best_point if best_point else region_points[0]
        
    def _calculate_depth_consistency(self, point: InteractionPoint, depth_data: np.ndarray, window_size: int = 15) -> float:
        """
        Calculate depth consistency around a point by measuring local depth variance.
        Lower variance indicates more consistent depth (better for grasping).
        
        Args:
            point: InteractionPoint to analyze
            depth_data: Depth image
            window_size: Size of analysis window around point
            
        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if depth_data is None:
            return 0.5  # Default consistency
            
        h, w = depth_data.shape
        half_window = window_size // 2
        
        # Define window bounds
        y_min = max(0, point.y - half_window)
        y_max = min(h, point.y + half_window)
        x_min = max(0, point.x - half_window)
        x_max = min(w, point.x + half_window)
        
        # Extract depth region
        depth_region = depth_data[y_min:y_max, x_min:x_max]
        
        # Filter valid depth values
        valid_mask = (depth_region > 0) & (depth_region < 10000)  # Valid depth range
        if not np.any(valid_mask):
            return 0.0  # No valid depth data
            
        valid_depths = depth_region[valid_mask]
        
        if len(valid_depths) < 5:  # Not enough points for reliable statistics
            return 0.0
            
        # Calculate depth consistency metrics
        depth_std = np.std(valid_depths)
        depth_mean = np.mean(valid_depths)
        
        # Normalize standard deviation by mean depth (relative consistency)
        if depth_mean > 0:
            relative_std = depth_std / depth_mean
            # Convert to consistency score (lower std = higher consistency)
            consistency = max(0.0, 1.0 - (relative_std * 10))  # Scale factor of 10
        else:
            consistency = 0.0
            
        return min(1.0, consistency)
    
    def _detect_ridges_valleys(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect ridges and valleys in depth data - high curvature features good for grasping."""
        points = []
        
        # Calculate second derivatives (Hessian matrix)
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # Second derivatives
        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate principal curvatures
        H = grad_xx + grad_yy  # Mean curvature approximation
        K = grad_xx * grad_yy - grad_xy**2  # Gaussian curvature approximation
        
        # Find ridge points (high mean curvature, positive Gaussian curvature)
        ridge_mask = (H > np.percentile(H[mask], 75)) & (K > 0) & mask  # Reduced from 85 to 75
        ridge_coords = np.where(ridge_mask)
        
        for i in range(0, len(ridge_coords[0]), max(1, len(ridge_coords[0]) // 12)):  # Increased sampling from //8 to //12
            y, x = ridge_coords[0][i], ridge_coords[1][i]
            curvature_strength = H[y, x]
            points.append(InteractionPoint(
                x=int(x), y=int(y),
                score=min(1.0, curvature_strength / H[mask].max()),
                interaction_type=InteractionType.GRASP_EDGE,
                confidence=0.85
            ))
        
        # Find valley points (high negative mean curvature)
        valley_mask = (H < np.percentile(H[mask], 25)) & (K > 0) & mask  # Increased from 15 to 25
        valley_coords = np.where(valley_mask)
        
        for i in range(0, len(valley_coords[0]), max(1, len(valley_coords[0]) // 12)):  # Increased sampling from //8 to //12
            y, x = valley_coords[0][i], valley_coords[1][i]
            curvature_strength = abs(H[y, x])
            points.append(InteractionPoint(
                x=int(x), y=int(y),
                score=min(1.0, curvature_strength / abs(H[mask].min())),
                interaction_type=InteractionType.HANDLE,
                confidence=0.8
            ))
        
        return points
    
    def _detect_depth_peaks(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect local maxima and peaks in depth - protruding features good for grasping."""
        points = []
        
        # Find local maxima using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        local_maxima = cv2.morphologyEx(depth, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold to find significant peaks
        peak_threshold = np.percentile(local_maxima[mask], 80)  # Reduced from 90 to 80
        peak_mask = (local_maxima > peak_threshold) & mask
        
        # Find connected components of peaks
        num_labels, labels = cv2.connectedComponents(peak_mask.astype(np.uint8))
        
        for label in range(1, num_labels):
            component_mask = (labels == label)
            if np.sum(component_mask) < 5:  # Reduced from 10 to 5
                continue
                
            # Find the highest point in this component
            component_depths = depth[component_mask]
            max_depth_idx = np.argmax(component_depths)
            coords = np.where(component_mask)
            y, x = coords[0][max_depth_idx], coords[1][max_depth_idx]
            
            # Calculate prominence (how much it stands out)
            prominence = local_maxima[y, x]
            
            points.append(InteractionPoint(
                x=int(x), y=int(y),
                score=min(1.0, prominence / local_maxima[mask].max()),
                interaction_type=InteractionType.GRASP_EDGE,
                confidence=0.9
            ))
        
        return points
    
    def _detect_stable_planes(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect individual surface planes and center points on them with high priority."""
        points = []
        
        # Step 1: Surface normal calculation for surface orientation
        surface_normals = self._calculate_surface_normals(depth, mask)
        
        # Step 2: Segment surfaces based on normal similarity and spatial connectivity
        surface_segments = self._segment_surfaces_by_normals(depth, mask, surface_normals)
        
        # Step 3: For each surface segment, find the best central point
        for segment_mask in surface_segments.values():
            if np.sum(segment_mask) < 50:  # Reduced from 100 to 50
                continue
                
            # Find the most central and stable point on this surface
            central_point = self._find_surface_center_point(depth, segment_mask, surface_normals)
            
            if central_point is not None:
                # Calculate surface quality metrics
                surface_area = np.sum(segment_mask)
                surface_stability = self._calculate_surface_stability(depth, segment_mask)
                surface_planarity = self._calculate_surface_planarity(depth, segment_mask)
                
                # Create high-priority surface point
                points.append(InteractionPoint(
                    x=central_point[0], y=central_point[1],
                    score=0.9 + surface_stability * 0.1,  # High base score
                    interaction_type=InteractionType.GRASP_SURFACE,
                    confidence=0.9,  # High confidence for surface centers
                    surface_area=surface_area,
                    surface_stability=surface_stability,
                    surface_planarity=surface_planarity
                ))
        
        return points
    
    def _detect_depth_discontinuities(self, depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Detect depth discontinuities - edges in 3D space."""
        points = []
        
        # Calculate depth gradients
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find significant discontinuities
        edge_threshold = np.percentile(gradient_magnitude[mask], 70)  # Reduced from 80 to 70
        edge_mask = (gradient_magnitude > edge_threshold) & mask
        
        # Apply non-maximum suppression to thin edges
        edges_thin = cv2.Canny(
            (gradient_magnitude * 255 / gradient_magnitude.max()).astype(np.uint8),
            threshold1=50, threshold2=150
        )
        edge_mask = edge_mask & (edges_thin > 0)
        
        # Sample points along edges
        edge_coords = np.where(edge_mask)
        step = max(1, len(edge_coords[0]) // 15)  # Increased sampling from //10 to //15
        
        for i in range(0, len(edge_coords[0]), step):
            y, x = edge_coords[0][i], edge_coords[1][i]
            edge_strength = gradient_magnitude[y, x]
            
            points.append(InteractionPoint(
                x=int(x), y=int(y),
                score=min(1.0, edge_strength / gradient_magnitude[mask].max()),
                interaction_type=InteractionType.GRASP_EDGE,
                confidence=0.8
            ))
        
        return points
    
    def _add_surface_normals(self, points: List[InteractionPoint], depth: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
        """Calculate surface normals and approach angles for interaction points."""
        if not points:
            return points
        
        # Calculate depth gradients for normal computation
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        for point in points:
            x, y = point.x, point.y
            
            # Check bounds
            if not (0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]):
                continue
                
            # Calculate surface normal (simplified - assumes depth is in camera frame)
            dx = grad_x[y, x]
            dy = grad_y[y, x]
            
            # Surface normal vector (pointing away from surface)
            normal = np.array([dx, dy, 1.0])
            normal = normal / np.linalg.norm(normal)
            
            # Calculate approach angle (angle from camera axis)
            camera_axis = np.array([0, 0, 1])
            cos_angle = np.dot(normal, camera_axis)
            approach_angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Store approach angle in degrees
            point.approach_angle = np.degrees(approach_angle)
            
            # Estimate grasp width based on local depth variation
            neighborhood_size = 10
            y_min = max(0, y - neighborhood_size)
            y_max = min(depth.shape[0], y + neighborhood_size)
            x_min = max(0, x - neighborhood_size)
            x_max = min(depth.shape[1], x + neighborhood_size)
            
            local_depth = depth[y_min:y_max, x_min:x_max]
            local_mask = mask[y_min:y_max, x_min:x_max]
            
            if np.any(local_mask):
                depth_range = np.ptp(local_depth[local_mask])
                # Estimate gripper width needed (in pixels, would need calibration for real units)
                point.grasp_width = float(depth_range * 100)  # Rough conversion
        
        return points
    
    def _calculate_3d_geometry_score(self, point: InteractionPoint) -> float:
        """Calculate score based on 3D geometric properties like approach angle and grasp width."""
        score = 0.5  # Default score if no 3D info available
        
        # Score based on approach angle (prefer angles that are feasible for robot)
        if point.approach_angle is not None:
            # Optimal approach angles are typically 30-60 degrees from surface normal
            optimal_min, optimal_max = 30.0, 60.0
            angle = point.approach_angle
            
            if optimal_min <= angle <= optimal_max:
                angle_score = 1.0
            elif angle < optimal_min:
                angle_score = max(0.0, angle / optimal_min)
            else:  # angle > optimal_max
                angle_score = max(0.0, 1.0 - (angle - optimal_max) / (90.0 - optimal_max))
            
            score += angle_score * 0.4
        
        # Score based on grasp width (prefer reasonable gripper widths)
        if point.grasp_width is not None:
            # Assume reasonable grasp widths are 10-100 pixels (would need calibration)
            optimal_min, optimal_max = 10.0, 100.0
            width = point.grasp_width
            
            if optimal_min <= width <= optimal_max:
                width_score = 1.0
            elif width < optimal_min:
                width_score = max(0.0, width / optimal_min)
            else:  # width > optimal_max
                width_score = max(0.0, 1.0 - (width - optimal_max) / (200.0 - optimal_max))
            
            score += width_score * 0.3
        
        # Bonus for surface-centered points with surface quality metrics
        if point.interaction_type == InteractionType.GRASP_SURFACE:
            surface_bonus = 0.0
            
            # Add bonus based on surface stability
            if hasattr(point, 'surface_stability') and point.surface_stability is not None:
                surface_bonus += point.surface_stability * 0.15
            
            # Add bonus based on surface planarity
            if hasattr(point, 'surface_planarity') and point.surface_planarity is not None:
                surface_bonus += point.surface_planarity * 0.15
            
            # Add bonus based on surface area (larger surfaces are more stable)
            if hasattr(point, 'surface_area') and point.surface_area is not None:
                # Normalize area (assume max useful area is ~1000 pixels)
                normalized_area = min(1.0, point.surface_area / 1000.0)
                surface_bonus += normalized_area * 0.1
            
            score += surface_bonus
        
        return min(1.0, score)
    
    def _calculate_surface_normals(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Calculate surface normals using robust multi-scale approach."""
        # Smooth depth slightly to reduce noise while preserving edges
        depth_smooth = cv2.bilateralFilter(depth.astype(np.float32), 5, 50, 50)
        depth_smooth[~mask] = 0
        
        # Calculate gradients at multiple scales for robustness
        grad_x_3 = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_3 = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=3)
        grad_x_5 = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=5)
        grad_y_5 = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=5)
        
        # Combine multi-scale gradients (weight smaller kernel more for fine details)
        grad_x = 0.7 * grad_x_3 + 0.3 * grad_x_5
        grad_y = 0.7 * grad_y_3 + 0.3 * grad_y_5
        
        # Initialize normal array
        normals = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
        
        # Calculate normals where mask is true using vectorized operations
        valid_mask = mask & (depth_smooth > 0)
        y_indices, x_indices = np.where(valid_mask)
        
        for i in range(len(y_indices)):
            y, x = y_indices[i], x_indices[i]
            # Surface normal vector (cross product method for better accuracy)
            normal = np.array([-grad_x[y, x], -grad_y[y, x], 1.0])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normals[y, x] = normal / norm
        
        return normals
    
    def _segment_surfaces_by_normals(self, depth: np.ndarray, mask: np.ndarray, normals: np.ndarray) -> dict:
        """Enhanced surface segmentation with curvature-based boundary detection."""
        # Step 1: Calculate normal similarity for adjacent pixels
        normal_similarity = self._calculate_normal_similarity(normals, mask)
        
        # Step 2: Add curvature-based surface boundary detection
        curvature_boundaries = self._detect_curvature_boundaries(depth, mask)
        
        # Step 3: Combine normal similarity with curvature boundaries
        similarity_threshold = 0.70  # Further reduced for better detection
        surface_mask = (normal_similarity > similarity_threshold) & mask & (~curvature_boundaries)
        
        # Step 4: Multi-scale planarity analysis
        planar_masks = self._multi_scale_planarity_analysis(depth, mask)
        
        # Step 5: Combine all criteria with adaptive weighting
        combined_mask = surface_mask & planar_masks
        
        # Step 6: Morphological operations to clean up and separate surfaces
        combined_mask = self._clean_and_separate_surfaces(combined_mask)
        
        # Step 7: Find connected components with size filtering
        segments = self._extract_surface_components(combined_mask)
        
        return segments
    
    def _calculate_normal_similarity(self, normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Calculate local normal similarity for surface segmentation."""
        h, w = mask.shape
        similarity = np.zeros((h, w), dtype=np.float32)
        
        # Check similarity with neighbors
        for y in range(1, h-1):
            for x in range(1, w-1):
                if not mask[y, x]:
                    continue
                    
                current_normal = normals[y, x]
                if np.linalg.norm(current_normal) == 0:
                    continue
                
                # Check 8-connected neighbors
                similarities = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                            neighbor_normal = normals[ny, nx]
                            if np.linalg.norm(neighbor_normal) > 0:
                                # Cosine similarity
                                sim = np.dot(current_normal, neighbor_normal)
                                similarities.append(max(0, sim))
                
                if similarities:
                    similarity[y, x] = np.mean(similarities)
        
        return similarity
    
    def _find_surface_center_point(self, depth: np.ndarray, segment_mask: np.ndarray, normals: np.ndarray) -> tuple:
        """Find the optimal center point using geometric moments and stability analysis."""
        # Method 1: Geometric center using moments
        geometric_center = self._calculate_geometric_center(segment_mask)
        
        # Method 2: Stability-based center (lowest variance point)
        stability_center = self._calculate_stability_center(depth, segment_mask)
        
        # Method 3: Distance-based center (maximal interior point)
        distance_center = self._calculate_distance_center(segment_mask)
        
        # Method 4: Depth-based center (consistent depth region)
        depth_center = self._calculate_depth_center(depth, segment_mask)
        
        # Combine methods using weighted voting
        candidates = [geometric_center, stability_center, distance_center, depth_center]
        weights = [0.3, 0.3, 0.2, 0.2]  # Prioritize geometric and stability centers
        
        # Filter out None candidates
        valid_candidates = [(c, w) for c, w in zip(candidates, weights) if c is not None]
        
        if not valid_candidates:
            return None
        
        # Find the consensus center point
        center = self._find_consensus_center(valid_candidates, segment_mask)
        
        return center
    
    def _calculate_geometric_center(self, segment_mask: np.ndarray) -> tuple:
        """Calculate the geometric center using image moments."""
        moments = cv2.moments(segment_mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None
        
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # Ensure the center is actually on the surface
        if segment_mask[cy, cx]:
            return (cx, cy)
        else:
            # Find nearest point on surface
            y_coords, x_coords = np.where(segment_mask)
            distances = (x_coords - cx)**2 + (y_coords - cy)**2
            nearest_idx = np.argmin(distances)
            return (int(x_coords[nearest_idx]), int(y_coords[nearest_idx]))
    
    def _calculate_stability_center(self, depth: np.ndarray, segment_mask: np.ndarray) -> tuple:
        """Find the most stable point (lowest local depth variance)."""
        # Calculate local depth variance
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        depth_mean = cv2.filter2D(depth, -1, kernel)
        depth_sq_mean = cv2.filter2D(depth**2, -1, kernel)
        depth_var = depth_sq_mean - depth_mean**2
        
        # Find minimum variance point within the segment
        segment_coords = np.where(segment_mask)
        if len(segment_coords[0]) == 0:
            return None
        
        min_var_idx = np.argmin(depth_var[segment_coords])
        best_y, best_x = segment_coords[0][min_var_idx], segment_coords[1][min_var_idx]
        
        return (int(best_x), int(best_y))
    
    def _calculate_distance_center(self, segment_mask: np.ndarray) -> tuple:
        """Find the point maximally distant from edges (most interior)."""
        dist_transform = cv2.distanceTransform(segment_mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Find the point with maximum distance from edge
        max_dist_idx = np.unravel_index(np.argmax(dist_transform * segment_mask), dist_transform.shape)
        
        if dist_transform[max_dist_idx] > 0:
            return (int(max_dist_idx[1]), int(max_dist_idx[0]))  # (x, y)
        
        return None
    
    def _calculate_depth_center(self, depth: np.ndarray, segment_mask: np.ndarray) -> tuple:
        """Find the center of the most consistent depth region."""
        # Get depth values on the surface
        surface_depths = depth[segment_mask]
        if len(surface_depths) == 0:
            return None
        
        # Find the modal depth (most common depth range)
        depth_median = np.median(surface_depths)
        depth_std = np.std(surface_depths)
        
        # Find points close to median depth
        depth_tolerance = min(depth_std, 2.0)  # Adaptive tolerance
        consistent_depth_mask = segment_mask & (np.abs(depth - depth_median) < depth_tolerance)
        
        if not np.any(consistent_depth_mask):
            return None
        
        # Calculate center of consistent depth region
        moments = cv2.moments(consistent_depth_mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None
        
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        return (cx, cy)
    
    def _find_consensus_center(self, candidates: list, segment_mask: np.ndarray) -> tuple:
        """Find consensus center point from multiple candidates."""
        if len(candidates) == 1:
            return candidates[0][0]
        
        # Calculate weighted average of candidate positions
        total_weight = sum(weight for _, weight in candidates)
        weighted_x = sum(pos[0] * weight for pos, weight in candidates) / total_weight
        weighted_y = sum(pos[1] * weight for pos, weight in candidates) / total_weight
        
        consensus_x, consensus_y = int(round(weighted_x)), int(round(weighted_y))
        
        # Ensure consensus point is on the surface
        if (0 <= consensus_y < segment_mask.shape[0] and 
            0 <= consensus_x < segment_mask.shape[1] and 
            segment_mask[consensus_y, consensus_x]):
            return (consensus_x, consensus_y)
        
        # If consensus point is not on surface, find nearest valid candidate
        min_distance = float('inf')
        best_candidate = candidates[0][0]
        
        for pos, _ in candidates:
            distance = (pos[0] - consensus_x)**2 + (pos[1] - consensus_y)**2
            if distance < min_distance:
                min_distance = distance
                best_candidate = pos
        
        return best_candidate
    
    def _store_debug_data(self, image, mask, depth_data, candidates, scored_points, final_points):
        """Store intermediate processing data for debug visualization."""
        self.debug_data = {
            'original_image': image.copy(),
            'mask': mask.copy(),
            'depth_data': depth_data.copy() if depth_data is not None else None,
            'all_candidates': candidates.copy(),
            'scored_points': scored_points.copy(),
            'final_points': final_points.copy()
        }
        
        # Store surface analysis data if depth is available
        if depth_data is not None:
            self._store_surface_analysis_data(depth_data, mask)
    
    def _store_surface_analysis_data(self, depth, mask):
        """Store surface analysis intermediate results for visualization."""
        try:
            # Calculate surface normals
            normals = self._calculate_surface_normals(depth, mask)
            self.debug_data['surface_normals'] = normals
            
            # Calculate curvature boundaries
            curvature_boundaries = self._detect_curvature_boundaries(depth, mask)
            self.debug_data['curvature_boundaries'] = curvature_boundaries
            
            # Calculate normal similarity
            normal_similarity = self._calculate_normal_similarity(normals, mask)
            self.debug_data['normal_similarity'] = normal_similarity
            
            # Multi-scale planarity analysis
            planar_masks = self._multi_scale_planarity_analysis(depth, mask)
            self.debug_data['planar_masks'] = planar_masks
            
            # Surface segments
            surface_segments = self._segment_surfaces_by_normals(depth, mask, normals)
            self.debug_data['surface_segments'] = surface_segments
            
        except Exception as e:
            print(f"Warning: Could not store surface analysis data: {e}")
    
    def _create_debug_visualizations(self):
        """Create comprehensive debug visualization images."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(20, 16))
            
            # Original image
            ax1 = plt.subplot(4, 5, 1)
            ax1.imshow(self.debug_data['original_image'])
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Mask
            ax2 = plt.subplot(4, 5, 2)
            ax2.imshow(self.debug_data['mask'], cmap='gray')
            ax2.set_title('Object Mask')
            ax2.axis('off')
            
            # Depth data
            if self.debug_data['depth_data'] is not None:
                ax3 = plt.subplot(4, 5, 3)
                depth_vis = self.debug_data['depth_data'].copy()
                depth_vis[~self.debug_data['mask']] = np.nan
                im3 = ax3.imshow(depth_vis, cmap='viridis')
                ax3.set_title('Depth Map')
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3, fraction=0.046)
                
                # Surface normals visualization
                if 'surface_normals' in self.debug_data:
                    ax4 = plt.subplot(4, 5, 4)
                    normals = self.debug_data['surface_normals']
                    # Visualize normal directions as RGB
                    normal_rgb = (normals + 1) / 2  # Normalize to [0,1]
                    normal_rgb[~self.debug_data['mask']] = 0
                    ax4.imshow(normal_rgb)
                    ax4.set_title('Surface Normals\n(RGB = XYZ)')
                    ax4.axis('off')
                
                # Curvature boundaries
                if 'curvature_boundaries' in self.debug_data:
                    ax5 = plt.subplot(4, 5, 5)
                    ax5.imshow(self.debug_data['original_image'])
                    ax5.imshow(self.debug_data['curvature_boundaries'], alpha=0.7, cmap='Reds')
                    ax5.set_title('Curvature Boundaries')
                    ax5.axis('off')
                
                # Normal similarity
                if 'normal_similarity' in self.debug_data:
                    ax6 = plt.subplot(4, 5, 6)
                    similarity = self.debug_data['normal_similarity']
                    similarity_vis = similarity.copy()
                    similarity_vis[~self.debug_data['mask']] = np.nan
                    im6 = ax6.imshow(similarity_vis, cmap='hot')
                    ax6.set_title('Normal Similarity')
                    ax6.axis('off')
                    plt.colorbar(im6, ax=ax6, fraction=0.046)
                
                # Planar regions
                if 'planar_masks' in self.debug_data:
                    ax7 = plt.subplot(4, 5, 7)
                    ax7.imshow(self.debug_data['original_image'])
                    ax7.imshow(self.debug_data['planar_masks'], alpha=0.7, cmap='Blues')
                    ax7.set_title('Planar Regions')
                    ax7.axis('off')
                
                # Surface segments
                if 'surface_segments' in self.debug_data:
                    ax8 = plt.subplot(4, 5, 8)
                    segments = self.debug_data['surface_segments']
                    segment_viz = np.zeros_like(self.debug_data['mask'], dtype=np.float32)
                    for i, (label, segment_mask) in enumerate(segments.items()):
                        segment_viz[segment_mask] = (i + 1) / len(segments)
                    ax8.imshow(segment_viz, cmap='tab10', vmin=0, vmax=1)
                    ax8.set_title(f'Surface Segments\n({len(segments)} found)')
                    ax8.axis('off')
            
            # All candidate points
            ax9 = plt.subplot(4, 5, 9)
            ax9.imshow(self.debug_data['original_image'])
            candidates = self.debug_data['all_candidates']
            for i, point in enumerate(candidates):
                color = self._get_type_color(point.interaction_type)
                ax9.scatter(point.x, point.y, c=color, s=20, alpha=0.6)
            ax9.set_title(f'All Candidates\n({len(candidates)} points)')
            ax9.axis('off')
            
            # Scored points
            ax10 = plt.subplot(4, 5, 10)
            ax10.imshow(self.debug_data['original_image'])
            scored = self.debug_data['scored_points']
            for i, point in enumerate(scored):
                color = self._get_type_color(point.interaction_type)
                size = point.score * 100  # Scale size by score
                ax10.scatter(point.x, point.y, c=color, s=size, alpha=0.7)
            ax10.set_title(f'Scored Points\n(Size = Score)')
            ax10.axis('off')
            
            # Final points with detailed info
            ax11 = plt.subplot(4, 5, 11)
            ax11.imshow(self.debug_data['original_image'])
            final = self.debug_data['final_points']
            for i, point in enumerate(final[:10]):  # Show top 10
                color = self._get_type_color(point.interaction_type)
                ax11.scatter(point.x, point.y, c=color, s=150, alpha=0.9, edgecolors='white', linewidth=2)
                ax11.text(point.x+5, point.y-10, f'{i+1}', fontsize=10, color='white', weight='bold')
            ax11.set_title(f'Final Points\n({len(final)} selected)')
            ax11.axis('off')
            
            # Score distribution
            ax12 = plt.subplot(4, 5, 12)
            if scored:
                scores = [p.score for p in scored]
                ax12.hist(scores, bins=20, alpha=0.7, color='blue')
                ax12.axvline(x=np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
                ax12.set_xlabel('Score')
                ax12.set_ylabel('Count')
                ax12.set_title('Score Distribution')
                ax12.legend()
                ax12.grid(True, alpha=0.3)
            
            # Interaction type distribution
            ax13 = plt.subplot(4, 5, 13)
            type_counts = {}
            for point in final:
                type_name = point.interaction_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            if type_counts:
                types = list(type_counts.keys())
                counts = list(type_counts.values())
                colors = [self._get_type_color_name(t) for t in types]
                ax13.bar(types, counts, color=colors, alpha=0.7)
                ax13.set_ylabel('Count')
                ax13.set_title('Point Types')
                ax13.tick_params(axis='x', rotation=45)
            
            # Surface quality analysis (if surface points exist)
            ax14 = plt.subplot(4, 5, 14)
            surface_points = [p for p in final if p.interaction_type == InteractionType.GRASP_SURFACE]
            if surface_points:
                stabilities = [p.surface_stability for p in surface_points if p.surface_stability is not None]
                planarities = [p.surface_planarity for p in surface_points if p.surface_planarity is not None]
                if stabilities and planarities:
                    ax14.scatter(stabilities, planarities, c='red', s=100, alpha=0.7)
                    ax14.set_xlabel('Surface Stability')
                    ax14.set_ylabel('Surface Planarity')
                    ax14.set_title('Surface Quality')
                    ax14.grid(True, alpha=0.3)
                else:
                    ax14.text(0.5, 0.5, 'No surface\nquality data', ha='center', va='center', transform=ax14.transAxes)
                    ax14.set_title('Surface Quality')
            else:
                ax14.text(0.5, 0.5, 'No surface\npoints found', ha='center', va='center', transform=ax14.transAxes)
                ax14.set_title('Surface Quality')
            
            # Detection pipeline summary
            ax15 = plt.subplot(4, 5, 15)
            ax15.axis('off')
            pipeline_text = f"""Detection Pipeline Summary:
            
Input:
• Image: {self.debug_data['original_image'].shape}
• Mask pixels: {np.sum(self.debug_data['mask'])}
• Depth data: {'Yes' if self.debug_data['depth_data'] is not None else 'No'}

Processing:
• Candidates: {len(candidates)}
• After scoring: {len(scored)}
• After NMS: {len(final)}

Surface Analysis:
• Segments: {len(self.debug_data.get('surface_segments', {}))}
• Surface points: {len(surface_points)}

Top Point:
• Position: ({final[0].x}, {final[0].y}) if final else 'None'
• Score: {final[0].score:.3f} if final else 'N/A'
• Type: {final[0].interaction_type.value} if final else 'N/A'
"""
            ax15.text(0.05, 0.95, pipeline_text, transform=ax15.transAxes, fontsize=10, 
                     verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save the comprehensive debug image
            debug_file = f"{self.debug_output_dir}/interaction_detection_debug_{timestamp}.png"
            plt.savefig(debug_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Debug visualization saved to: {debug_file}")
            
        except Exception as e:
            print(f"Warning: Could not create debug visualization: {e}")
    
    def _write_debug_explanation(self, final_points):
        """Write detailed text explanation of the detection process."""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            explanation_file = f"{self.debug_output_dir}/detection_explanation_{timestamp}.txt"
            
            with open(explanation_file, 'w') as f:
                f.write("INTERACTION POINT DETECTION EXPLANATION\n")
                f.write("="*50 + "\n\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                # Input data summary
                f.write("INPUT DATA:\n")
                f.write("-"*20 + "\n")
                f.write(f"Image dimensions: {self.debug_data['original_image'].shape}\n")
                f.write(f"Object mask pixels: {np.sum(self.debug_data['mask'])}\n")
                f.write(f"Depth data available: {'Yes' if self.debug_data['depth_data'] is not None else 'No'}\n\n")
                
                # Detection methods used
                f.write("DETECTION METHODS APPLIED:\n")
                f.write("-"*30 + "\n")
                f.write("1. Geometric feature detection (curvature analysis)\n")
                f.write("2. Contour-based analysis (corners, convexity defects)\n")
                f.write("3. Corner and edge detection (Harris, FAST)\n")
                f.write("4. Surface analysis (low-gradient regions)\n")
                f.write("5. Affordance detection (circular features)\n")
                if self.debug_data['depth_data'] is not None:
                    f.write("6. 3D feature analysis (ridges, valleys, peaks, surfaces)\n")
                f.write("\n")
                
                # Surface analysis details
                if self.debug_data['depth_data'] is not None:
                    f.write("3D SURFACE ANALYSIS:\n")
                    f.write("-"*25 + "\n")
                    if 'surface_segments' in self.debug_data:
                        segments = self.debug_data['surface_segments']
                        f.write(f"Surface segments detected: {len(segments)}\n")
                        for i, (label, segment) in enumerate(segments.items()):
                            f.write(f"  Segment {i+1}: {np.sum(segment)} pixels\n")
                    f.write("\n")
                
                # Candidate generation
                candidates = self.debug_data['all_candidates']
                f.write("CANDIDATE GENERATION:\n")
                f.write("-"*25 + "\n")
                f.write(f"Total candidates generated: {len(candidates)}\n")
                
                # Count by type
                type_counts = {}
                for point in candidates:
                    type_name = point.interaction_type.value
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1
                
                for type_name, count in sorted(type_counts.items()):
                    f.write(f"  {type_name}: {count}\n")
                f.write("\n")
                
                # Scoring and ranking
                scored = self.debug_data['scored_points']
                f.write("SCORING AND RANKING:\n")
                f.write("-"*25 + "\n")
                f.write("Scoring factors:\n")
                f.write("  • Base detection score (30%)\n")
                f.write("  • Boundary distance score (15%)\n")
                f.write("  • Local stability score (15%)\n")
                f.write("  • Accessibility score (15%)\n")
                f.write("  • Interaction type bonus (10%)\n")
                f.write("  • 3D geometry score (20%)\n\n")
                
                if scored:
                    scores = [p.score for p in scored]
                    f.write(f"Score statistics:\n")
                    f.write(f"  Mean: {np.mean(scores):.3f}\n")
                    f.write(f"  Std:  {np.std(scores):.3f}\n")
                    f.write(f"  Min:  {np.min(scores):.3f}\n")
                    f.write(f"  Max:  {np.max(scores):.3f}\n\n")
                
                # Final selection
                f.write("FINAL POINT SELECTION:\n")
                f.write("-"*25 + "\n")
                f.write(f"Points after NMS: {len(final_points)}\n")
                f.write("Non-maximum suppression removes points too close together\n\n")
                
                # Detailed analysis of top points
                f.write("TOP SELECTED POINTS:\n")
                f.write("-"*25 + "\n")
                for i, point in enumerate(final_points[:10]):
                    f.write(f"\nPoint {i+1}:\n")
                    f.write(f"  Position: ({point.x}, {point.y})\n")
                    f.write(f"  Score: {point.score:.3f}\n")
                    f.write(f"  Type: {point.interaction_type.value}\n")
                    f.write(f"  Confidence: {point.confidence:.3f}\n")
                    f.write(f"  Stability: {point.stability:.3f}\n")
                    f.write(f"  Accessibility: {point.accessibility:.3f}\n")
                    
                    if point.approach_angle is not None:
                        f.write(f"  Approach angle: {point.approach_angle:.1f}°\n")
                    if point.grasp_width is not None:
                        f.write(f"  Grasp width: {point.grasp_width:.1f}\n")
                    if point.surface_area is not None:
                        f.write(f"  Surface area: {point.surface_area:.1f}\n")
                    if point.surface_stability is not None:
                        f.write(f"  Surface stability: {point.surface_stability:.3f}\n")
                    if point.surface_planarity is not None:
                        f.write(f"  Surface planarity: {point.surface_planarity:.3f}\n")
                
                # Recommendations
                f.write(f"\n\nRECOMMENDATIONS:\n")
                f.write("-"*20 + "\n")
                
                surface_points = [p for p in final_points if p.interaction_type == InteractionType.GRASP_SURFACE]
                if surface_points:
                    f.write(f"✓ Found {len(surface_points)} surface points - good for stable grasping\n")
                    best_surface = max(surface_points, key=lambda p: p.score)
                    f.write(f"  Best surface point at ({best_surface.x}, {best_surface.y}) with score {best_surface.score:.3f}\n")
                else:
                    f.write("⚠ No surface points found - consider adjusting detection parameters\n")
                
                handle_points = [p for p in final_points if p.interaction_type == InteractionType.HANDLE]
                if handle_points:
                    f.write(f"✓ Found {len(handle_points)} handle points - good for manipulation\n")
                
                if final_points:
                    best_point = max(final_points, key=lambda p: p.score)
                    f.write(f"✓ Best overall point: ({best_point.x}, {best_point.y}) - {best_point.interaction_type.value}\n")
                else:
                    f.write("⚠ No interaction points detected - check input data and parameters\n")
            
            print(f"Detection explanation saved to: {explanation_file}")
            
        except Exception as e:
            print(f"Warning: Could not write debug explanation: {e}")
    
    def _get_type_color(self, interaction_type):
        """Get color for interaction point type."""
        color_map = {
            InteractionType.GRASP_SURFACE: 'red',
            InteractionType.GRASP_EDGE: 'blue', 
            InteractionType.HANDLE: 'green',
            InteractionType.CONTACT: 'orange',
            InteractionType.PUSH_POINT: 'purple',
            InteractionType.PIVOT: 'cyan'
        }
        return color_map.get(interaction_type, 'gray')
    
    def _get_type_color_name(self, type_name):
        """Get color name for interaction type string."""
        color_map = {
            'grasp_surface': 'red',
            'grasp_edge': 'blue',
            'handle': 'green', 
            'contact': 'orange',
            'push_point': 'purple',
            'pivot': 'cyan'
        }
        return color_map.get(type_name, 'gray')
    
    def _calculate_surface_stability(self, depth: np.ndarray, segment_mask: np.ndarray) -> float:
        """Calculate stability metric for a surface segment."""
        if not np.any(segment_mask):
            return 0.0
        
        # Calculate depth variance across the surface
        surface_depths = depth[segment_mask]
        depth_std = np.std(surface_depths)
        
        # Normalize stability (lower variance = higher stability)
        max_expected_std = 5.0  # Adjust based on depth units
        stability = max(0.0, 1.0 - (depth_std / max_expected_std))
        
        return stability
    
    def _calculate_surface_planarity(self, depth: np.ndarray, segment_mask: np.ndarray) -> float:
        """Calculate how planar a surface segment is."""
        if not np.any(segment_mask):
            return 0.0
        
        # Get surface coordinates and depths
        coords = np.where(segment_mask)
        if len(coords[0]) < 3:
            return 0.0
        
        # Fit a plane to the surface points
        points_3d = np.column_stack([coords[1], coords[0], depth[coords]])
        
        # Calculate plane fit quality using SVD
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        
        # SVD to find best-fit plane
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1, :]  # Last row is the normal to the best-fit plane
        
        # Calculate distances from points to plane
        distances = np.abs(np.dot(centered_points, normal))
        mean_distance = np.mean(distances)
        
        # Normalize planarity (lower mean distance = higher planarity)
        max_expected_distance = 2.0  # Adjust based on depth units
        planarity = max(0.0, 1.0 - (mean_distance / max_expected_distance))
        
        return planarity
    
    def _detect_curvature_boundaries(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detect surface boundaries using curvature analysis."""
        # Calculate mean and Gaussian curvature
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)
        
        # Mean curvature and Gaussian curvature
        H = grad_xx + grad_yy  # Mean curvature * 2
        K = grad_xx * grad_yy - grad_xy**2  # Gaussian curvature
        
        # Detect high curvature regions (surface boundaries)
        curvature_magnitude = np.sqrt(H**2 + K**2)
        
        # Adaptive threshold based on local statistics
        curvature_threshold = np.percentile(curvature_magnitude[mask], 85)
        boundary_mask = (curvature_magnitude > curvature_threshold) & mask
        
        # Clean up boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary_mask = cv2.morphologyEx(boundary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return boundary_mask.astype(bool)
    
    def _multi_scale_planarity_analysis(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Analyze planarity at multiple scales for robust surface detection."""
        planar_masks = []
        
        # Test different neighborhood sizes for planarity
        for kernel_size in [5, 9, 13]:
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            depth_mean = cv2.filter2D(depth, -1, kernel)
            depth_sq_mean = cv2.filter2D(depth**2, -1, kernel)
            depth_var = depth_sq_mean - depth_mean**2
            
            # Adaptive threshold based on scale
            var_threshold = np.percentile(depth_var[mask], 40 - (kernel_size - 5) * 2)
            planar_mask = (depth_var < var_threshold) & mask
            planar_masks.append(planar_mask)
        
        # Combine scales - require consistency across scales
        combined_planar = planar_masks[0] & planar_masks[1]  # At least 2 scales agree
        
        return combined_planar
    
    def _clean_and_separate_surfaces(self, surface_mask: np.ndarray) -> np.ndarray:
        """Clean surface mask and separate touching surfaces."""
        # Remove small holes
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(surface_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_fill)
        
        # Separate touching surfaces using watershed-like approach
        # Distance transform to find surface centers
        dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        
        # Find local maxima (surface centers)
        local_maxima = dist_transform > 0.7 * dist_transform.max()
        
        # Use erosion to separate surfaces that might be touching
        if np.sum(local_maxima) > 1:  # Multiple surfaces detected
            kernel_separate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.erode(cleaned, kernel_separate, iterations=1)
        
        return cleaned.astype(bool)
    
    def _detect_depth_planes_with_distribution(
        self, 
        depth: np.ndarray, 
        mask: np.ndarray,
        depth_shift_threshold: float = 0.015,  # 1.5cm depth shift threshold
        min_region_area: int = 80,             # Minimum pixels for a valid region
        points_per_region: int = 3,            # Points to distribute per region
        gaussian_blur_size: int = 3,           # Blur kernel size for noise reduction
        gradient_threshold: float = 0.01       # Threshold for detecting depth gradients/shifts
    ) -> List[InteractionPoint]:
        """
        Detect distinct depth regions based on clusters of depth shifts and create distributed interaction points.
        
        This method analyzes depth gradients and discontinuities to identify regions where significant
        depth changes occur, then clusters these regions and places interaction points strategically.
        
        Args:
            depth: Depth image (in meters)
            mask: Object mask
            depth_shift_threshold: Minimum depth shift to consider significant (meters)
            min_region_area: Minimum area in pixels for a valid depth region
            points_per_region: Number of points to distribute per depth region
            gaussian_blur_size: Size of Gaussian blur kernel for depth smoothing
            gradient_threshold: Threshold for depth gradient magnitude to detect shifts
            
        Returns:
            List of interaction points distributed across detected depth shift regions
        """
        if depth is None or not np.any(mask):
            return []
        
        if self.debug:
            print(f"\n--- Detecting Depth Shift Regions with Distribution ---")
        
        # Apply mask to depth and smooth to reduce noise
        depth_masked = depth.copy()
        depth_masked[~mask] = 0
        
        # Gaussian blur to reduce depth noise
        if gaussian_blur_size > 0:
            depth_smooth = cv2.GaussianBlur(depth_masked, (gaussian_blur_size, gaussian_blur_size), 0)
            depth_smooth[~mask] = 0
        else:
            depth_smooth = depth_masked
        
        # Get valid depth values for analysis
        valid_depths = depth_smooth[mask & (depth_smooth > 0)]
        if self.debug:
            total_mask_pixels = np.sum(mask)
            valid_depth_pixels = len(valid_depths)
            depth_min, depth_max = np.min(valid_depths) if len(valid_depths) > 0 else (0, 0), np.max(valid_depths) if len(valid_depths) > 0 else (0, 0)
            depth_range = depth_max - depth_min
            print(f"[DEBUG] Depth validation - Mask pixels: {total_mask_pixels}, Valid depth pixels: {valid_depth_pixels}")
            print(f"[DEBUG] Depth range: {depth_min:.3f}m to {depth_max:.3f}m (range: {depth_range:.3f}m)")
            print(f"[DEBUG] Depth shift threshold: {depth_shift_threshold:.3f}m, Gradient threshold: {gradient_threshold:.3f}")
            
            # Check if there are regions with significant depth differences
            if depth_range > depth_shift_threshold * 10:  # If range is much larger than threshold
                print(f"[DEBUG] Large depth range detected - handle/depression should be detectable!")
        
        if len(valid_depths) < min_region_area:
            if self.debug:
                print(f"[DEBUG] FILTERED OUT: Insufficient valid depth points ({len(valid_depths)} < {min_region_area} required)")
            return []
        
        # 1. Detect depth gradients and discontinuities
        depth_shift_regions = self._detect_depth_shift_regions(
            depth_smooth, mask, gradient_threshold, depth_shift_threshold
        )
        
        if self.debug:
            print(f"[DEBUG] Depth plane detection - Step 1: Detected {len(depth_shift_regions)} initial depth shift regions")
        
        # 2. Cluster nearby depth shift regions
        clustered_regions = self._cluster_depth_regions(
            depth_shift_regions, depth_smooth, mask, min_region_area
        )
        
        if self.debug:
            print(f"[DEBUG] Depth plane detection - Step 2: Clustered into {len(clustered_regions)} final depth regions")
            regions_filtered = len(depth_shift_regions) - len(clustered_regions)
            if regions_filtered > 0:
                print(f"[DEBUG] Depth plane detection - FILTERED OUT {regions_filtered} regions during clustering")
        
        # 3. Create distributed points for each clustered region
        # Calculate region sizes to determine how many points to distribute
        region_areas = [np.sum(region_mask) for region_mask in clustered_regions]
        region_areas = [area for area in region_areas if area >= min_region_area]
        
        if len(region_areas) == 0:
            if self.debug:
                total_regions = len(clustered_regions)
                print(f"[DEBUG] FILTERED OUT: No regions meet minimum area requirement (had {total_regions} regions, need area >= {min_region_area})")
                for i, region_mask in enumerate(clustered_regions):
                    area = np.sum(region_mask)
                    print(f"[DEBUG]   Region {i}: {area} pixels (too small)")
            return []
        
        # Calculate adaptive points per region based on relative size
        max_area = max(region_areas)
        min_points_per_region = 1  # Minimum points for any valid region
        max_points_per_region = 8  # Maximum points for largest regions
        
        region_points = []
        for i, region_mask in enumerate(clustered_regions):
            region_area = np.sum(region_mask)
            if region_area < min_region_area:
                if self.debug:
                    print(f"[DEBUG] FILTERED OUT: Region {i} too small ({region_area} < {min_region_area} pixels)")
                continue
            
            # Calculate adaptive number of points based on region size
            size_ratio = region_area / max_area
            adaptive_points = int(min_points_per_region + 
                                size_ratio * (max_points_per_region - min_points_per_region))
            adaptive_points = max(min_points_per_region, min(adaptive_points, max_points_per_region))
            
            # Calculate representative depth for this region
            region_depths = depth_smooth[region_mask & (depth_smooth > 0)]
            if len(region_depths) > 0:
                representative_depth = np.median(region_depths)
                
                if self.debug:
                    print(f"Region {i}: depth={representative_depth:.3f}m, area={region_area} pixels, points={adaptive_points}")
                
                # Create distributed points for this region with adaptive count
                region_interaction_points = self._create_distributed_points_on_region(
                    depth_smooth, region_mask, representative_depth, adaptive_points, region_id=i
                )
                if self.debug:
                    print(f"[DEBUG]   Generated {len(region_interaction_points)} points for region {i}")
                region_points.extend(region_interaction_points)
        
        # 4. Add border points along object segmentation boundary
        border_points = self._create_segmentation_border_points(depth_smooth, mask)
        region_points.extend(border_points)
        
        if self.debug:
            print(f"Added {len(border_points)} border points along object segmentation")
        
        # 5. If no significant regions found, fall back to basic depth analysis
        if len(region_points) == 0:
            if self.debug:
                print("[DEBUG] FALLBACK: No significant depth shift regions found, using fallback method")
            fallback_points = self._create_distributed_points_single_plane(depth_smooth, mask, points_per_region)
            # Still add border points for fallback
            fallback_border_points = self._create_segmentation_border_points(depth_smooth, mask)
            if self.debug:
                print(f"[DEBUG] Fallback generated {len(fallback_points)} plane points + {len(fallback_border_points)} border points")
            return fallback_points + fallback_border_points
        
        if self.debug:
            print(f"Generated {len(region_points)} total points ({len(region_points) - len(border_points)} region + {len(border_points)} border)")
        
        return region_points
    
    def diagnose_handle_detection(self, depth: np.ndarray, mask: np.ndarray, handle_center_approx: tuple = None):
        """
        Diagnostic function to help debug why handles/depressions aren't detected.
        Call this with debug=True to get detailed information.
        
        Args:
            depth: Depth image
            mask: Object mask  
            handle_center_approx: Optional (x, y) approximate center of handle for focused analysis
        """
        if not self.debug:
            print("[DIAGNOSTIC] Enable debug mode (self.debug = True) to see diagnostic output")
            return
            
        print("\n=== HANDLE DETECTION DIAGNOSTIC ===")
        
        # 1. Basic depth and mask info
        valid_depths = depth[mask & (depth > 0)]
        if len(valid_depths) == 0:
            print("[DIAGNOSTIC] CRITICAL: No valid depth data in mask!")
            return
            
        depth_min, depth_max = np.min(valid_depths), np.max(valid_depths)
        depth_range = depth_max - depth_min
        avg_depth = np.median(valid_depths)
        
        print(f"[DIAGNOSTIC] Mask coverage: {np.sum(mask)} pixels")
        print(f"[DIAGNOSTIC] Depth range: {depth_min:.3f}m to {depth_max:.3f}m (range: {depth_range:.3f}m)")
        print(f"[DIAGNOSTIC] Average depth: {avg_depth:.3f}m")
        
        # 2. Check handle area if coordinates provided
        if handle_center_approx:
            x, y = handle_center_approx
            if y < mask.shape[0] and x < mask.shape[1]:
                # Check small region around handle
                radius = 20
                y_min, y_max = max(0, y-radius), min(mask.shape[0], y+radius)
                x_min, x_max = max(0, x-radius), min(mask.shape[1], x+radius)
                
                handle_mask_included = mask[y_min:y_max, x_min:x_max]
                handle_depth_values = depth[y_min:y_max, x_min:x_max]
                
                print(f"[DIAGNOSTIC] Handle area ({x},{y} ±{radius}px):")
                print(f"[DIAGNOSTIC]   Mask coverage: {np.sum(handle_mask_included)}/{(y_max-y_min)*(x_max-x_min)} pixels")
                if np.any(handle_mask_included):
                    handle_depths = handle_depth_values[handle_mask_included & (handle_depth_values > 0)]
                    if len(handle_depths) > 0:
                        handle_depth_avg = np.mean(handle_depths)
                        print(f"[DIAGNOSTIC]   Handle depth: {handle_depth_avg:.3f}m vs surface avg {avg_depth:.3f}m")
                        print(f"[DIAGNOSTIC]   Depth difference: {abs(handle_depth_avg - avg_depth):.3f}m")
                else:
                    print(f"[DIAGNOSTIC]   WARNING: Handle area not covered by mask!")
        
        # 3. Check current thresholds
        gradient_threshold = 0.01
        depth_shift_threshold = 0.015
        min_region_area = 80
        
        depth_scale_factor = avg_depth / 1.0
        scaled_gradient_threshold = gradient_threshold * depth_scale_factor
        scaled_depth_shift_threshold = depth_shift_threshold * depth_scale_factor
        
        print(f"[DIAGNOSTIC] Current thresholds:")
        print(f"[DIAGNOSTIC]   Gradient threshold: {gradient_threshold:.4f} → {scaled_gradient_threshold:.4f} (scaled)")
        print(f"[DIAGNOSTIC]   Depth shift threshold: {depth_shift_threshold:.4f} → {scaled_depth_shift_threshold:.4f} (scaled)")
        print(f"[DIAGNOSTIC]   Min region area: {min_region_area} pixels")
        
        # 4. Recommendations
        print(f"[DIAGNOSTIC] Recommendations:")
        if depth_range > 0.1:  # 10cm range
            print(f"[DIAGNOSTIC]   ✓ Depth range ({depth_range:.3f}m) should be detectable")
        else:
            print(f"[DIAGNOSTIC]   ⚠ Small depth range ({depth_range:.3f}m) - may need lower thresholds")
            
        if avg_depth > 1.0:
            print(f"[DIAGNOSTIC]   ⚠ Far object - consider using fixed thresholds instead of scaled ones")
            print(f"[DIAGNOSTIC]   ⚠ Try: gradient_threshold=0.002, depth_shift_threshold=0.01")
            
        if handle_center_approx and not mask[handle_center_approx[1], handle_center_approx[0]]:
            print(f"[DIAGNOSTIC]   ⚠ Handle center not in mask - check segmentation!")
            
        print("=== END DIAGNOSTIC ===\n")
    
    def _create_distributed_points_single_plane(
        self, 
        depth: np.ndarray, 
        mask: np.ndarray, 
        num_points: int
    ) -> List[InteractionPoint]:
        """Create distributed points on a single depth plane."""
        if not np.any(mask):
            return []
        
        # Find the center point of the plane
        center_point = self._find_plane_center(depth, mask)
        if center_point is None:
            return []
        
        points = [center_point]
        
        # Add perimeter points if more points requested
        if num_points > 1:
            perimeter_points = self._find_plane_perimeter_points(depth, mask, num_points - 1)
            points.extend(perimeter_points)
        
        return points
    
    def _create_distributed_points_on_plane(
        self, 
        depth: np.ndarray, 
        plane_mask: np.ndarray, 
        plane_depth: float,
        num_points: int,
        plane_id: int = 0
    ) -> List[InteractionPoint]:
        """Create distributed points on a specific depth plane."""
        if not np.any(plane_mask):
            return []
        
        points = []
        
        # 1. Start with center point of the plane
        center_point = self._find_plane_center(depth, plane_mask, plane_depth, plane_id)
        if center_point is not None:
            points.append(center_point)
        
        # 2. Add perimeter/outline points if more points needed
        if num_points > 1:
            remaining_points = num_points - 1
            perimeter_points = self._find_plane_perimeter_points(
                depth, plane_mask, remaining_points, plane_depth, plane_id
            )
            points.extend(perimeter_points)
        
        return points
    
    def _find_plane_center(
        self, 
        depth: np.ndarray, 
        mask: np.ndarray, 
        target_depth: float = None,
        plane_id: int = 0
    ) -> InteractionPoint:
        """Find the center point of a depth plane."""
        if not np.any(mask):
            return None
        
        # Calculate the centroid of the masked region
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None
        
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        # Ensure the center point is within bounds and masked
        if (0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and 
            mask[center_y, center_x]):
            
            actual_depth = depth[center_y, center_x]
            confidence = 0.9  # High confidence for center points
            
            return InteractionPoint(
                x=center_x,
                y=center_y,
                score=confidence,
                interaction_type=InteractionType.CONTACT,
                confidence=confidence,
                stability=0.8,  # Centers are typically stable
                accessibility=0.7,  # Usually accessible
                surface_area=float(np.sum(mask)),
                surface_stability=0.8,
                surface_planarity=0.9  # Planes are planar by definition
            )
        
        return None
    
    def _find_plane_perimeter_points(
        self, 
        depth: np.ndarray, 
        mask: np.ndarray, 
        num_points: int,
        target_depth: float = None,
        plane_id: int = 0
    ) -> List[InteractionPoint]:
        """Find evenly distributed points along the perimeter/outline of a depth plane."""
        if not np.any(mask) or num_points <= 0:
            return []
        
        # Find contours of the plane mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Use the largest contour (main outline of the plane)
        main_contour = max(contours, key=cv2.contourArea)
        contour_length = cv2.arcLength(main_contour, True)
        
        if contour_length < 10:  # Too small contour
            return []
        
        # Calculate evenly spaced points along the contour
        points = []
        step_size = contour_length / num_points
        
        for i in range(num_points):
            # Calculate position along contour
            target_distance = i * step_size
            
            # Find the point at this distance along the contour
            current_distance = 0
            point_found = False
            
            for j in range(len(main_contour)):
                next_j = (j + 1) % len(main_contour)
                segment_start = main_contour[j][0]
                segment_end = main_contour[next_j][0]
                segment_length = np.linalg.norm(segment_end - segment_start)
                
                if current_distance + segment_length >= target_distance:
                    # Interpolate along this segment
                    ratio = (target_distance - current_distance) / segment_length if segment_length > 0 else 0
                    point_pos = segment_start + ratio * (segment_end - segment_start)
                    
                    x, y = int(point_pos[0]), int(point_pos[1])
                    
                    # Ensure point is within bounds
                    if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                        actual_depth = depth[y, x] if mask[y, x] else 0
                        
                        if actual_depth > 0:  # Valid depth
                            confidence = 0.7  # Lower confidence for perimeter points
                            
                            points.append(InteractionPoint(
                                x=x,
                                y=y,
                                score=confidence,
                                interaction_type=InteractionType.GRASP_EDGE,
                                confidence=confidence,
                                stability=0.6,  # Edge points may be less stable
                                accessibility=0.8,  # Edge points often more accessible
                                surface_area=float(np.sum(mask)),
                                surface_stability=0.6,
                                surface_planarity=0.8
                            ))
                    
                    point_found = True
                    break
                
                current_distance += segment_length
            
            if not point_found and len(main_contour) > 0:
                # Fallback: use contour point directly
                fallback_idx = min(i * len(main_contour) // num_points, len(main_contour) - 1)
                x, y = main_contour[fallback_idx][0]
                
                if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1] and mask[y, x]:
                    actual_depth = depth[y, x]
                    if actual_depth > 0:
                        points.append(InteractionPoint(
                            x=x,
                            y=y,
                            score=0.6,
                            interaction_type=InteractionType.GRASP_EDGE,
                            confidence=0.6,
                            stability=0.5,
                            accessibility=0.7,
                            surface_area=float(np.sum(mask)),
                            surface_stability=0.5,
                            surface_planarity=0.7
                        ))
        
        return points
    
    def _detect_depth_shift_regions(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        gradient_threshold: float = 0.01,
        depth_shift_threshold: float = 0.015
    ) -> List[np.ndarray]:
        """
        Detect regions where significant depth shifts occur, scaled by average region depth.
        
        Args:
            depth: Smoothed depth image
            mask: Object mask
            gradient_threshold: Threshold for depth gradient magnitude (will be scaled by depth)
            depth_shift_threshold: Minimum depth shift to consider significant (will be scaled by depth)
            
        Returns:
            List of binary masks for each detected depth shift region
        """
        if not np.any(mask):
            return []
        
        # Calculate average depth for scaling thresholds
        valid_depths = depth[mask & (depth > 0)]
        if len(valid_depths) == 0:
            return []
        
        avg_depth = np.median(valid_depths)  # Use median for robustness
        
        # Scale thresholds based on average depth (relative scaling)
        # Objects further away have proportionally larger depth variations
        depth_scale_factor = avg_depth / 1.0  # Normalize to 1 meter baseline
        scaled_gradient_threshold = gradient_threshold * depth_scale_factor
        scaled_depth_shift_threshold = depth_shift_threshold * depth_scale_factor
        
        if self.debug:
            print(f"[DEBUG] Depth shift detection - Average depth: {avg_depth:.3f}m, scale factor: {depth_scale_factor:.2f}")
            print(f"[DEBUG] Scaled gradient threshold: {scaled_gradient_threshold:.4f} (original: {gradient_threshold:.4f})")
            print(f"[DEBUG] Scaled depth shift threshold: {scaled_depth_shift_threshold:.4f} (original: {depth_shift_threshold:.4f})")
            if avg_depth > 1.0:
                print(f"[DEBUG] WARNING: Far object detected - scaled thresholds may be too high for handle detection!")
        
        # Calculate depth gradients
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply mask to gradients
        gradient_magnitude[~mask] = 0
        
        # Find areas with significant gradients using scaled threshold
        significant_gradients = (gradient_magnitude > scaled_gradient_threshold) & mask
        
        if self.debug:
            gradient_stats = gradient_magnitude[mask & (gradient_magnitude > 0)]
            if len(gradient_stats) > 0:
                grad_min, grad_max, grad_mean = np.min(gradient_stats), np.max(gradient_stats), np.mean(gradient_stats)
                significant_pixel_count = np.sum(significant_gradients)
                total_mask_pixels = np.sum(mask)
                print(f"[DEBUG] Gradient stats: min={grad_min:.4f}, max={grad_max:.4f}, mean={grad_mean:.4f}")
                print(f"[DEBUG] Significant gradient pixels: {significant_pixel_count}/{total_mask_pixels} ({100*significant_pixel_count/total_mask_pixels:.1f}%)")
                if grad_max < scaled_gradient_threshold:
                    print(f"[DEBUG] WARNING: Max gradient ({grad_max:.4f}) < threshold ({scaled_gradient_threshold:.4f}) - no regions will be detected!")
        
        # Use morphological operations to clean up gradient regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        significant_gradients = cv2.morphologyEx(
            significant_gradients.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        
        # Find connected components in gradient regions
        num_labels, labels = cv2.connectedComponents(significant_gradients.astype(np.uint8))
        
        if self.debug:
            print(f"[DEBUG] Found {num_labels-1} potential gradient regions")
        
        gradient_regions = []
        for label_id in range(1, num_labels):  # Skip background (label 0)
            region_mask = (labels == label_id) & mask
            region_area = np.sum(region_mask)
            
            if self.debug:
                print(f"[DEBUG]   Gradient region {label_id}: {region_area} pixels")
            
            # Check if this gradient region has sufficient area and depth variation
            if region_area > 10:  # Minimum area for gradient regions
                region_depths = depth[region_mask & (depth > 0)]
                if len(region_depths) > 5:
                    depth_variation = np.ptp(region_depths)  # Peak-to-peak depth range
                    # Use scaled threshold for depth variation check
                    if depth_variation > scaled_depth_shift_threshold:
                        gradient_regions.append(region_mask)
        
        # Also detect depth discontinuities using scaled threshold
        discontinuity_regions = self._detect_depth_discontinuities_regions(
            depth, mask, scaled_depth_shift_threshold
        )
        
        # Combine gradient-based and discontinuity-based regions
        all_regions = gradient_regions + discontinuity_regions
        
        if self.debug:
            print(f"Found {len(gradient_regions)} gradient regions and {len(discontinuity_regions)} discontinuity regions")
        
        return all_regions
    
    def _detect_depth_discontinuities_regions(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        depth_shift_threshold: float = 0.015
    ) -> List[np.ndarray]:
        """
        Detect regions with depth discontinuities using local depth statistics.
        
        Args:
            depth: Depth image
            mask: Object mask
            depth_shift_threshold: Minimum depth shift to consider significant
            
        Returns:
            List of binary masks for discontinuity regions
        """
        if not np.any(mask):
            return []
        
        # Calculate local depth statistics in neighborhoods
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        
        # Local mean and standard deviation
        local_mean = cv2.filter2D(depth, -1, kernel)
        local_sq_mean = cv2.filter2D(depth**2, -1, kernel)
        local_std = np.sqrt(np.maximum(0, local_sq_mean - local_mean**2))
        
        # Apply mask
        local_std[~mask] = 0
        
        # Find areas with high local depth variation
        discontinuity_threshold = depth_shift_threshold / 2  # More sensitive for local variations
        discontinuity_mask = (local_std > discontinuity_threshold) & mask
        
        # Clean up using morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        discontinuity_mask = cv2.morphologyEx(
            discontinuity_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
        )
        discontinuity_mask = cv2.morphologyEx(
            discontinuity_mask, cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(discontinuity_mask.astype(np.uint8))
        
        discontinuity_regions = []
        for label_id in range(1, num_labels):
            region_mask = (labels == label_id) & mask
            region_area = np.sum(region_mask)
            
            if region_area > 20:  # Minimum area for discontinuity regions
                discontinuity_regions.append(region_mask)
        
        return discontinuity_regions
    
    def _cluster_depth_regions(
        self,
        depth_regions: List[np.ndarray],
        depth: np.ndarray,
        mask: np.ndarray,
        min_region_area: int = 80
    ) -> List[np.ndarray]:
        """
        Cluster nearby depth regions focusing on depth discontinuities rather than gradual changes.
        
        This method analyzes depth continuity between regions and only separates regions
        where there are sudden depth jumps, not gradual transitions.
        
        Args:
            depth_regions: List of initial depth region masks
            depth: Depth image
            mask: Object mask
            min_region_area: Minimum area for final clustered regions
            
        Returns:
            List of clustered region masks representing distinct depth discontinuities
        """
        if not depth_regions:
            return []
        
        # Calculate region properties for clustering
        region_properties = []
        for i, region_mask in enumerate(depth_regions):
            if not np.any(region_mask):
                continue
                
            # Calculate centroid
            moments = cv2.moments(region_mask.astype(np.uint8))
            if moments["m00"] > 0:
                centroid_x = moments["m10"] / moments["m00"]
                centroid_y = moments["m01"] / moments["m00"]
                
                # Calculate representative depth and depth continuity metrics
                region_depths = depth[region_mask & (depth > 0)]
                if len(region_depths) > 0:
                    representative_depth = np.median(region_depths)
                    depth_std = np.std(region_depths)  # Internal depth variation
                    region_area = np.sum(region_mask)
                    
                    # Calculate depth gradient at region boundaries for continuity analysis
                    boundary_gradient = self._calculate_region_boundary_gradient(region_mask, depth)
                    
                    region_properties.append({
                        'index': i,
                        'centroid': (centroid_x, centroid_y),
                        'depth': representative_depth,
                        'depth_std': depth_std,
                        'boundary_gradient': boundary_gradient,
                        'area': region_area,
                        'mask': region_mask
                    })
        
        if not region_properties:
            return []
        
        # Enhanced clustering based on depth continuity analysis
        clustered_regions = []
        used_regions = set()
        
        for i, region in enumerate(region_properties):
            if i in used_regions:
                continue
                
            # Start a new cluster with this region
            cluster_mask = region['mask'].copy()
            cluster_regions = [i]
            used_regions.add(i)
            
            # Find nearby regions to merge based on depth continuity
            for j, other_region in enumerate(region_properties):
                if j in used_regions or j == i:
                    continue
                
                # Calculate spatial distance
                spatial_dist = np.sqrt(
                    (region['centroid'][0] - other_region['centroid'][0])**2 +
                    (region['centroid'][1] - other_region['centroid'][1])**2
                )
                
                # Enhanced depth continuity analysis
                should_merge = self._should_merge_regions_by_continuity(
                    region, other_region, depth, spatial_dist
                )
                
                if should_merge:
                    # Merge this region into the cluster
                    cluster_mask = cluster_mask | other_region['mask']
                    cluster_regions.append(j)
                    used_regions.add(j)
            
            # Add cluster if it meets minimum area requirement
            cluster_area = np.sum(cluster_mask)
            if cluster_area >= min_region_area:
                clustered_regions.append(cluster_mask)
        
        # If no regions meet the area requirement, create larger regions by relaxing constraints
        if not clustered_regions and region_properties:
            if self.debug:
                print("No clustered regions meet area requirement, creating relaxed clusters")
            
            # Create relaxed clusters by merging all nearby regions regardless of depth
            used_regions = set()
            for i, region in enumerate(region_properties):
                if i in used_regions:
                    continue
                
                cluster_mask = region['mask'].copy()
                used_regions.add(i)
                
                # Merge all nearby regions regardless of depth
                for j, other_region in enumerate(region_properties):
                    if j in used_regions or j == i:
                        continue
                    
                    spatial_dist = np.sqrt(
                        (region['centroid'][0] - other_region['centroid'][0])**2 +
                        (region['centroid'][1] - other_region['centroid'][1])**2
                    )
                    
                    if spatial_dist < 80:  # More relaxed spatial threshold
                        cluster_mask = cluster_mask | other_region['mask']
                        used_regions.add(j)
                
                cluster_area = np.sum(cluster_mask)
                if cluster_area >= min_region_area // 2:  # Relaxed area requirement
                    clustered_regions.append(cluster_mask)
        
        return clustered_regions
    
    def _calculate_region_boundary_gradient(
        self, 
        region_mask: np.ndarray, 
        depth: np.ndarray
    ) -> float:
        """
        Calculate the average depth gradient at the boundary of a region.
        
        This helps determine if the region represents a sudden depth change (high gradient)
        or a gradual transition (low gradient).
        """
        # Find region boundary using erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded_mask = cv2.erode(region_mask.astype(np.uint8), kernel, iterations=1)
        boundary_mask = region_mask.astype(np.uint8) - eroded_mask
        
        if not np.any(boundary_mask):
            return 0.0
        
        # Calculate depth gradients
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Get boundary gradients
        boundary_gradients = gradient_magnitude[boundary_mask.astype(bool)]
        
        if len(boundary_gradients) > 0:
            return np.mean(boundary_gradients)
        else:
            return 0.0
    
    def _should_merge_regions_by_continuity(
        self,
        region1: Dict,
        region2: Dict,
        depth: np.ndarray,
        spatial_dist: float
    ) -> bool:
        """
        Determine if two regions should be merged based on depth continuity analysis.
        
        Regions are merged if they represent continuous depth changes rather than 
        sudden discontinuities.
        """
        # Spatial proximity check
        max_spatial_distance = 60  # pixels
        if spatial_dist > max_spatial_distance:
            return False
        
        # Calculate depth difference
        depth_diff = abs(region1['depth'] - region2['depth'])
        
        # Scale depth threshold based on average depth with improved scaling
        avg_depth = (region1['depth'] + region2['depth']) / 2
        depth_scale_factor = max(avg_depth / 1.0, 0.5)  # Normalize to 1 meter, minimum 0.5x scaling
        # More permissive threshold for continuous regions: 4cm at 1m, scales with distance
        scaled_depth_threshold = 0.04 * depth_scale_factor
        
        # Check if depth difference is within continuous range
        depth_continuous = depth_diff < scaled_depth_threshold
        
        # Check boundary gradients - if both regions have low boundary gradients,
        # they likely represent gradual changes rather than sharp discontinuities
        gradient_threshold = 0.025 * depth_scale_factor  # Slightly more permissive gradient threshold
        
        region1_has_low_gradient = region1['boundary_gradient'] < gradient_threshold
        region2_has_low_gradient = region2['boundary_gradient'] < gradient_threshold
        
        # Also check if regions are spatially connected or very close
        # Dilate both regions slightly and check for overlap
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_region1 = cv2.dilate(region1['mask'].astype(np.uint8), kernel, iterations=1)
        dilated_region2 = cv2.dilate(region2['mask'].astype(np.uint8), kernel, iterations=1)
        
        regions_connected = np.any(dilated_region1 & dilated_region2)
        
        # Decision logic for merging:
        # Primary condition: regions on same depth plane with small depth difference
        same_depth_plane = depth_continuous and spatial_dist < 40  # Relaxed spatial distance for same plane
        
        # Secondary conditions for gradual transitions
        gradual_transition = (
            regions_connected and 
            depth_continuous and 
            (region1_has_low_gradient or region2_has_low_gradient)
        )
        
        # Alternative: merge if both regions have very similar internal depth variation
        # (indicates they're part of the same surface with similar characteristics)
        std_similarity = abs(region1['depth_std'] - region2['depth_std']) < 0.01 * depth_scale_factor
        similar_internal_variation = std_similarity and depth_continuous
        
        # Enhanced merging logic prioritizing depth plane continuity
        final_decision = (
            same_depth_plane or  # Primary: same depth plane (relaxed spatial requirement)
            gradual_transition or  # Secondary: connected gradual transitions
            (regions_connected and similar_internal_variation)  # Tertiary: similar surface characteristics
        )
        
        return final_decision
    
    def _create_distributed_points_on_region(
        self, 
        depth: np.ndarray, 
        region_mask: np.ndarray, 
        representative_depth: float,
        num_points: int,
        region_id: int = 0
    ) -> List[InteractionPoint]:
        """Create distributed points on a specific depth region (similar to plane method but for irregular regions)."""
        if not np.any(region_mask):
            if self.debug:
                print(f"[DEBUG] Region {region_id}: Empty region mask, no points created")
            return []
        
        points = []
        
        # 1. Start with center point of the region
        center_point = self._find_region_center(depth, region_mask, representative_depth, region_id)
        if center_point is not None:
            points.append(center_point)
            if self.debug:
                print(f"[DEBUG] Region {region_id}: Added center point at ({center_point.x}, {center_point.y})")
        else:
            if self.debug:
                print(f"[DEBUG] Region {region_id}: Failed to create center point")
        
        # 2. Add perimeter/boundary points if more points needed
        if num_points > 1:
            remaining_points = num_points - 1
            boundary_points = self._find_region_boundary_points(
                depth, region_mask, remaining_points, representative_depth, region_id
            )
            points.extend(boundary_points)
            if self.debug:
                print(f"[DEBUG] Region {region_id}: Added {len(boundary_points)} boundary points (requested {remaining_points})")
        
        if self.debug:
            print(f"[DEBUG] Region {region_id}: Created {len(points)} total points (requested {num_points})")
        
        return points
    
    def _create_segmentation_border_points(
        self, 
        depth: np.ndarray, 
        mask: np.ndarray,
        num_border_points: int = 6,
        border_erosion: int = 2
    ) -> List[InteractionPoint]:
        """
        Create interaction points distributed along the object segmentation border.
        
        Args:
            depth: Depth image
            mask: Object segmentation mask
            num_border_points: Number of points to distribute along border
            border_erosion: Pixels to erode from mask edge to avoid edge artifacts
            
        Returns:
            List of interaction points along the object border
        """
        if not np.any(mask):
            return []
        
        # Find the contour of the object mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return []
        
        # Use the largest contour (main object boundary)
        main_contour = max(contours, key=cv2.contourArea)
        
        if len(main_contour) < 3:
            return []
        
        # Smooth the contour to reduce noise
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        smooth_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Create an eroded mask to avoid placing points exactly on the edge
        if border_erosion > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_erosion*2+1, border_erosion*2+1))
            eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            eroded_mask = mask
        
        # Distribute points evenly along the contour
        border_points = []
        contour_length = len(smooth_contour)
        
        if contour_length > 0:
            step = max(1, contour_length // num_border_points)
            
            for i in range(0, contour_length, step):
                if len(border_points) >= num_border_points:
                    break
                
                # Get contour point
                contour_point = smooth_contour[i][0]
                x, y = int(contour_point[0]), int(contour_point[1])
                
                # Ensure point is within image bounds
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    # Move point slightly inward if using erosion
                    if border_erosion > 0 and eroded_mask[y, x]:
                        # Point is good as-is (inside eroded region)
                        pass
                    elif border_erosion > 0:
                        # Try to find a nearby point inside the eroded mask
                        found_valid = False
                        for offset in range(1, border_erosion + 1):
                            for dy in [-offset, 0, offset]:
                                for dx in [-offset, 0, offset]:
                                    ny, nx = y + dy, x + dx
                                    if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and 
                                        eroded_mask[ny, nx]):
                                        x, y = nx, ny
                                        found_valid = True
                                        break
                                if found_valid:
                                    break
                            if found_valid:
                                break
                        
                        if not found_valid:
                            continue  # Skip this point if we can't find a valid position
                    
                    # Create interaction point (border points are good for edge grasping)
                    border_points.append(InteractionPoint(
                        x=x, y=y,
                        score=0.7,  # Good score for border points
                        interaction_type=InteractionType.GRASP_EDGE,  # Border points are good for edge grasping
                        confidence=0.8,
                        grasp_width=20.0  # Default grasp width for border points
                    ))
        
        if self.debug and len(border_points) > 0:
            print(f"Created {len(border_points)} border points along object boundary")
        
        return border_points
    
    def _find_region_center(
        self, 
        depth: np.ndarray, 
        region_mask: np.ndarray, 
        representative_depth: float,
        region_id: int = 0
    ) -> InteractionPoint:
        """Find the center point of a depth region."""
        if not np.any(region_mask):
            return None
        
        # Calculate the centroid of the region
        moments = cv2.moments(region_mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None
        
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        # Ensure the center point is within bounds and in the region
        if (0 <= center_y < region_mask.shape[0] and 0 <= center_x < region_mask.shape[1] and 
            region_mask[center_y, center_x]):
            
            actual_depth = depth[center_y, center_x]
            confidence = 0.85  # High confidence for region centers
            
            return InteractionPoint(
                x=center_x,
                y=center_y,
                score=confidence,
                interaction_type=InteractionType.CONTACT,
                confidence=confidence,
                stability=0.75,  # Region centers are stable
                accessibility=0.7,  # Usually accessible
                surface_area=float(np.sum(region_mask)),
                surface_stability=0.75,
                surface_planarity=0.6  # Regions may be less planar than planes
            )
        
        return None
    
    def _find_region_boundary_points(
        self, 
        depth: np.ndarray, 
        region_mask: np.ndarray, 
        num_points: int,
        representative_depth: float,
        region_id: int = 0
    ) -> List[InteractionPoint]:
        """Find evenly distributed points along the boundary of a depth region."""
        if not np.any(region_mask) or num_points <= 0:
            return []
        
        # Find the boundary of the region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary = cv2.morphologyEx(region_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        
        # Find contours of the boundary
        contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Use the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        contour_length = cv2.arcLength(main_contour, True)
        
        if contour_length < 10:
            return []
        
        # Distribute points along the contour
        points = []
        step_size = contour_length / num_points
        
        for i in range(num_points):
            target_distance = i * step_size
            current_distance = 0
            point_found = False
            
            for j in range(len(main_contour)):
                next_j = (j + 1) % len(main_contour)
                segment_start = main_contour[j][0]
                segment_end = main_contour[next_j][0]
                segment_length = np.linalg.norm(segment_end - segment_start)
                
                if current_distance + segment_length >= target_distance:
                    ratio = (target_distance - current_distance) / segment_length if segment_length > 0 else 0
                    point_pos = segment_start + ratio * (segment_end - segment_start)
                    
                    x, y = int(point_pos[0]), int(point_pos[1])
                    
                    if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                        actual_depth = depth[y, x] if region_mask[y, x] else 0
                        
                        if actual_depth > 0:
                            confidence = 0.65  # Moderate confidence for boundary points
                            
                            points.append(InteractionPoint(
                                x=x,
                                y=y,
                                score=confidence,
                                interaction_type=InteractionType.GRASP_EDGE,
                                confidence=confidence,
                                stability=0.55,  # Boundary points may be less stable
                                accessibility=0.85,  # Boundary points often more accessible
                                surface_area=float(np.sum(region_mask)),
                                surface_stability=0.55,
                                surface_planarity=0.5
                            ))
                    
                    point_found = True
                    break
                
                current_distance += segment_length
            
            # Fallback if interpolation fails
            if not point_found and len(main_contour) > 0:
                fallback_idx = min(i * len(main_contour) // num_points, len(main_contour) - 1)
                x, y = main_contour[fallback_idx][0]
                
                if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1] and region_mask[y, x]:
                    actual_depth = depth[y, x]
                    if actual_depth > 0:
                        points.append(InteractionPoint(
                            x=x,
                            y=y,
                            score=0.6,
                            interaction_type=InteractionType.GRASP_EDGE,
                            confidence=0.6,
                            stability=0.5,
                            accessibility=0.8,
                            surface_area=float(np.sum(region_mask)),
                            surface_stability=0.5,
                            surface_planarity=0.5
                        ))
        
        return points
    
    def _extract_surface_components(self, surface_mask: np.ndarray) -> dict:
        """Extract connected components representing individual surfaces."""
        num_labels, labels = cv2.connectedComponents(surface_mask.astype(np.uint8))
        
        segments = {}
        for label in range(1, num_labels):
            segment_mask = (labels == label)
            segment_area = np.sum(segment_mask)
            
            # Size filtering with more permissive thresholds
            if segment_area >= 30:  # Further reduced from 50
                # Additional quality checks for surfaces
                if self._is_valid_surface_segment(segment_mask, segment_area):
                    segments[label] = segment_mask
        
        return segments
    
    def _is_valid_surface_segment(self, segment_mask: np.ndarray, area: int) -> bool:
        """Check if a surface segment meets quality criteria."""
        # Calculate aspect ratio and compactness
        moments = cv2.moments(segment_mask.astype(np.uint8))
        if moments["m00"] == 0:
            return False
        
        # Calculate bounding box
        y_coords, x_coords = np.where(segment_mask)
        if len(y_coords) == 0:
            return False
        
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        bbox_area = (max_y - min_y + 1) * (max_x - min_x + 1)
        if bbox_area == 0:
            return False
        
        # Calculate fill ratio (how much of bounding box is filled)
        fill_ratio = area / bbox_area
        
        # Accept surfaces with reasonable fill ratios (not too elongated)
        return fill_ratio > 0.3  # At least 30% of bounding box should be filled