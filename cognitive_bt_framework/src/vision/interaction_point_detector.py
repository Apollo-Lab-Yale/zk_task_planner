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
        shift_factor: float = 0.25,
        min_distance: int = 20,  # Added parameter for minimum distance between points
        fast_mode: bool = True   # Enable fast mode by default
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
            
        Returns:
            List of interaction points
        """
        if not np.any(mask):
            return []
            
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
        final_points = self._apply_nms(scored_points, min_distance=min_distance)
        
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
        max_shift: int = 10
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
        
        # 5. Calculate surface normals and approach angles for all points
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
    
    def _apply_nms(self, points: List[InteractionPoint], min_distance: int = 30) -> List[InteractionPoint]:
        """
        Apply non-maximum suppression to remove nearby duplicate points.
        
        Args:
            points: List of candidate interaction points
            min_distance: Minimum allowed distance between points in pixels
            
        Returns:
            Filtered list of points with no points closer than min_distance
        """
        if not points:
            return []
            
        # Keep track of which points to keep
        keep = [True] * len(points)
        
        for i in range(len(points)):
            if not keep[i]:
                continue
                
            for j in range(i + 1, len(points)):
                if not keep[j]:
                    continue
                    
                # Calculate distance between points
                distance = np.sqrt(
                    (points[i].x - points[j].x)**2 + 
                    (points[i].y - points[j].y)**2
                )
                
                if distance < min_distance:
                    # Keep the higher scored point
                    if points[i].score >= points[j].score:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        
        return [point for i, point in enumerate(points) if keep[i]]
    
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