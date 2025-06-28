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

class RobustInteractionDetector:
    def __init__(self, debug: bool = False):
        self.debug = debug
        
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
        min_distance: int = 30  # Added parameter for minimum distance between points
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
        
        return final_points[:max_points]

    
    
    def shift_edge_points_to_center(
        self,
        points: List[InteractionPoint],
        mask: np.ndarray,
        edge_threshold: float = 15.0,
        shift_factor: float = 0.4,
        min_shift: int = 5,
        max_shift: int = 25
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
            if len(contour) < 10:
                continue
                
            # Detect high curvature points
            curvature_points = self._calculate_curvature_points(contour)
            for (x, y), curvature in curvature_points:
                if curvature > 0.3:  # Significant curvature
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
            if cv2.contourArea(contour) < 100:
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
                        if d > 5000:  # Significant defect
                            far = tuple(contour[f][0])
                            points.append(InteractionPoint(
                                x=far[0], y=far[1],
                                score=min(d/10000.0, 1.0),
                                interaction_type=InteractionType.HANDLE,
                                confidence=0.7
                            ))
        
        return points
    
    def _detect_corners_edges(self, gray: np.ndarray, mask: np.ndarray) -> List[InteractionPoint]:
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
            step = max(1, len(y_coords) // 10)
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
            
        # Find depth edges (object boundaries in 3D)
        depth_masked = depth.copy()
        depth_masked[~mask] = 0
        
        # Calculate depth gradients
        grad_x = cv2.Sobel(depth_masked, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_masked, cv2.CV_64F, 0, 1, ksize=3)
        depth_edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find significant depth discontinuities
        edge_threshold = np.percentile(depth_edges[mask], 80)
        significant_edges = (depth_edges > edge_threshold) & mask
        
        y_coords, x_coords = np.where(significant_edges)
        for i in range(0, len(y_coords), max(1, len(y_coords) // 5)):
            x, y = x_coords[i], y_coords[i]
            points.append(InteractionPoint(
                x=int(x), y=int(y),
                score=0.7,
                interaction_type=InteractionType.GRASP_EDGE,
                confidence=0.8
            ))
        
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
            
            # Combine scores with weights
            final_score = (
                total_score * 0.3 +
                boundary_score * 0.2 +
                stability_score * 0.2 +
                accessibility_score * 0.2 +
                type_bonus * 0.1
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
            InteractionType.GRASP_EDGE: 0.9,
            InteractionType.HANDLE: 0.95,
            InteractionType.GRASP_SURFACE: 0.7,
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