# Perception Pipeline Documentation

Complete technical reference for the perception system in the cognitive BT framework, covering object detection, point of interest detection and filtering, and surface normal calculation.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Object Detection](#2-object-detection)
3. [Point of Interest Detection](#3-point-of-interest-detection)
4. [Filtering and Scoring](#4-filtering-and-scoring)
5. [Surface Normal Calculation](#5-surface-normal-calculation)
6. [Pose Estimation](#6-pose-estimation)
7. [Perception-to-Skill Bridge](#7-perception-to-skill-bridge)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. Pipeline Overview

```
RGB Image + Depth Image
         |
         v
  HybridYOLO Detection (YOLOWorld + FastSAM)
         |
         v
  ObjectInfo (bbox, mask, class, confidence)
         |
         +-------+----------+-----------------+
         |       |          |                 |
         v       v          v                 v
      ROI      Surface   Surface           6D Pose
    Detection  Segmentation  Normals      Estimation
         |       |          |                 |
         v       v          v                 v
  InteractionPoints  surface_masks  normal vectors  [x,y,z,r,p,y]
         |
         v
  Filtering: Score -> NMS -> Center-Shift
         |
         v
  PointOfInterest dict (normalized coords + metadata)
         |
         v
  SkillGenerator (LLM selects points, generates primitives)
```

### Key Files

| File | Role |
|------|------|
| `src/vision/perception_system.py` | Main orchestrator |
| `src/vision/interaction_point_detector_v2.py` | Primary interaction point detector (v2, geometric + ML) |
| `src/vision/interaction_point_detector.py` | Legacy/fallback detector (v1, depth-plane mode) |
| `src/vision/object_detection/yolo.py` | HybridYOLO: YOLOWorld + FastSAM |
| `src/vision/sam/fast_sam.py` | FastSAM wrapper for surface segmentation |
| `src/vision/contour_shape_detector.py` | Contour detection and shape classification |
| `src/skills/skill_handler.py` | Converts perception output to skill inputs |
| `src/skills/skill_generator.py` | LLM-based skill generation from points |

All paths below are relative to `cognitive_bt_framework/`.

---

## 2. Object Detection

### 2.1 HybridYOLO Architecture

**File:** `src/vision/object_detection/yolo.py`

The detection backend is a two-stage pipeline combining open-vocabulary detection with instance segmentation:

**Stage 1 — YOLOWorld (open-vocabulary detection):**
- Model: `yolov8x-worldv2.pt`
- `set_classes(classes)` dynamically configures target class names at runtime
- Produces bounding boxes with class labels and confidence scores

**Stage 2 — FastSAM (instance segmentation):**
- Model: `FastSAM-x.pt`
- Runs independently per detection bounding box on a cropped region
- Parameters: `imgsz=640, conf=0.4, iou=0.9`
- Selects the largest mask by area from FastSAM output
- Executed in parallel via `ThreadPoolExecutor(max_workers=4)`
- **Fallback:** If FastSAM returns no valid mask (area < 100px), a rectangular mask from the bbox is used

**`predict()`** (line 69): Calls `detect_objects()` internally, constructs ultralytics-compatible `Results` objects with fused `Boxes` and `Masks`.

### 2.2 PerceptionSystem Initialization

**File:** `src/vision/perception_system.py`, lines 51–130

Constructor parameters:
- `yolo_model_path: str = 'yoloe-11l-seg.pt'` — YOLO model (actual model loaded is `HybridYOLO`)
- `camera_matrix: Optional[np.ndarray]` — 3x3 intrinsic matrix
- `depth_scale: float = 0.001` — raw depth to meters conversion
- `default_conf: float = 0.5`
- `fast_sam_config: Optional[FastSAMConfig]`

Initializes:
1. `HybridYOLO` with `FastSAM-x.pt`, confidence threshold 0.1
2. `RobustInteractionDetector` (from `interaction_point_detector_v2.py`)
3. `FastSAMMaskGenerator` (for surface segmentation, optional)
4. Camera intrinsics: `fx, fy, cx, cy` and their inverses
5. `ContourShapeDetector` with `min_area=100, max_area=100000, min_circularity=0.5`

### 2.3 `detect_objects()` — Main Entry Point

**File:** `src/vision/perception_system.py`, lines 132–435

**Input:** RGB image, optional class list, confidence threshold, depth image

**Processing flow:**

1. **Class configuration** (line 169): Sets target classes on detector via `set_classes(classes)`
2. **YOLO prediction** (line 181): `results = self.detector.predict(temp_img)`
3. **Coordinate transformation** (lines 216–236): Computes `scale_x, scale_y` from YOLO input resolution to original image dimensions, transforms all bounding boxes
4. **Per-detection processing** (lines 255–423):
   - Extracts `xyxy` bbox, transforms to camera coords, converts to `[x, y, w, h]`
   - Extracts class name and confidence
   - **Mask processing** (lines 307–337): Gets polygon coordinates from `detection.masks.xy`, transforms each vertex by `scale_x/scale_y`, fills with `cv2.fillPoly`
   - Constructs `ObjectInfo`
   - **ROI detection** (line 355): `detect_regions_of_interest(image, obj_info, max_points=15, min_distance=25, apply_center_shift=True)` — for highest-confidence detection (or all, depending on `roi_highest_confidence_only`)
   - **Surface segmentation** (lines 388–401): `segment_surfaces_by_plane_fitting()` via FastSAM
   - **Pose estimation** (lines 404–421): `_estimate_object_pose()` via PCA on depth point cloud

### 2.4 `detect_object()` — Single Object Shorthand

**File:** `src/vision/perception_system.py`, lines 443–504

Calls `detect_objects(classes=[target_object], roi_highest_confidence_only=True)`, returns the highest-confidence detection.

### 2.5 `ObjectInfo` Dataclass

**File:** `src/vision/perception_system.py`, lines 20–36

```python
@dataclass
class ObjectInfo:
    id: int                                     # Detection index
    name: str                                   # Class name from YOLO
    bbox: List[int]                             # [x, y, w, h] in camera space
    confidence: float                           # YOLO detection confidence
    image: Optional[np.ndarray] = None          # Original RGB image
    mask: Optional[np.ndarray] = None           # Boolean mask (camera resolution)
    pose: Optional[np.ndarray] = None           # 6D pose [x, y, z, roll, pitch, yaw]
    pixel_pose: Optional[np.ndarray] = None     # 2D pixel center [x, y]
    components: Dict[str, 'ObjectInfo'] = None  # Nested sub-objects
    depth_image: Optional[np.ndarray] = None    # Aligned depth image
    alpha_id: Optional[str] = None              # 3-char random letter ID (e.g. "qkz")
    points: Optional[np.ndarray] = None         # ROI detection results dict
    surface_masks: Dict[str, np.ndarray] = None # Surface segmentation masks
    camera_intrinsics: Dict[str, float] = None  # {fx, fy, cx, cy}
```

---

## 3. Point of Interest Detection

### 3.1 Orchestrator: `detect_regions_of_interest()`

**File:** `src/vision/perception_system.py`, lines 507–936

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `'robust'` | Detection method: `'robust'`, `'contour'`, or `'legacy'` |
| `max_points` | `5` | Maximum points to return |
| `min_distance` | `50` px | Minimum pixel spacing between points |
| `apply_center_shift` | `True` | Shift edge-adjacent points toward centroid |
| `depth_plane_only` | `False` | Use only depth plane detection (v1) |
| `apply_distance_filter` | `False` | Filter by 3D depth range |
| `distance_filter_params` | `{min: 0.0, max: 0.7, threshold: 1.0}` | Distance filter bounds (meters) |

**Flow:**
1. Gets object mask from `obj_info.mask` or creates from bbox
2. **Optional distance filtering** (lines 579–635): `create_distance_mask()` converts raw depth to meters, masks pixels outside `[min_distance, max_distance]`, applies morphological erode/dilate for cleanup
3. **Method dispatch:**
   - `'robust'` (lines 641–713) → `RobustInteractionDetector.detect_interaction_points()` with `fast_mode=True, edge_threshold=10.0, shift_factor=0.2`
   - `'contour'` (lines 716–873) → Contour-based detection with `cv2.approxPolyDP`
   - `'legacy'` → `_detect_regions_legacy()`

**Returns:**
```python
{
    'keypoints': List[cv2.KeyPoint],
    'descriptors': None,
    'pixel_coords': List[(x, y)],
    'scores': List[float],
    'object_name': str,
    'method': str,
    'mask_area': int,
    'ids': List[str],                    # Random 3-letter alpha IDs
    'point_types': List[str],            # InteractionType values
    'interaction_points': List[InteractionPoint]
}
```

### 3.2 `RobustInteractionDetector` (v2) — Primary Detector

**File:** `src/vision/interaction_point_detector_v2.py`, lines 125–1853

**Constructor** (lines 125–179):
- `min_distance: int = 40` px
- `cascade_enabled: bool = True` (early termination if geometry is confident)
- `resolution_scale: float = 1.0`
- Contains `DFormerTiny` ML model (RGB-D fusion via cross-modal attention)
- Cascade config: `geometric_confidence_threshold=0.75`, `max_texture_complexity=0.6`, `min_edge_strength=0.02`, `max_points=30`

**`detect_interaction_points()`** (lines 215–425) uses a tiered architecture:

#### Tier 1: Geometric Features (lines 299–323)

Calls `_detect_geometric_features(depth, mask)` which runs **5 sub-detectors**:

**1. Protrusion Detection** — `_detect_protrusions()` (lines 487–540)
- Multi-scale analysis with kernel sizes `[7, 11, 15]`
- For each kernel: local mean via `cv2.filter2D`, protrusion map = `depth - local_mean`
- Adaptive threshold: `1.5 * local_std + 0.005` (5mm base)
- Local maxima found via morphological dilation
- Validates via `_is_stable_feature()`: depth range 3mm–10cm, std > 1mm
- Creates `HANDLE` type points, score = `min(1.0, prominence * 20)`

**2. Indentation Detection** — `_detect_indentations()` (lines 542–632)
- Multi-scale with kernels `[(5, "small"), (11, "medium"), (17, "large")]`
- Gaussian kernel for smoother local mean
- Detects depressions (`local_mean - depth`) and protrusions (`depth - local_mean`)
- Adaptive threshold: `max(percentile_90, mean + 1.5 * std)`
- `_process_depth_features()` (lines 634–682): Connected component analysis, quality score:
  ```
  score = (prominence * 0.5 + compactness * 0.3 + consistency_inv * 0.2) * (area/300) * 0.8
  ```
- Area filtering per scale: small 10–200px, medium 50–800px, large 200–2000px
- Internal spatial NMS via `_apply_spatial_nms()` with min_distance=15

**3. Surface Normal Analysis** — `_detect_surface_normals()` (lines 703–781)
- Parameters: `patch_size=9, min_area=50, normal_threshold=0.10`
- Calls `_compute_surface_normals()` (lines 783–821) — see [Section 5.1](#51-per-pixel-normals-v2)
- `_segment_surfaces_by_normals()` (lines 823–850): Region growing via flood fill, dot-product threshold for normal similarity
- Surface classification by normal direction:
  - `camera_alignment > 0.85` → `GRASP_SURFACE` (horizontal)
  - `side_alignment > 0.75` → `GRASP_EDGE` (vertical)
- Validates flat surfaces: depth std < 2cm, depth range < 6cm

**4. Depth Edge Detection** — `_detect_depth_edges()` (lines 994–1039)
- Scharr operator: `cv2.Scharr(depth, CV_64F, 1, 0)` for x/y gradients
- Strong edges: 90th percentile threshold
- Non-maximum suppression via 8-direction edge thinning (`_non_max_suppression_edges()`, lines 1549–1588)
- Graspability validation via `_is_graspable_edge()`: `0.002 < std < 0.05`, range > 0.005

**5. Surface Boundary Detection** — `_detect_surface_boundaries()` (lines 1041–1116)
- Creates coverage map (25px radius around existing points)
- Identifies uncovered regions with valid normals
- Segments and adds boundary points for spatial completeness

#### Early Termination (lines 326–343)

If cascade is enabled and geometric features yield enough high-confidence points above `geometric_confidence_threshold = 0.75`, Tier 2 is skipped.

#### Tier 2: RGB-D Fusion via DFormerTiny (lines 346–357)

**Currently disabled** to avoid CUDA memory issues.

The DFormerTiny model (lines 53–122) architecture:
- RGB encoder: `Conv2d(3→32→64)` with BatchNorm + ReLU, stride 2
- Depth encoder: `Conv2d(1→32→64)` with BatchNorm + ReLU, stride 2
- Cross-modal attention: `nn.MultiheadAttention(64, num_heads=4)`
- 6 interaction type predictors: `Conv2d(64→32→1)` per type

### 3.3 `RobustInteractionDetector` (v1) — Fallback Detector

**File:** `src/vision/interaction_point_detector.py`, lines 30–2800+

**`detect_interaction_points()`** (lines 41–180) has two modes:

**Depth-plane-only mode** (default, lines 78–116):
Uses `_detect_depth_planes_with_distribution()` exclusively — the primary v1 mode.

**Multi-method mode** (lines 118–180):

*Fast mode (3 methods):*
- `_detect_geometric_features()` (line 290): Curvature-based via `_calculate_curvature_points()` — smoothed contour, angle between neighboring vectors, curvature > 0.2 creates `GRASP_EDGE`
- `_detect_contour_features()` (line 315): `cv2.approxPolyDP` corners (score 0.8), convex hull defects for handles (defect distance > 3000)
- `_detect_corners_edges(corners_only=True)` (line 357): Harris corners (`cv2.cornerHarris(gray, 2, 3, 0.04)`, threshold 0.01 * max), FAST corners (threshold=10)

*Full mode adds:*
- `_detect_surface_features()` (line 394): Sobel gradients, flat regions below 25th percentile gradient magnitude → `GRASP_SURFACE`
- `_detect_affordances()` (line 426): Hough circle detection (`dp=1, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100`) → `HANDLE`
- `_detect_3d_features()` (line 453): If depth available, runs:
  - `_detect_ridges_valleys()` (line 832): Hessian matrix (2nd-order Sobel), mean curvature H, Gaussian curvature K; ridges: H > 75th pct and K > 0; valleys: H < 25th pct and K > 0
  - `_detect_depth_peaks()` (line 879): Morphological TOPHAT with elliptical kernel (15,15), peaks > 80th percentile
  - `_detect_stable_planes()` (line 917): Multi-scale normal computation, surface segmentation by normal similarity, center via 4-method consensus (geometric, stability, distance, depth centers with weights 0.3, 0.3, 0.2, 0.2)
  - `_detect_depth_discontinuities()` (line 954): Sobel gradients + Canny (50/150 thresholds)
  - `_detect_depth_planes_with_distribution()` (line 1840): Main depth plane detector
  - `_add_surface_normals()` (line 991): Post-processing to add approach angles

### 3.4 `_detect_depth_planes_with_distribution()` — Primary v1 POI Method

**File:** `src/vision/interaction_point_detector.py`, lines 1840–1994

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `depth_shift_threshold` | 0.015 (1.5cm) | Minimum depth discontinuity |
| `min_region_area` | 80 px | Minimum region size |
| `points_per_region` | 3 | Base points per region |
| `gaussian_blur_size` | 3 | Smoothing kernel size |
| `gradient_threshold` | 0.01 | Depth gradient threshold |

**Algorithm:**
1. Gaussian blur of masked depth for smoothing
2. `_detect_depth_shift_regions()` (line 1906): Computes depth gradients, finds significant discontinuities above threshold
3. `_cluster_depth_regions()` (line 1914): Groups nearby shift regions into coherent clusters
4. **Adaptive point distribution** (lines 1944–1971): For each valid region:
   - `size_ratio = area / max_area`
   - `adaptive_points = 1 + size_ratio * 7` (1–8 points per region)
   - `_create_distributed_points_on_region()` places points within the region
5. **Border points** (line 1974): `_create_segmentation_border_points()` adds 6 points along the object boundary
6. **Fallback** (lines 1981–1989): If no depth shift regions found, uses `_create_distributed_points_single_plane()`

### 3.5 Advanced 3D Features (v2)

**File:** `src/vision/interaction_point_detector_v2.py`, lines 1145–1222

If PCL or Open3D is available:
- Converts depth image to 3D point cloud
- Estimates normals via KDTree search (PCL: `KSearch=10`, Open3D: `radius=0.1, max_nn=30`)
- Finds grasp candidates where z-normal < 0.3 (perpendicular to viewing direction)

### 3.6 Data Classes

**`InteractionType` Enum** (v1: line 7, v2: line 29):
```python
class InteractionType(Enum):
    GRASP_EDGE = "grasp_edge"
    GRASP_SURFACE = "grasp_surface"
    PUSH_POINT = "push_point"
    HANDLE = "handle"
    PIVOT = "pivot"
    CONTACT = "contact"
```

**`InteractionPoint` Dataclass** (v2, lines 38–49):
```python
@dataclass
class InteractionPoint:
    x: int                                  # Pixel x coordinate
    y: int                                  # Pixel y coordinate
    score: float                            # Composite quality score
    interaction_type: InteractionType       # Classified type
    confidence: float                       # Detection confidence
    approach_angle: Optional[float] = None  # Suggested approach direction (degrees)
    grasp_width: Optional[float] = None     # Estimated grasp aperture (pixels)
    stability: float = 0.5                  # Grasp stability estimate [0,1]
    accessibility: float = 0.5              # How accessible the point is [0,1]
    detection_method: str = "unknown"       # Which sub-detector found this point
```

**`PointOfInterest` Dataclass** (`src/skills/skill_generator.py`, lines 18–29):
```python
@dataclass
class PointOfInterest:
    label: str                                      # 3-letter alpha ID
    position: Tuple[float, float]                   # Normalized (x, y) in [0,1]
    description: str = ""                           # Human-readable description
    pixel_coords: Tuple[Tuple[float, float]] = ()   # Original pixel coords
    detection_method: str = "unknown"               # Source sub-detector
    interaction_type: str = "unknown"               # InteractionType value
    confidence: float = 1.0                         # Detection confidence
    score: float = 1.0                              # Composite score
    stability: float = 0.5                          # Stability estimate
    accessibility: float = 0.5                      # Accessibility estimate
```

---

## 4. Filtering and Scoring

### 4.1 Multi-Criteria Scoring (v2)

**File:** `src/vision/interaction_point_detector_v2.py`, lines 1381–1442

`_score_and_rank_points(points, image, depth, mask)`:

```
total_score = base_score      * 0.40    # Original detection score
            + texture_penalty * 0.25    # Lower texture = higher score (Laplacian)
            + depth_reliability * 0.20  # Fewer depth edges = more reliable (Sobel)
            + boundary_distance * 0.15  # Optimal = 15px from boundary (exp. decay)
            + type_bonus       * 0.10   # HANDLE=0.4, PUSH=0.3, GRASP_EDGE=0.2,
                                        #   GRASP_SURFACE=0.1, CONTACT=0.0
```

**Texture penalty** (lines 1394–1397): Computed via Laplacian filter + Gaussian blur on the RGB image. Points in low-texture regions score higher (more reliable for depth-based grasping).

**Depth reliability** (lines 1400–1402): Sobel edge magnitude on depth. Points in smooth depth regions score higher (more accurate 3D estimation).

**Boundary distance** (lines 1420–1425): `cv2.distanceTransform(mask, DIST_L2, 5)`. Optimal distance is 15px from mask edge; score decays exponentially for closer/farther points.

### 4.2 Multi-Criteria Scoring (v1)

**File:** `src/vision/interaction_point_detector.py`, lines 544–586

`_score_and_rank_points(candidates, mask, image_shape)`:

```
final_score = base_score         * 0.25
            + boundary_score     * 0.15    # cv2.distanceTransform, optimal = 10px
            + stability_score    * 0.15    # Neighborhood mask coverage (15px window)
            + accessibility_score * 0.15   # Distance from centroid, normalized
            + type_bonus         * 0.10    # HANDLE=0.95, GRASP_SURFACE=0.95,
                                           #   GRASP_EDGE=0.85, PUSH=0.8, etc.
            + 3d_geometry_score  * 0.20    # Approach angle (optimal 30-60 deg)
                                           #   + grasp width (optimal 10-100px)
                                           #   + surface bonuses
```

### 4.3 Non-Maximum Suppression (NMS)

**v2 — Greedy NMS** (`interaction_point_detector_v2.py`, lines 1616–1648):
1. Sort all points by score, descending
2. For each point, keep it if its Euclidean distance to every already-kept point >= `min_distance`
3. Simple, fast, deterministic

**v1 — Two-Stage NMS** (`interaction_point_detector.py`, lines 647–698):
1. **Spatial grouping:** `_group_points_by_spatial_regions()` — sorted by score descending, groups points within `min_distance` into regions
2. **Depth-consistent selection:** `_select_best_point_by_depth_consistency()` — picks the best point per group using `0.7 * depth_consistency + 0.3 * score`, where depth consistency = `1.0 - (relative_std * 10)` (std/mean in 15px window)

### 4.4 Center-Shift

Prevents interaction points from landing too close to the object edge, which causes unreliable grasps.

**v2 Implementation** (`interaction_point_detector_v2.py`, lines 1650–1746):
1. Compute edge distances via `cv2.distanceTransform(mask, DIST_L2, 5)`
2. Find centroid via `np.mean(np.where(mask))`
3. For each point with `edge_distance < edge_threshold`:
   - `shift_amount = shift_factor * (1.0 - edge_distance / edge_threshold)` — stronger shift for points closer to the edge
   - New position: `new_x = x + dx * shift_amount`, `new_y = y + dy * shift_amount` (toward centroid)
   - Confidence reduced by 5% for shifted points
   - `detection_method` appended with `"+center_shift"`

**v1 Implementation** (`interaction_point_detector.py`, lines 184–288):
1. Same distance transform and centroid computation
2. For points with `edge_distance < edge_threshold` (default 15px):
   - `edge_ratio = (threshold - distance) / threshold`
   - `shift_distance = max(min_shift=5, min(max_shift=20, dist_to_center * shift_factor * edge_ratio))`
   - Shift along vector toward centroid
   - Validates the new position still lies on the mask

**Called with:** `edge_threshold=10.0, shift_factor=0.2` (from `detect_regions_of_interest`, line 668–669).

---

## 5. Surface Normal Calculation

### 5.1 Per-Pixel Normals (v2)

**File:** `src/vision/interaction_point_detector_v2.py`, lines 783–821

`_compute_surface_normals(depth, mask)`:

1. Convert depth to `float32`
2. Compute Sobel gradients:
   ```
   grad_x = cv2.Sobel(depth, CV_32F, 1, 0, ksize=3)
   grad_y = cv2.Sobel(depth, CV_32F, 0, 1, ksize=3)
   ```
3. For each masked pixel, construct two tangent vectors and take the cross product:
   ```
   t1 = (1, 0, dx)    # tangent in x direction
   t2 = (0, 1, dy)    # tangent in y direction
   normal = cross(t1, t2) = (-dx, -dy, 1.0)
   ```
4. Normalize to unit length
5. Flip to face camera: if `normal[2] > 0`, negate the vector
6. Returns `(H, W, 3)` normal map

### 5.2 Multi-Scale Normals (v1)

**File:** `src/vision/interaction_point_detector.py`, lines 1096–1127

`_calculate_surface_normals(depth, mask)`:

1. Edge-preserving smoothing: `cv2.bilateralFilter(depth, 5, 50, 50)`
2. Multi-scale Sobel gradients:
   - ksize=3 (weight 0.7)
   - ksize=5 (weight 0.3)
   - Combined: `grad_x = 0.7 * grad_x_3 + 0.3 * grad_x_5`
3. Same cross-product method as v2

### 5.3 Object-Level Surface Normal (RANSAC/PCA)

**File:** `src/vision/perception_system.py`, lines 2178–2345

`calculate_surface_normal(mask, depth_image)` supports two methods:

**RANSAC (default):**
1. Convert masked depth pixels to 3D points via pinhole model:
   ```
   x_3d = (pixel_x - cx) * depth / fx
   y_3d = (pixel_y - cy) * depth / fy
   z_3d = depth
   ```
2. Subsample to 1000 points
3. Run RANSAC for `ransac_iterations=100`:
   - Sample 3 random points
   - Compute normal via cross product of two edge vectors
   - Plane equation: `ax + by + cz + d = 0`
   - Count inliers within `max_plane_distance=0.01` meters
4. Confidence = `inlier_count / total_points`
5. Orient normal toward camera (positive z)

**PCA:**
1. Convert to 3D point cloud
2. Compute covariance matrix of 3D points
3. Eigendecomposition: normal = eigenvector with smallest eigenvalue (least variance direction)
4. Confidence from planarity = `1.0 - (eigenvalue_0 / eigenvalue_1)`

### 5.4 Surface Classification by Normal

**File:** `src/vision/interaction_point_detector_v2.py`, lines 880–903

`_classify_surface_by_normal(normal)`:

| Condition | Classification |
|-----------|---------------|
| `dot(normal, [0,0,-1]) > 0.85` | `GRASP_SURFACE` (horizontal/top-facing) |
| `dot(normal, [1,0,0]) > 0.75` | `GRASP_EDGE` (vertical/side-facing) |
| `max_alignment > 0.6` | `CONTACT` (angled surface) |
| Otherwise | `PUSH_POINT` |

### 5.5 Approach Angle from Normal

**File:** `src/vision/interaction_point_detector_v2.py`, lines 905–917

| Surface Type | Approach Angle |
|-------------|---------------|
| `GRASP_SURFACE` | `0.0` degrees (top-down) |
| `GRASP_EDGE` | `arctan2(ny, nx) + 90` degrees (perpendicular) |
| Other | `arctan2(ny, nx) + 180` degrees (opposite to normal) |

### 5.6 Normal-Based Surface Segmentation

**File:** `src/vision/interaction_point_detector_v2.py`, lines 823–850

`_segment_surfaces_by_normals(normals, mask, threshold)`:
- Region growing via flood fill
- Dot-product threshold determines if adjacent normals belong to the same surface
- Returns labeled surface map where each region has a consistent normal direction

---

## 6. Pose Estimation

### 6.1 Object-Level 6D Pose

**File:** `src/vision/perception_system.py`, lines 1443–1582

`_estimate_object_pose(mask, depth_image)`:

1. **Depth filtering:** `MIN_DEPTH=0.05m, MAX_DEPTH=2.0m`, plus 2-sigma statistical outlier rejection
2. Subsample to 1000 points
3. Back-project to 3D via pinhole model
4. **Position:** Centroid = mean of 3D points
5. **Orientation:** PCA on 3D points → eigenvalues/eigenvectors → roll/pitch/yaw from eigenvectors
6. Returns `[x, y, z, roll, pitch, yaw]` and reprojected pixel center `(pixel_x, pixel_y)`

### 6.2 Point-Level 3D Estimation

**File:** `src/vision/perception_system.py`, lines 1585–1768

`_estimate_point_pose(point, depth_image, region_size=1)`:

1. Samples `(2*region_size+1)^2` depth values around the pixel
2. Filters: `MIN_DEPTH=0.05m, MAX_DEPTH=2.0m`
3. **Distance-weighted averaging:** Closer pixels contribute more to the depth estimate
4. **Optional edge-aware filtering:** Bilateral-style depth weights suppress depth discontinuities
5. **Outlier rejection:** MAD (Median Absolute Deviation), threshold = `3.0 * MAD`
6. **3D conversion:**
   ```
   x = (pixel_x - cx) * depth / fx
   y = (pixel_y - cy) * depth / fy
   z = depth
   ```
7. Optional temporal smoothing with `smoothing_factor`

---

## 7. Perception-to-Skill Bridge

### 7.1 `SkillHandler.instantiate_skill()`

**File:** `src/skills/skill_handler.py`, lines 81–213

1. Calls `perception_system.detect_object()` if no pre-detected `ObjectInfo`
2. Reads `object_info.points` (the ROI results dict)
3. For each point in `pixel_coords`, creates a `PointOfInterest`:
   - Normalizes coords: `norm_x = pixel_x / image_width`, `norm_y = pixel_y / image_height`
   - Extracts metadata from `InteractionPoint` objects with fallback chains:
     - `detection_methods` → `detection_method` → `"unknown"`
     - `interaction_types` → `types` → `"unknown"`
     - `confidences` → `confidence` field → `1.0`
     - `interaction_points` list → attribute access for `stability`, `accessibility`, etc.
4. Passes `points_of_interest` dict + annotated image to `SkillGenerator.generate_skill()` (LLM-based)

### 7.2 Skill Instantiation

`_instantiate_skill()` converts the abstract skill to `ExecutableAction` objects:
- For each labeled point selected by the LLM, converts normalized coords back to pixels
- Converts pixels to 3D via `_estimate_point_pose()`
- Determines grasp type (top-down vs side) from surface normals
- Packages into `InstantiatedSkill` with `action_sequence`

---

## 8. Configuration Reference

### Detection Parameters

| Parameter | Value | File | Description |
|-----------|-------|------|-------------|
| YOLO confidence | 0.01 (detect) / 0.5 (default) | `perception_system.py:134,58` | Detection threshold |
| YOLOWorld model | `yolov8x-worldv2.pt` | `yolo.py:12` | Open-vocabulary detector |
| FastSAM model | `FastSAM-x.pt` | `yolo.py:13` | Instance segmentation |
| FastSAM conf/iou | 0.4 / 0.9 | `yolo.py:146` | Segmentation thresholds |
| Depth scale | 0.001 m/unit | `perception_system.py:57` | Raw depth to meters |
| Min/Max depth | 0.05m / 2.0m | `perception_system.py:1489-1490` | Valid depth range |

### Point of Interest Parameters

| Parameter | Value | File | Description |
|-----------|-------|------|-------------|
| ROI max_points | 15 | `perception_system.py:355` | Max returned points |
| ROI min_distance | 25 px | `perception_system.py:355` | Min spacing (orchestrator) |
| NMS min_distance (v2) | 40 px | `interaction_point_detector_v2.py:131` | Min spacing (detector) |
| Center-shift edge_threshold | 10 px | `perception_system.py:668` | Edge proximity trigger |
| Center-shift factor | 0.2 | `perception_system.py:669` | Shift magnitude |
| Cascade confidence threshold | 0.75 | `interaction_point_detector_v2.py:163` | Early termination |
| Cascade max_points | 30 | `interaction_point_detector_v2.py:166` | Max pre-filter points |

### Geometric Sub-Detector Parameters

| Parameter | Value | File | Description |
|-----------|-------|------|-------------|
| Protrusion kernels | [7, 11, 15] | `interaction_point_detector_v2.py:497` | Multi-scale analysis |
| Protrusion threshold | 1.5*std + 5mm | `interaction_point_detector_v2.py:512` | Adaptive threshold |
| Indentation kernels | [5, 11, 17] | `interaction_point_detector_v2.py:555` | Multi-scale analysis |
| Indentation area (small) | 10–200 px | `interaction_point_detector_v2.py:590` | Area filtering |
| Indentation area (medium) | 50–800 px | `interaction_point_detector_v2.py:591` | Area filtering |
| Indentation area (large) | 200–2000 px | `interaction_point_detector_v2.py:592` | Area filtering |
| Normal patch_size | 9 | `interaction_point_detector_v2.py:711` | Surface normal patch |
| Flat surface max_std | 0.02m | `interaction_point_detector_v2.py:954` | Flatness validation |
| Flat surface max_range | 0.06m | `interaction_point_detector_v2.py:955` | Flatness validation |

### Surface Normal Parameters

| Parameter | Value | File | Description |
|-----------|-------|------|-------------|
| RANSAC iterations | 100 | `perception_system.py:2184` | Normal estimation |
| RANSAC max_plane_distance | 0.01m | `perception_system.py:2183` | Inlier threshold |
| Camera alignment threshold | 0.85 | `interaction_point_detector_v2.py:885` | GRASP_SURFACE classification |
| Side alignment threshold | 0.75 | `interaction_point_detector_v2.py:889` | GRASP_EDGE classification |

### Scoring Weights

**v2 Scoring:**
| Component | Weight | Description |
|-----------|--------|-------------|
| Base score | 0.40 | Original detection confidence |
| Texture penalty | 0.25 | Low texture = better |
| Depth reliability | 0.20 | Smooth depth = better |
| Boundary distance | 0.15 | Optimal 15px from edge |
| Type bonus | 0.10 | Favors HANDLE, PUSH types |

**v1 Scoring:**
| Component | Weight | Description |
|-----------|--------|-------------|
| Base score | 0.25 | Original detection confidence |
| Boundary score | 0.15 | Optimal 10px from edge |
| Stability score | 0.15 | Neighborhood mask coverage |
| Accessibility score | 0.15 | Distance from centroid |
| Type bonus | 0.10 | Favors HANDLE, GRASP_SURFACE |
| 3D geometry score | 0.20 | Approach angle + grasp width |

### Depth Plane Detection (v1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| depth_shift_threshold | 0.015m (1.5cm) | Minimum depth discontinuity |
| min_region_area | 80 px | Minimum valid region |
| points_per_region | 3 | Base point count |
| gaussian_blur_size | 3 | Smoothing kernel |
| gradient_threshold | 0.01 | Depth gradient cutoff |
| Adaptive points range | 1–8 per region | Scales with region area |
| Border points | 6 | Points along object boundary |
