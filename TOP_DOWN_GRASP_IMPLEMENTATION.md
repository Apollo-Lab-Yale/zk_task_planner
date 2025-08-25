# Top-Down Grasp Positioning Implementation

## Overview

This implementation adds specialized positioning logic for top-down grasps that finds the topmost point within an object mask region and centers the grasp on that point on the depth surface.

## Key Features

### 1. Top-Down Grasp Detection
- Automatically detects when a grasp is configured as top-down (`is_top_down_grasp=True`)
- Uses different positioning logic compared to side grasps
- Optimized for vertical approach grasps

### 2. Object Mask Integration
- Accepts object masks from the perception system
- Constrains search to only the object region
- Falls back to circular search region if no mask is provided

### 3. Topmost Point Detection
- Finds the highest Y-coordinate point within the object mask
- Corresponds to the highest point on the object surface
- Handles multiple points with same Y-coordinate by choosing closest to target X

### 4. Depth Surface Centering
- Uses depth clustering to identify the main object surface
- Applies depth threshold to focus on surface points
- Centers grasp on the detected topmost point

## Implementation Details

### New Method: `adjust_tcp_pose_for_top_down_grasp`

```python
def adjust_tcp_pose_for_top_down_grasp(self, target_position, object_mask=None, 
                                       search_radius_m=0.03, depth_threshold_ratio=0.05, 
                                       cluster_min_size=5):
```

**Parameters:**
- `target_position`: Original target position [x, y, z] in camera coordinates
- `object_mask`: Binary mask of the object region (optional)
- `search_radius_m`: Physical search radius in meters (default 3cm for top-down)
- `depth_threshold_ratio`: Ratio for depth clustering threshold (default 5%)
- `cluster_min_size`: Minimum cluster size for consideration (default 5 pixels)

**Returns:**
- `adjusted_position`: Position adjusted to center on the topmost point

### Algorithm Steps

1. **Project Target to Pixels**: Convert 3D target position to 2D pixel coordinates
2. **Define Search Region**: Use object mask if provided, otherwise create circular region
3. **Extract Valid Depth**: Get depth values within the search region
4. **Surface Detection**: Find points near the surface using depth threshold
5. **Topmost Point Selection**: Find the highest Y-coordinate point (topmost)
6. **Coordinate Conversion**: Convert back to 3D camera coordinates
7. **Position Adjustment**: Return adjusted position centered on topmost point

### Integration with Existing System

The method is integrated into the existing grasp execution pipeline:

```python
# In execute_action method
if action.is_top_down_grasp:
    self.logger.info("Applying top-down grasp positioning adjustment")
    object_mask = None
    if hasattr(action, 'object_info') and action.object_info is not None:
        object_mask = action.object_info.mask
    
    adjusted_position = self.adjust_tcp_pose_for_top_down_grasp(
        target_position, 
        object_mask=object_mask,
        search_radius_m=0.03,
        depth_threshold_ratio=0.05,
        cluster_min_size=5
    )
else:
    # Use existing depth centering for side grasps
    adjusted_position = self.adjust_tcp_pose_for_depth_centering(target_position, search_radius_m=search_radius)
```

## Usage Examples

### Basic Usage
```python
# Create an ExecutableAction with top-down grasp
action = ExecutableAction(
    action_type='move_gripper_to_pose',
    position=[0.1, 0.2, 0.5],
    orientation=[0, 0, -1, 0],  # Top-down orientation
    pixel_position=(320, 240),
    parameters={'speed': 'medium'},
    is_top_down_grasp=True,
    is_side_grasp=False,
    object_info=object_info  # Contains object mask
)

# Execute the action
success = skill_executor.execute_action(action)
```

### With Object Mask
```python
# Get object info from perception system
object_info = perception_system.detect_object("cup", image, depth_image)

# Create action with object mask
action = ExecutableAction(
    action_type='move_gripper_to_pose',
    position=target_position,
    orientation=top_down_orientation,
    pixel_position=object_info.pixel_pose,
    parameters={},
    is_top_down_grasp=True,
    is_side_grasp=False,
    object_info=object_info  # Contains mask for precise positioning
)
```

## Testing

A comprehensive test script is provided (`test_top_down_grasp.py`) that:

1. **Creates synthetic test data** with known object shapes and depth profiles
2. **Tests the mock implementation** to verify algorithm correctness
3. **Tests with real skill executor** to verify integration
4. **Validates adjustment magnitudes** are reasonable (< 0.5m)

### Running Tests
```bash
python test_top_down_grasp.py
```

Expected output:
```
✅ Test passed: Adjustment is reasonable
```

## Configuration Parameters

### Search Radius
- **Side grasps**: 5cm (0.05m) - larger area for lateral positioning
- **Top-down grasps**: 3cm (0.03m) - smaller, more precise area

### Depth Threshold
- **Side grasps**: 10% of average depth
- **Top-down grasps**: 5% of average depth - more precise surface detection

### Cluster Size
- **Side grasps**: 10 pixels minimum
- **Top-down grasps**: 5 pixels minimum - allows smaller surface regions

## Benefits

1. **Improved Grasp Success**: Centers on the highest point for better contact
2. **Object-Aware Positioning**: Uses object masks for precise region constraints
3. **Robust Depth Handling**: Handles depth noise and invalid values
4. **Configurable Parameters**: Adjustable for different object types and scenarios
5. **Backward Compatibility**: Falls back gracefully when masks aren't available

## Future Enhancements

1. **Multi-Surface Detection**: Handle objects with multiple graspable surfaces
2. **Grasp Quality Scoring**: Evaluate multiple candidate points
3. **Dynamic Parameter Tuning**: Adjust parameters based on object type
4. **Visual Feedback**: Show topmost point detection in debug visualizations

## Troubleshooting

### Common Issues

1. **Large Adjustments**: Check depth scale and camera calibration
2. **No Valid Points**: Verify object mask and depth image quality
3. **Incorrect Topmost Point**: Check Y-coordinate interpretation for your camera

### Debug Information

The method provides extensive debug logging:
```
DEBUG - Top-down grasp adjustment: target_position=[0.0, 0.0, 1.0]
DEBUG - Using provided object mask, shape: (480, 640)
DEBUG - Valid depth points in object mask: 15000
DEBUG - Topmost point: (320, 190), depth: 775.0 mm
DEBUG - Topmost 3D coordinates: (0.0000, -0.0738, 0.7750)
Top-down grasp adjustment: offset=[0, -0.074, -0.225], magnitude=0.237m
``` 