# Data Recording System for Task Planner

## Overview

The data recording system captures comprehensive data from task execution runs, including:
- Natural language task description
- Task decomposition into skill commands
- Generated skills and LLM responses
- Points of interest and surface image frames
- Timing measurements for planning, inference, and execution phases
- Robot execution results and failure messages
- User feedback on task success/failure with explanations

## Quick Start

### Basic Usage

```python
from cognitive_bt_framework.src.task_planner import TaskPlanner

# Development mode (default - no data recording)
planner = TaskPlanner(
    robot_ip="192.168.1.224",
    use_llm=True
    # enable_data_recording=False is the default
)

# Production mode (with data recording)
planner = TaskPlanner(
    robot_ip="192.168.1.224", 
    use_llm=True,
    enable_data_recording=True
)

# Execute a task - recording depends on planner setting
results = planner.execute_task(
    task_description="move the bottle to the counter and open it",
    execute_on_robot=True
)

# Or override recording for specific tasks
results = planner.execute_task(
    task_description="important task",
    execute_on_robot=True,
    record_data=True  # Force recording on for this task
)
```

### Data Recording Control

```python
# Toggle recording at runtime
planner.set_data_recording(True)   # Enable recording
planner.set_data_recording(False)  # Disable recording

# Check current status
if planner.is_data_recording_enabled():
    print("Recording is active")

# Per-task control
planner.execute_task("task1", record_data=True)   # Force on
planner.execute_task("task2", record_data=False)  # Force off  
planner.execute_task("task3", record_data=None)   # Use default
```

## Data Schema

Each session records the following data:

### Core Task Information
- `natural_language_task`: Original task description
- `task_decomposition`: List of skill commands
- `skills_generated`: List of generated skill objects
- `llm_responses`: Raw LLM responses for task decomposition and skill generation

### Visual Data
- `environment_image_path`: Path to captured environment image
- `surface_images_paths`: Paths to cropped object/surface images
- `points_of_interest`: List of detected interaction points with coordinates

### Timing Data
- `planning_duration`: Time spent on task decomposition
- `inference_duration`: Time spent on skill generation
- `execution_duration`: Time spent on robot execution
- `total_duration`: Total task execution time

### Execution Results
- `execution_success`: Boolean indicating if execution succeeded
- `failure_messages`: List of failure messages during execution
- `robot_errors`: List of robot-specific errors

### User Feedback
- `user_success_rating`: Boolean user assessment of task success
- `user_explanation`: Text explanation from user about execution

### Technical Details
- `session_id`: Unique identifier for the session
- `timestamp`: ISO timestamp of session start
- `robot_ip`: Robot IP address used
- `camera_type`: Camera type (ZED, RealSense, etc.)
- `execution_mode`: "real" or "simulation"

## File Structure

The data recording system creates the following directory structure:

```
task_execution_data/
├── sessions/
│   ├── session_uuid1.json    # Session data files
│   ├── session_uuid2.json
│   └── ...
└── images/
    ├── session_uuid1_environment.jpg      # Environment images
    ├── session_uuid1_surface_bottle.jpg   # Surface/object images
    ├── session_uuid2_environment.jpg
    └── ...
```

## Advanced Usage

### Standalone Data Recorder

```python
from cognitive_bt_framework.src.data_recorder import DataRecorder
import numpy as np

# Initialize recorder
recorder = DataRecorder(data_dir="my_custom_data")

# Start a session
session_id = recorder.start_recording_session(
    natural_language_task="custom task",
    robot_ip="192.168.1.224",
    camera_type="realsense",
    execution_mode="real"
)

# Record different phases manually
recorder.record_planning_complete(["detect_object bottle", "open bottle"])

# Record images
image = np.zeros((480, 640, 3), dtype=np.uint8)  # Your image data
recorder.record_environment_image(image)

# Record points of interest
points_data = {
    'pixel_coords': [(320, 240), (350, 200)],
    'ids': ['A', 'B'], 
    'scores': [0.9, 0.8],
    'detection_methods': ['fastsam', 'fastsam'],
    'interaction_types': ['grasp', 'twist'],
    'confidences': [0.95, 0.85]
}
recorder.record_points_of_interest(points_data)

# Record execution phases
recorder.record_inference_start()
# ... skill generation happens ...
recorder.record_inference_complete(skills_list, llm_responses_dict)

recorder.record_execution_start()
# ... robot execution happens ...
recorder.record_execution_complete(success=True, failure_messages=[], robot_errors=[])

# Finalize (includes user feedback collection)
session_path = recorder.finalize_session()
```

### Analyzing Recorded Data

```python
from cognitive_bt_framework.src.data_recorder import DataRecorder
import json

# Get session statistics
recorder = DataRecorder(data_dir="task_execution_data")
stats = recorder.get_session_stats()

print(f"Total sessions: {stats['total_sessions']}")
print(f"Success rate: {stats['execution_success_rate']:.2%}")
print(f"Average planning time: {stats['average_planning_time']:.3f}s")
print(f"Most common failures: {stats['most_common_failures']}")

# Load and analyze specific session
session_file = "task_execution_data/sessions/some_session_id.json"
with open(session_file, 'r') as f:
    session_data = json.load(f)
    
print(f"Task: {session_data['natural_language_task']}")
print(f"Commands: {len(session_data['task_decomposition'])}")
print(f"Points detected: {len(session_data['points_of_interest'])}")
print(f"User success: {session_data['user_success_rating']}")
```

## User Feedback Collection

When a task completes, the system automatically prompts for user feedback:

```
============================================================
TASK EXECUTION FEEDBACK  
============================================================

Did the task execute successfully? (y/n): y

Please provide details about the execution:
(Enter your explanation, then press Enter twice to finish)
The robot successfully moved the bottle to the counter. 
Opening the bottle worked perfectly. 
Overall execution was smooth with no issues.


Feedback recorded: Success
Notes: The robot successfully moved the bottle to the counter...
```

## Testing

Run the test suite to verify the data recording system:

```bash
# Test the complete workflow
python test_data_recording_simple.py

# Test with robot dependencies (requires robot connection)
python test_data_recording.py
```

## Configuration

### Recording Toggle Options

```python
# Method 1: Set at initialization (recommended)
planner = TaskPlanner(enable_data_recording=True)   # Production
planner = TaskPlanner(enable_data_recording=False)  # Development (default)

# Method 2: Runtime toggle
planner = TaskPlanner()  # Starts with recording disabled
planner.set_data_recording(True)   # Enable for experiments
planner.set_data_recording(False)  # Disable for debugging

# Method 3: Per-task override
planner = TaskPlanner()  # Recording disabled by default
planner.execute_task("regular task")                    # Not recorded
planner.execute_task("important task", record_data=True)  # Recorded
```

### Custom Data Directory

```python
# Use custom directory for data storage
recorder = DataRecorder(data_dir="/path/to/my/data")
```

### Image Quality Settings

The system automatically handles image compression and storage. Images are saved as JPEG with 85% quality for efficient storage while maintaining visual fidelity.

## Data Privacy & Storage

- All data is stored locally in the specified directory
- No data is transmitted to external services
- Image files are compressed for efficient storage
- Session files are human-readable JSON format
- Raw LLM responses are stored for analysis but contain no personal data

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure the data directory is writable
2. **Disk space**: Monitor disk usage as images can accumulate over time
3. **Session interruption**: If a session is interrupted, data up to that point is preserved

### Performance Considerations

- Image recording adds ~50-100ms per session
- JSON session files are typically 3-5KB each
- One session with images uses ~200-500KB storage

## Integration with Existing Code

The data recording system is designed to be minimally invasive:

- Existing task planner code works unchanged
- Recording can be disabled with a single parameter
- No performance impact when disabled
- Backward compatible with existing workflows

## Future Enhancements

Potential future additions:
- Video recording of robot execution
- Automatic data analysis and insights
- Export to common ML training formats
- Integration with experiment tracking tools
- Compressed data archiving