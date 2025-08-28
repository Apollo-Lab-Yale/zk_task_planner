#!/usr/bin/env python3
"""
Simple demonstration of data recording toggle without robot dependencies
"""

import sys
import os

# Add the cognitive_bt_framework to the path  
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_bt_framework', 'src'))

def demo_toggle_concept():
    """Demonstrate the toggle concept"""
    
    print("="*60)
    print("DATA RECORDING TOGGLE - CONCEPT DEMONSTRATION")  
    print("="*60)
    
    print("""
The data recording system has been enhanced with flexible toggle controls:

1. DEFAULT BEHAVIOR:
   - Data recording is now DISABLED by default
   - This is better for development and testing
   - No performance overhead when disabled
   
2. INITIALIZATION OPTIONS:
   
   # Development mode (default - recording off)
   planner = TaskPlanner()
   
   # Production mode (recording on)  
   planner = TaskPlanner(enable_data_recording=True)
   
3. RUNTIME CONTROL:
   
   # Enable/disable at any time
   planner.set_data_recording(True)   # Turn on
   planner.set_data_recording(False)  # Turn off
   
   # Check current status
   if planner.is_data_recording_enabled():
       print("Recording is active")
   
4. PER-TASK OVERRIDE:
   
   # Override default for specific tasks
   planner.execute_task("task", record_data=True)   # Force recording on
   planner.execute_task("task", record_data=False)  # Force recording off
   planner.execute_task("task", record_data=None)   # Use default setting
   
5. PRACTICAL USAGE PATTERNS:
   
   DEVELOPMENT/TESTING:
   ==================
   planner = TaskPlanner()  # Recording disabled by default
   # Fast execution, no data collection overhead
   
   DATA COLLECTION:
   ================  
   planner = TaskPlanner(enable_data_recording=True)
   # All tasks recorded automatically
   
   SELECTIVE RECORDING:
   ===================
   planner = TaskPlanner()  # Default off
   # Only record important tasks
   planner.execute_task("critical_task", record_data=True)
   
   RUNTIME SWITCHING:
   ==================
   planner = TaskPlanner()
   # Development phase
   planner.execute_task("test1")  # No recording
   planner.execute_task("test2")  # No recording
   
   # Switch to data collection  
   planner.set_data_recording(True)
   planner.execute_task("experiment1")  # Recorded
   planner.execute_task("experiment2")  # Recorded
   
   # Back to development
   planner.set_data_recording(False)
   planner.execute_task("debug_task")  # Not recorded
    """)
    
    print("\n" + "="*60)
    print("KEY BENEFITS")
    print("="*60)
    print("""
✅ DEVELOPMENT FRIENDLY:
   - No data recording overhead during development
   - Faster startup and execution
   - No unwanted data files created
   
✅ PRODUCTION READY:
   - Easy to enable comprehensive data collection  
   - All execution data captured automatically
   - User feedback collection included
   
✅ FLEXIBLE:
   - Toggle recording at any time
   - Override per-task as needed
   - Check recording status programmatically
   
✅ BACKWARD COMPATIBLE:
   - Existing code continues to work
   - No breaking changes
   - Optional feature that doesn't interfere
    """)

if __name__ == "__main__":
    demo_toggle_concept()