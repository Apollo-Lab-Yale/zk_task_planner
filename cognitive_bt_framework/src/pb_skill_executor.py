#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import time
import sys
import logging
import os
import pybullet as p
import pybullet_data
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for threading safety

from cognitive_bt_framework.src.collect_skill_data import DirectSkillExecutor
from cognitive_bt_framework.src.pybullet_visualizer import PyBulletVisualizer, XARM_7_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pybullet_integration')

class ThreadSafeDirectSkillExecutor(DirectSkillExecutor):
    """Thread-safe version of DirectSkillExecutor with patched visualization methods"""
    
    def __init__(self, **kwargs):
        """Initialize with all the same parameters as DirectSkillExecutor"""
        # Store original debug window setting
        self.show_debug_windows_original = kwargs.get('show_debug_windows', True)
        
        # Disable debug windows in the original class - we'll handle visualization ourselves
        kwargs['show_debug_windows'] = False
        
        # Initialize the parent class
        super().__init__(**kwargs)
        
        # Override the visualization method to our thread-safe version
        self._original_update_visualization = getattr(self, 'update_visualization', None)
        self.update_visualization = self._thread_safe_update_visualization
        
        # Threading safety
        self.visualization_lock = threading.RLock()
        self.main_thread_id = threading.current_thread().ident
        
        # Storage for visualizations that need to be updated on the main thread
        self.pending_visualizations = []
        
        # Flag to track if we're in the main thread
        self.is_main_thread = lambda: threading.current_thread().ident == self.main_thread_id
        
        # Create visualization windows in the main thread if requested
        if self.show_debug_windows_original:
            self.create_visualization_windows()
        
        logger.info("ThreadSafeDirectSkillExecutor initialized")
    
    def create_visualization_windows(self):
        """Create visualization windows in the main thread"""
        if not self.is_main_thread():
            logger.warning("Attempted to create visualization windows from non-main thread")
            return
        
        # Create windows for standard visualizations
        window_names = [
            "RGB Image", 
            "Depth Image", 
            "Segmentation", 
            "Detection",
            "Surface Points",
            "Grasp Points"
        ]
        
        for window_name in window_names:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
        
        logger.info(f"Created {len(window_names)} visualization windows")
    
    def _thread_safe_update_visualization(self, image, window_name):
        """Thread-safe version of visualization update"""
        # Skip if visualization is disabled
        if not self.show_debug_windows_original:
            return True
            
        try:
            # If we're in the main thread, update directly
            if self.is_main_thread():
                cv2.imshow(window_name, image)
                cv2.waitKey(1)
                return True
            
            # Otherwise, store for later update
            with self.visualization_lock:
                # Store a copy of the image
                self.pending_visualizations.append((window_name, image.copy()))
            
            return True
        except Exception as e:
            logger.error(f"Error updating visualization: {str(e)}")
            return False
    
    def process_pending_visualizations(self):
        """Process any pending visualizations on the main thread"""
        # Skip if visualization is disabled
        if not self.show_debug_windows_original:
            return
            
        if not self.is_main_thread():
            logger.warning("Attempted to process visualizations from non-main thread")
            return
            
        with self.visualization_lock:
            for window_name, image in self.pending_visualizations:
                try:
                    cv2.imshow(window_name, image)
                    cv2.waitKey(1)
                except Exception as e:
                    logger.error(f"Error showing visualization: {str(e)}")
            
            # Clear processed visualizations
            self.pending_visualizations = []


class ImprovedPyBulletIntegration:
    """Improved PyBullet integration with proper threading model"""
    
    def __init__(self, robot_ip="192.168.1.224", show_pybullet=True, show_camera_views=True):
        """Initialize the integration"""
        self.robot_ip = robot_ip
        self.show_pybullet = show_pybullet
        self.show_camera_views = show_camera_views
        self.is_running = False
        
        # Initialize PyBullet first (must be in main thread)
        if self.show_pybullet:
            self.initialize_pybullet()
        
        # Initialize the thread-safe executor
        self.executor = ThreadSafeDirectSkillExecutor(
            robot_ip=self.robot_ip,
            show_debug_windows=self.show_camera_views  # Enable visualization if requested
        )
        
        # Robot control thread
        self.robot_control_thread = None
        
        # Storage for visualization data
        self.latest_joint_positions = None
        self.lock = threading.Lock()
        
        logger.info("ImprovedPyBulletIntegration initialized")
    
    def initialize_pybullet(self):
        """Initialize PyBullet in the main thread"""
        try:
            logger.info(f"Initializing PyBullet with URDF: {XARM_7_PATH}")
            
            # Connect to PyBullet GUI
            self.physics_client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
            p.setGravity(0, 0, -9.8)
            
            # Create ground plane
            self.plane_id = p.loadURDF("plane.urdf")
            
            # Load robot URDF
            self.robot_id = p.loadURDF(
                XARM_7_PATH, 
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
            )
            
            # Get number of joints and joint information
            self.num_joints = p.getNumJoints(self.robot_id)
            logger.info(f"Robot loaded with {self.num_joints} joints")
            
            # Store joint info
            self.joint_indices = []
            for i in range(self.num_joints):
                info = p.getJointInfo(self.robot_id, i)
                joint_name = info[1].decode('utf-8')
                link_name = info[12].decode('utf-8')
                logger.info(f"Joint {i}: {joint_name}, Link: {link_name}")
                
                # Store movable joints
                if info[2] != p.JOINT_FIXED:
                    self.joint_indices.append(i)
            
            # Create a simple visualization for the world frame
            p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], lineWidth=2)  # X-axis (red)
            p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], lineWidth=2)  # Y-axis (green)
            p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], lineWidth=2)  # Z-axis (blue)
            
            logger.info("PyBullet visualization initialized successfully")
            return True
        except Exception as e:
            logger.error(f"PyBullet initialization error: {str(e)}", exc_info=True)
            self.show_pybullet = False
            return False
    
    def update_robot_visualization(self):
        """Update the robot visualization in PyBullet"""
        try:
            # Get latest joint positions from the executor
            joint_positions = self.get_joint_positions()
            
            # Update the robot configuration
            if joint_positions is not None:
                with self.lock:
                    self.latest_joint_positions = joint_positions
                
                for i, pos in enumerate(joint_positions):
                    if i < len(self.joint_indices):
                        p.resetJointState(self.robot_id, self.joint_indices[i], pos)
                
                # logger.info(f"Joint positions updated: {joint_positions}")
            
            # Process any pending visualizations from the executor
            self.executor.process_pending_visualizations()
            
            return True
        except Exception as e:
            logger.error(f"Error updating robot visualization: {str(e)}")
            return False
    
    def get_joint_positions(self):
        """Get current joint positions from the executor"""
        try:
            # Try to get joint positions from the motion planner
            if hasattr(self.executor, 'motion_planner') and self.executor.motion_planner is not None:
                if hasattr(self.executor.motion_planner, 'current_joints'):
                    joint_positions = self.executor.motion_planner.current_joints
                    if joint_positions is not None and len(joint_positions) > 0:
                        return joint_positions.copy()
            
            # Try alternative methods
            if hasattr(self.executor, 'motion_planner') and hasattr(self.executor.motion_planner, 'get_robot_joint_state'):
                joint_state = self.executor.motion_planner.get_robot_joint_state()
                if joint_state is not None:
                    return joint_state
            
            return None
        except Exception as e:
            logger.error(f"Error getting joint positions: {str(e)}")
            return None
    
    def run_robot_control(self):
        """Execute robot control sequence"""
        try:
            # Check robot status
            status = self.executor.get_robot_status()
            if status:
                logger.info(f"Robot status: Mode={status['mode']}, State={status['state']}")
            
            # Move to home position
            logger.info("Moving to home position...")
            self.executor.move_to_home()
            
            # Execute a skill
            logger.info("Executing pickup skill...")
            success, message = self.executor.execute_skill("pickup", "marker")
            
            if success:
                logger.info(f"Skill execution successful: {message}")
            else:
                logger.error(f"Skill execution failed: {message}")
                
        except Exception as e:
            logger.error(f"Error in robot control: {str(e)}", exc_info=True)
        finally:
            logger.info("Robot control completed")
    
    def start_robot_control(self):
        """Start the robot control in a background thread"""
        if self.robot_control_thread is not None and self.robot_control_thread.is_alive():
            logger.warning("Robot control thread already running")
            return
        
        self.robot_control_thread = threading.Thread(target=self.run_robot_control, daemon=True)
        self.robot_control_thread.start()
        logger.info("Robot control thread started")
    
    def run_main_loop(self):
        """Run the main loop for PyBullet visualization"""
        self.is_running = True
        logger.info("Starting main visualization loop")
        
        try:
            # Main loop - this must run in the main thread for PyBullet to work correctly
            while self.is_running:
                # Update robot visualization
                self.update_robot_visualization()
                
                # Process PyBullet GUI events
                p.stepSimulation()
                
                # Sleep briefly to avoid consuming too much CPU
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Main loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}", exc_info=True)
        finally:
            logger.info("Main visualization loop ended")
    
    def run(self):
        """Run the complete system"""
        try:
            # Start robot control thread
            self.start_robot_control()
            
            # Run main loop for visualization (in main thread)
            self.run_main_loop()
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown all components"""
        logger.info("Shutting down...")
        
        # Stop main loop
        self.is_running = False
        
        # Shutdown executor
        if hasattr(self, 'executor'):
            try:
                self.executor.shutdown()
                logger.info("Executor shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down executor: {str(e)}")
        
        # Disconnect from PyBullet
        if self.show_pybullet and hasattr(self, 'physics_client'):
            try:
                p.disconnect(self.physics_client)
                logger.info("PyBullet disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting from PyBullet: {str(e)}")
        
        # Close all OpenCV windows
        try:
            cv2.destroyAllWindows()
            logger.info("Closed all visualization windows")
        except Exception as e:
            logger.error(f"Error closing visualization windows: {str(e)}")
        
        logger.info("Shutdown complete")


def main():
    """Main entry point"""
    try:
        # Create and run the improved integration
        integration = ImprovedPyBulletIntegration(
            robot_ip="192.168.1.224",  # Replace with your robot's IP
            show_pybullet=True,
            show_camera_views=True     # Set to True to enable camera visualization
        )
        
        # Run the complete system (this will block until completion)
        integration.run()
        
    except Exception as e:
        logger.error(f"Main function error: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()