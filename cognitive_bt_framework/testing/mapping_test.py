from dataclasses import dataclass
import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple
import copy

from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv
from cognitive_bt_framework.src.vision.sam.fast_sam import FastSAMMaskGenerator, FastSAMConfig
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI


@dataclass
class ObjectState:
    name: str
    predicates: Dict
    region_id: str
    caption: str
    position: np.ndarray  # 3D position
    points: np.ndarray    # Point cloud
    confidence: float     # Detection confidence

class EnvironmentStateTracker:
    def __init__(self):
        self.objects: Dict[str, ObjectState] = {}
        self.environment_map = o3d.geometry.PointCloud()
        self.camera_poses = []
        
    async def update_state(self, 
                          rgbd_image: np.ndarray,
                          depth: np.ndarray,
                          camera_pose: np.ndarray,
                          mask_gen,
                          llm_interface) -> List[ObjectState]:
        # Get new detections
        object_states, masks, metadata = await llm_interface.get_object_states(
            rgbd_image, mask_gen
        )
        
        # Convert depth and masks to point clouds
        new_points = self._depth_to_pointcloud(depth, camera_pose)
        self.camera_poses.append(camera_pose)
        
        # Update existing objects and add new ones
        updated_objects = []
        for state_info in object_states:
            region_id = state_info['region_id']
            mask = state_info.get('mask')
            if mask is None:
                continue
                
            # Get object points from mask
            object_points = new_points[mask]
            position = np.mean(object_points, axis=0)
            
            # Update existing object or create new one
            if region_id in self.objects:
                self._update_object(region_id, state_info, position, object_points)
            else:
                self._add_new_object(state_info, position, object_points)
            
            updated_objects.append(self.objects[region_id])
        
        # Update environment map
        self._update_environment_map(new_points)
        
        return updated_objects

    def _depth_to_pointcloud(self, 
                           depth: np.ndarray, 
                           camera_pose: np.ndarray) -> np.ndarray:
        """Convert depth image to point cloud using camera parameters"""
        # Implementation depends on your camera parameters
        # This is a simplified version
        points = o3d.geometry.PointCloud.create_from_depth_image(
            depth,
            o3d.camera.PinholeCameraIntrinsic(
                width=depth.shape[1],
                height=depth.shape[0],
                fx=525.0,  # Replace with actual camera parameters
                fy=525.0,
                cx=depth.shape[1]/2,
                cy=depth.shape[0]/2
            )
        )
        points.transform(camera_pose)
        return np.asarray(points.points)

    def _update_object(self, 
                      region_id: str, 
                      state_info: Dict, 
                      position: np.ndarray,
                      points: np.ndarray):
        obj = self.objects[region_id]
        # Update state information
        obj.predicates = state_info['predicates']
        obj.caption = state_info['caption']
        # Update position using Kalman filter or moving average
        obj.position = 0.7 * obj.position + 0.3 * position
        # Update point cloud
        obj.points = np.vstack([obj.points, points])
        # Optional: downsample points to manage memory
        if len(obj.points) > 1000:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj.points)
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            obj.points = np.asarray(pcd.points)

    def _add_new_object(self, 
                       state_info: Dict, 
                       position: np.ndarray,
                       points: np.ndarray):
        self.objects[state_info['region_id']] = ObjectState(
            name=state_info['name'],
            predicates=state_info['predicates'],
            region_id=state_info['region_id'],
            caption=state_info['caption'],
            position=position,
            points=points,
            confidence=1.0
        )

    def _update_environment_map(self, new_points: np.ndarray):
        # Add new points to environment map
        current_map = np.asarray(self.environment_map.points)
        if len(current_map) == 0:
            self.environment_map.points = o3d.utility.Vector3dVector(new_points)
        else:
            combined = np.vstack([current_map, new_points])
            self.environment_map.points = o3d.utility.Vector3dVector(combined)
        
        # Downsample to manage memory
        self.environment_map = self.environment_map.voxel_down_sample(voxel_size=0.02)

    def get_environment_map(self) -> o3d.geometry.PointCloud:
        return copy.deepcopy(self.environment_map)

    def get_object_state(self, region_id: str) -> Optional[ObjectState]:
        return self.objects.get(region_id)
    

def main():
    tracker = EnvironmentStateTracker()
    sim = RobosuiteSimEnv()
    mask_gen = FastSAMMaskGenerator(FastSAMConfig())
    llm_interface = LLMInterfaceOpenAI()
    while True:
        # Get new RGBD frame
        rgbd_image, depth = sim.get_camera_frames()
        camera_pose = sim.get_camera_pose()

            # Update state
        updated_objects = tracker.update_state(
            rgbd_image, depth, camera_pose, mask_gen, llm_interface
        )
        
        # Get current environment map
        current_map = tracker.get_environment_map()

if __name__ == '__main__':
    main()