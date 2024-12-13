import argparse
import json
import time
from collections import deque
import threading
from copy import deepcopy
from typing import Tuple, List, Optional, Dict, Any
import logging
import base64
import math

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils import camera_utils
from robosuite.utils import transform_utils as T
from robosuite.utils.observables import Observable
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

import numpy as np
import cv2

from cognitive_bt_framework.src.vision.object_detection.yolo import ObjectDetection

MAX_TIMESTEPS = 200
GRASP_ERROR = 0.0375
WORKING_RADIUS = 0.57

class RobosuiteSimEnv(object):
    def __init__(self, task='BeverageSorting', layout=LayoutType.ONE_WALL_SMALL, style=StyleType.COASTAL,
                 robot='B1Z1Floating', renderer='mjviewer', arm_ctl="", turn_speed=1.0, move_speed=1, use_camera='leg0_robotview',
                 scene_index=-1, width=600, height=600, gridSize=0.25, visibilityDistance=20, 
                 single_room='kitchen', save_video=False, use_find=False):
        """Initialize RobosuiteSim with improved parameter validation."""
        if robot == 'B1Z1':
            raise NotImplementedError("B1Z1 is not yet implemented.")
        else:
            self.get_action = self.cmd_to_action_floating_base
        self.single_room = single_room
        self.scene_index = scene_index
        self.use_find = use_find
        self.save_video = save_video

        self.config = {
            "env_name": task,
            "robots": robot,
            "controller_configs": load_composite_controller_config(robot=robot),
            "layout_ids": layout,
            "camera_depths": True,
            "style_ids": style,
            "translucent_robot": False,
            "camera_names": ["leg0_robotview"],
            "obj_instance_split": 'A'
        }
        # Thread-safe variables
        self._running = False
        self._gripper_closed = False
        self._obs_mutex = threading.Lock()
        self._queue_mutex = threading.Lock()
        self._gripper_mutex = threading.Lock()
        self._action_queue = deque()

        # Configuration
        self.camera_name = use_camera
        self.renderer = renderer
        self.control_freq = 20
        self.dt = 1.0 / self.control_freq
        self.turn_speed = float(turn_speed)
        self.move_speed = float(move_speed)
        
        # Initialize environment
        try:
            self.env = robosuite.make(
                **self.config,
                has_renderer=True,
                has_offscreen_renderer=True,
                render_camera=use_camera,
                ignore_done=True,
                use_camera_obs=True,
                control_freq=self.control_freq,
                renderer=renderer,
            )
            print('made env')
        except Exception as e:
            logging.error(f"Failed to initialize environment: {e}")
            raise

        self.object_detection = ObjectDetection()
        self.object_names = self.get_object_names()
        
        # Initialize state
        self.last_obs = self.env.reset()
        self.init_ee_pos = self.get_gripper_pose()
        
        self.action_fn_from_str = {
            "walk_to_object": self.navigate_to_object,
            "walk": self.navigate_to_object,
            "moveright": self.move_right,
            "grab": self.grab_object,
            "turnleft": self.turn_left,
            "turnright": self.turn_right,
            "put": self.place_object,
            "lookup": self.look_up
        }
    
    def get_object_names(self):
        return self.object_detection.get_classes()

    def __del__(self):
        """Ensure proper cleanup of resources."""
        self.stop()
        if hasattr(self, 'env'):
            self.env.close()

    def get_camera_image(self):
        obs = self.get_last_obs()
        return obs[f"{self.camera_name}_image"]

    def get_gripper_pose(self) -> np.ndarray:
        """Get current gripper pose in robot frame."""
        last_obs = self.get_last_obs()
        ee_pose = last_obs['robot0_eef_pos']
        robot_quat =last_obs['robot0_base_quat']
        robot_pose = last_obs['robot0_base_pos']

        R = T.quat2mat(robot_quat)
        return (ee_pose - robot_pose) @ R

    def start(self) -> None:
        """Start simulation thread with proper synchronization."""
        print('starting sim')
        if self._running:
            return
        self._running = True
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def stop(self) -> None:
        """Stop simulation thread safely."""
        self._running = False
        if hasattr(self, 'simulation_thread'):
            self.simulation_thread.join(timeout=5.0)
    
    def cmd_to_action_floating_base(self, base_vx=0.0, base_vy=0.0, 
                                  base_omega=0.0, gripper_pos=None) -> np.ndarray:
        """Convert commands to robot actions with input validation."""
        if gripper_pos is None:
            gripper_pos = np.zeros(6)
        with self._gripper_mutex:
            gripper = 1.0 if self._gripper_closed else -1.0
        action = np.zeros(self.env.action_dim)
        action[-4:] = [base_vx, base_vy, base_omega, gripper]
        action[:6] = gripper_pos
        return action

    def get_last_obs(self) -> Dict[str, Any]:
        """Thread-safe access to last observation."""
        with self._obs_mutex:
            obs, _, _, _ = self.env.step(np.zeros(self.env.action_dim))
            self.last_obs = obs
            return deepcopy(obs)
    
    def set_last_obs(self, obs):
        with self.obs_mutex:
            self.last_obs = deepcopy(obs)
    
    def get_closest_obj(self, obj: str) -> Tuple[Optional[Dict], Optional[float]]:
        """Find closest object of specified type."""
        state = self.get_state()
        objs = [ob for ob in state['objects'] if ob['name'] == obj]
        
        if not objs:
            return None, None
            
        min_dist = float('inf')
        closest_obj = None
        
        for ob in objs:
            distance = np.linalg.norm(ob['position'])
            if distance < min_dist:
                closest_obj = ob
                min_dist = distance
                
        return closest_obj, min_dist
    
    def navigate_to_object(self, obj: str) -> Tuple[bool, str]:
        """Navigate to object with improved error handling."""
        vx = vy = 0.0
        
        for _ in range(MAX_TIMESTEPS):
            closest_obj, min_dist = self.get_closest_obj(obj)
            if closest_obj is None:
                return False, f"Couldn't find any {obj}."
            pos = closest_obj['position']
            vx = vy = 0.0
            while self._action_queue:
                if abs(pos[0]) <= WORKING_RADIUS and abs(pos[1]) <= WORKING_RADIUS:
                    return True, ""

            if abs(pos[0]) <= WORKING_RADIUS and abs(pos[1]) <= WORKING_RADIUS:
                return True, ""
            # Calculate required velocities
            if abs(pos[0]) > WORKING_RADIUS:
                vx = self.move_speed * (-1 if pos[0] < 0 else 1)
            if abs(pos[1]) > WORKING_RADIUS:
                vy = self.move_speed * (-1 if pos[1] < 0 else 1)
            action = self.cmd_to_action_floating_base(base_vx=vx, base_vy=vy)
            self.add_action(action, 5)
            
           

        return False, f"Navigation timeout after {MAX_TIMESTEPS} steps"

        
    def move_gripper_to_pos(self, pos):
        for i in range(MAX_TIMESTEPS):
            ee_pose = self.get_gripper_pose()
            pos_diff = np.zeros(6)
            pos_diff[:3] = pos[:3] - ee_pose[:3]
            while self._action_queue:
                if abs(pos_diff[0]) < GRASP_ERROR and abs(pos_diff[1]) < GRASP_ERROR and (pos_diff[2] < GRASP_ERROR):
                    return True, ""
                time.sleep(self.dt)
            
            action = self.get_action(base_vx=0, base_vy=0, base_omega=0.0, gripper_pos=pos_diff)
            self.add_action(action, 1)
            
        return False, f"Failed to move gripper to pos {pos} in {MAX_TIMESTEPS} timesteps"

    def close_gripper(self):
        with self._gripper_mutex:
            self._gripper_closed = True
        for step in range(100):
            action = self.cmd_to_action_floating_base(gripper_pos=np.zeros(6))
            self.add_action(action, 1)
            while self._action_queue:
                time.sleep(self.dt)
        return True, f"Failed to close gripper in {MAX_TIMESTEPS} timesteps"

    #TODO: implement collision detection
    def grab_object(self, obj: str) -> Tuple[bool, str]:
        """Grab object with improved error handling and safety checks."""
        closest_obj, _ = self.get_closest_obj(obj)
        if closest_obj is None:
            return False, f"Couldn't find any {obj}."

        obj_pos = np.zeros(6)
        obj_pos[:3] = closest_obj['position']

        try:
            success, msg = self.move_gripper_to_pos(obj_pos)
            success, msg = self.close_gripper()
            if not success:
                return False, msg
            # Lift object
            obj_pos[2] += 0.2
            self.move_gripper_to_pos(obj_pos)
            self.move_gripper_to_pos(self.init_ee_pos)
            return True, ""
            
        except Exception as e:
            self.move_gripper_to_pos(self.init_ee_pos)
            return False, f"Error during grab: {str(e)}"
        

    def place_object(self, obj):
        pass

    def turn_left(self, rads=np.pi/6):
        action = self.get_action(base_vx=0, base_vy=0, base_omega=self.turn_speed, gripper_pos=(0,)*6)
        action_time = math.ceil(rads * self.turn_speed / self.dt)
        self.action_queue.append((action, action_time))

    def turn_right(self, rads=np.pi/6):
        action = self.get_action(base_vx=0, base_vy=0, base_omega=-self.turn_speed, gripper_pos=0)
        action_time = math.ceil(rads * self.turn_speed / self.dt)
        self.action_queue.append((action, action_time))

    def move_right(self, dis=1.0):
        action_time = math.ceil(rads * self.turn_speed / self.dt)

    def look_up(self, rads=np.pi/6):
        pass

    def get_object_positions(self, camera_detections):
        global itern
        object_positions = []
        camera_name = self.camera_name
        sim = self.env.sim
        self.last_obs = self.get_last_obs()
        
        img = self.last_obs[f"{camera_name}_image"]
        img_height, img_width = img.shape[:2]
        
        camera_to_world_transform = camera_utils.get_camera_transform_matrix(sim, camera_name, 640, 640)
        camera_to_world_transform = np.linalg.inv(camera_to_world_transform)
        
        depth_map = self.last_obs[f"{camera_name}_depth"]
        depth_map = camera_utils.get_real_depth_map(sim, depth_map)
        
        for detection in camera_detections:
            mask = detection['mask'].astype(bool)
            if not np.any(mask):
                continue
                
            y_coords, x_coords = np.where(mask)
            x_center = int(np.mean(x_coords))
            y_center = int(np.percentile(y_coords, 75))
            
            pixel_coords = np.array([[y_center, x_center]])
            depth_value = depth_map[y_center, x_center]
            depth_array = np.array([depth_map])
            
            world_coord = camera_utils.transform_from_pixels_to_world(
                pixel_coords, 
                depth_array,
                camera_to_world_transform
            )
            world_coord[0][2] -= 0.03
            robot_quat = self.last_obs['robot0_base_quat']
            R = T.quat2mat(robot_quat)
            detection['position'] = (world_coord[0]  - self.last_obs['robot0_base_pos']) @ R
            detection['position_world'] = world_coord[0]
            object_positions.append(detection)

        return object_positions

    def get_pose_in_gripper_frame(self, obs, object_positions):
        ee_pose = obs['robot0_eef_pos']
        ee_quat = obs['robot0_eef_quat']

        # Convert quaternion to rotation matrix
        # Ensure quaternion is in (x, y, z, w) format
        ee_quat = T.convert_quat(ee_quat, to='xyzw')
        R_world_ee = T.quat2mat(ee_quat)  # Rotation matrix from ee to world frame
        R_ee_world = R_world_ee.T  # Rotation matrix from world to ee frame
        
        # Translate object positions to the ee frame
        for detection in object_positions:
            P_world = detection['position']  # Object position in world frame
            # Compute object position relative to the ee frame
            P_ee = R_ee_world @ (P_world - ee_pose)
            detection['position_in_ee'] = P_ee  # Add position in ee frame to detection
        return object_positions

    def get_state(self):
        """Get current state with object positions and metadata."""
        obs = self.get_last_obs()
        
        # Get detections and object positions
        bgr_frame = obs["leg0_robotview_image"]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_RGB2BGR)
        camera_detections = self.object_detection.detect_objects(rgb_frame)
        object_positions = self.get_object_positions(camera_detections)
        
        # Build state dict
        state = {
            'rooms': [{'roomType': 'kitchen', 'name': 'kitchen'}],
            'objects': [],
            'predicates': [],
            'robot_state': {
                'name': 'robot',
                'position': obs["robot0_base_pos"],
                'rotation': obs["robot0_base_quat"],
                'isStanding': True
            },
            'memory_dict': {}
        }
        
        # Format object data
        for obj in object_positions:
            obj_data = {
                'name': obj['name'],
                'position': obj["position"],
                'rotation': np.array([0.0,0.0,0.0]),
                'visible': True,
                'isInteractable': True,
                'pickupable': True,
                'distance': float(np.linalg.norm(obj['position']))
            }
            state['objects'].append(obj_data)
            
            # Add common predicates
            state['predicates'].append(f"GRABBABLE {obj['name']}")
            
        return state

    def simulation_loop(self) -> None:
        """Main simulation loop with improved timing and error handling."""
        while self._running:
            start_time = time.time()
            
            try:
                action = self.cmd_to_action_floating_base()
                n_time_steps = 1

                with self._queue_mutex:
                    if self._action_queue:
                        action, n_time_steps = self._action_queue.popleft()

                for _ in range(n_time_steps):
                    with self._obs_mutex:
                        self.last_obs, _, _, _ = self.env.step(action)
                        
                    exec_time = time.time() - start_time
                    if exec_time < self.dt:
                        time.sleep(self.dt - exec_time)
                        
            except Exception as e:
                logging.error(f"Error in simulation loop: {e}")
                time.sleep(self.dt)


    def add_action(self, action: np.ndarray, n_timesteps: int) -> None:
        """Thread-safe action queue management."""
        with self._queue_mutex:
            self._action_queue.append((action, n_timesteps))

    def select_grasp(self, item_pos, arm):
        pass

    def execute_action(self, action):
        pass


    def get_context(self, num_images=5):
        """Gather multi-view context by rotating and capturing images/states."""
        
        rotation = 360 / num_images # Degrees to rotate between views
        pause_time = 0.1
        images = []
        states = []
        
        # Get initial view
        obs = self.get_last_obs()
        img = obs[f"{self.camera_name}_image"]
        _, frame = cv2.imencode('.png', img)
        images.append(base64.b64encode(frame.tobytes()).decode('utf-8'))
        states.append(self.get_state())

        # Rotate and collect views
        for i in range(num_images + 1):
            # Turn left by rotation amount
            action = self.cmd_to_action_floating_base(base_omega=self.turn_speed)
            action_time = math.ceil(math.radians(rotation) / self.turn_speed / self.dt)
            self.add_action(action, action_time)
            
            while self._action_queue:
                time.sleep(pause_time)
                
            # Capture image and state
            obs = self.get_last_obs()
            img = obs[f"{self.camera_name}_image"]
            _, frame = cv2.imencode('.png', img)
            images.append(base64.b64encode(frame.tobytes()).decode('utf-8'))
            states.append(self.get_state())

        return images, states
    
    def translate_action_for_sim(self, action, state):
        """Translate actions between different sim formats."""
        return [action]

    def check_satisfied(self, sub_goal, memory):
        """Check if a sub-goal is satisfied."""
        if sub_goal is None:
            return True, ""
            
        parts = sub_goal.split()
        if len(parts) <= 1:
            return False, f"Invalid subgoal format: {sub_goal}"
            
        value = int(parts[-1])
        pred = parts[0] 
        params = parts[1:-1]
        target = params[0]
        recipient = params[1] if len(params) > 1 else None
        
        return self.check_condition(pred, target, recipient, memory, value)

    def check_condition(self, cond, target, recipient, memory, value=1):
        """Check conditions for objects and state."""
        state = self.get_state()
        
        if "visible" in cond.lower():
            visible = any(obj['name'].lower() == target.lower() for obj in state['objects'])
            return visible, "" if visible else f"{target} not visible"
            
        if "isclose" in cond.lower():
            close_obj = next((obj for obj in state['objects'] 
                            if obj['name'].lower() == target.lower() 
                            and obj['distance'] < WORKING_RADIUS), None)
            return bool(close_obj), f"{target} {'is' if close_obj else 'is not'} close"

        return False, f"Condition {cond} not implemented"

    def check_goal(self, goal):
        """Check if overall goal is satisfied."""
        return False

    def validate_goal(self, goal):
        """Validate goal format and parameters.""" 
        if not goal.get('conditions'):
            return False
        condition = goal['conditions'][0].split()
        if len(condition) < 3:
            return False
        return True

    def environment_graph(self):
        """Get environment graph representation."""
        graph = {}
        state = self.get_state()
        graph['nodes'] = state['objects']
        for node in graph['nodes']:
            node['class_name'] = node['name']
            node['id'] = node['name']
        return True, graph

    def execute_actions(self, actions, memory):
        """Execute a sequence of actions."""
        for action in actions:
            act, target = action.split(' ', 1)
            if target.lower() not in [obj['name'].lower() for obj in self.get_state()['objects']] and\
                action != 'search':
                return False, f"Object {target} not found"
            
            fn = self.action_fn_from_str.get(act)
            if not fn:
                return False, f"Action {act} not implemented"
                
            success, msg = fn(target)
            if not success:
                return False, msg
                
        return True, "Actions executed successfully"
    
def main():
    global itern
    # Instantiate the simulation
    sim = RobosuiteSimEnv()
    sim.start()
    print('started sim')
    sim.env.render()
    print('env_rendered')
    print(sim.get_last_obs().keys())
    print(sim.navigate_to_object("bottle"))
    print(sim.grab_object("bottle"))
    input('press key to exit')
    sim.stop()

if __name__ == "__main__":
   main()

