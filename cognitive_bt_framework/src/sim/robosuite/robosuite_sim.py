import argparse
import json
import time
from collections import deque
import threading
from copy import deepcopy

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils import camera_utils
from robosuite.utils import transform_utils as T
from robosuite.utils.observables import Observable

from robocasa.models.scenes.scene_registry import LayoutType, StyleType



import h5py
import imageio
import mujoco
import numpy as np
import math
from robocasa.scripts.collect_demos import collect_human_trajectory
import cv2
import memory_profiler
import math

from cognitive_bt_framework.src.vision.object_detection.yolo import ObjectDetection
itern = 0

MAX_TIMESTEPS = 50
GRASP_ERROR = 0.025
WORKING_RADIUS = 0.75

class RobosuiteSim(object):
    def __init__(self, task='PrewashFoodAssembly', layout=LayoutType.ONE_WALL_SMALL, style=StyleType.COASTAL,
                 robot='B1Z1Floating', renderer='mjviewer', arm_ctl="", turn_speed=1.0, move_speed=10.0, use_camera='leg0_robotview'):
        self.config = {
            "env_name": task,
            "robots": robot,
            "controller_configs": load_composite_controller_config(robot=robot),
            "layout_ids": layout,
            "camera_depths": True,
            "style_ids": style,
            "translucent_robot": True,
            "camera_names": ["leg0_robotview"],
            "obj_instance_split": 'A'
        }
        self.camera_name = use_camera
        self.renderer = renderer
        self.env = robosuite.make(
            **self.config,
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera="leg0_robotview",
            ignore_done=True,
            use_camera_obs=True,
            control_freq=20,
            renderer=renderer,
        )
        self.running = False
        self.obs_mutex = threading.Lock()
        # self.env = VisualizationWrapper(self.env)
        if robot == 'B1Z1':
            raise NotImplementedError("B1Z1 is not yet implemented.")
        else:
            self.get_action = self.cmd_to_action_floating_base

        # Initialize the action queue
        self.action_queue = deque()
        self.control_freq = 20  # Control frequency in Hz
        self.dt = 1.0 / self.control_freq
        self.turn_speed = turn_speed # 1.0 rad / sec by default
        self.move_speed = move_speed

        self.object_detection = ObjectDetection()

        self.last_obs = self.env.reset()

        # Start the simulation thread
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

    def start(self):
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True  # Allow thread to exit when main program exits
        self.running = True
        self.simulation_thread.start()
    
    def stop(self):
        self.running = False
        self.simulation_thread.join()
        # self.env.sim.end()
    
    def cmd_to_action_floating_base(self, base_vx=0.0, base_vy=0.0, base_omega=0.0, gripper_pos=np.zeros(6), close_gripper=False):
        action = np.zeros(self.env.action_dim)
        action[-4] = base_vx
        action[-3] = base_vy
        action[-2] = base_omega
        action[:6] = gripper_pos
        action[-1] = int(close_gripper)
        return action

    def get_last_obs(self):
        with self.obs_mutex:
            obs, _, _, _ = self.env.step(np.zeros(self.env.action_dim))
            self.last_obs = obs
            return obs
    
    def set_last_obs(self, obs):
        with self.obs_mutex:
            self.last_obs = deepcopy(obs)
    
    def get_closest_obj(self, obj):
        state = self.get_state()
        objs = [ob for ob in state if ob['name'] == obj]
        if len(objs) == 0:
            return None, None
        min_dist = np.inf
        closest_obj = None
        for ob in objs:
            distance = np.linalg.norm(ob['position'])
            print(ob['position'])
            if distance < min_dist:
                closest_obj = ob
                min_dist = distance
        return closest_obj, min_dist
    
    def navigate_to_object(self, obj):
        for i in range(MAX_TIMESTEPS):
            closest_obj, min_dist = self.get_closest_obj(obj)
            if closest_obj is None:
                return False, f"Couldn't find any {obj}."
            vx, vy = 0.0, 0.0
            print(closest_obj['position'], min_dist)
            if math.fabs(closest_obj['position'][0]) > WORKING_RADIUS:
                vx = self.move_speed
                if closest_obj['position'][0] < 0: vx = -vx

            if math.fabs(closest_obj['position'][1]) > WORKING_RADIUS:
                vy = self.move_speed
                if closest_obj['position'][1] < 0: vy = -vy

            if vx < self.move_speed and vy < self.move_speed:
                return True, ""
            action = self.get_action(base_vx=vx, base_vy = vy)
            self.add_action(action, 5)
        return False, f"Failed to navigate to {obj} in {MAX_TIMESTEPS} timesteps"
        

    def grab_object(self, obj):
        obj_name = obj
        for i in range(MAX_TIMESTEPS):
            closest_obj, min_dist = self.get_closest_obj( obj)
            if closest_obj is None:
                return False, f"Couldn't find any {obj}."
            if min_dist < GRASP_ERROR:
                action = self.get_action(close_gripper=True)
                self.action_queue.append((action, 1))
                return True, ""
            gripper_pos = np.zeros(6)
            gripper_pos[:3] = closest_obj['position']
            action = self.get_action(base_vx=0, base_vy=0, base_omega=0.0, gripper_pos=gripper_pos)
            self.action_queue.append((action, 3))
            time.sleep(self.dt)
        return False, f"Failed to grab {obj_name} in {MAX_TIMESTEPS} timesteps."
        
        
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
        folder_path = '/home/liam/dev/zk_task_planner/cognitive_bt_framework/src/sim/robosuite/test/'
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
            y_center = int(np.mean(y_coords))
            
            pixel_coords = np.array([[y_center, x_center]])
            depth_value = depth_map[y_center, x_center]
            depth_array = np.array([depth_map])
            
            world_coord = camera_utils.transform_from_pixels_to_world(
                pixel_coords, 
                depth_array,
                camera_to_world_transform
            )
            robot_quat = self.last_obs['robot0_base_quat']
            R = T.quat2mat(robot_quat)
            detection['position'] = (world_coord[0]  - self.last_obs['robot0_base_pos']) @ R
            print(detection['position'], detection['name'])
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
        global itern
        obs = self.get_last_obs()
        
        rgb_frames = []
        depth_frames = []
        detections = []
        object_positions = []
            
        bgr_frame = obs["leg0_robotview_image"]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_RGB2BGR)
        depth_frame = obs["leg0_robotview_depth"]
        camera_detections = self.object_detection.detect_objects(rgb_frame)
        # print([det['name'] for det in camera_detections])
        object_positions = self.get_object_positions(camera_detections)
        object_positions = self.get_pose_in_gripper_frame(obs, object_positions)
        return object_positions

            

    def simulation_loop(self):
        global itern
        count = 0
        max_fr = 20
        while self.running:
            start = time.time()
            # Pop action from the queue if available
            action = np.zeros(self.env.action_dim)
            if self.action_queue:
                act = self.action_queue.popleft()
                print(act)
                action, n_time_steps = act
            else:
                # # Use zero action if no action is available
                action = np.zeros(self.env.action_dim)
                n_time_steps = 1
                # continue
            if n_time_steps < 1:
                n_time_steps = 1
            # Step the environment

            for step in range(n_time_steps):
                with self.obs_mutex:
                    obs, reward, done, info = self.env.step(action)
                # Wait for the next control step
                # if max_fr is not None:
                #     elapsed = time.time() - start
                #     diff = 1 / max_fr - elapsed
                #     if diff > 0:
                #         time.sleep(diff)

    def add_action(self, action, n_timesteps):
        """Add an action to the action queue."""
        self.action_queue.append((action, n_timesteps))


    def select_grasp(self, item_pos, arm):
        pass

    def execute_action(self, action):
        pass
    
def main():
    global itern
    # Instantiate the simulation
    sim = RobosuiteSim()
    sim.start()
    sim.env.render()
    print(sim.get_last_obs().keys())
    print(sim.navigate_to_object("bowl"))
    print(sim.grab_object("bowl"))
    input('press key to exit')
    sim.stop()

if __name__ == "__main__":
   main()

