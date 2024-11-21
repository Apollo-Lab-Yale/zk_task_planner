import argparse
import json
import time
from collections import deque
import threading
from robosuite.utils.observables import Observable
from copy import deepcopy
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import h5py
import imageio
import mujoco
import numpy as np
import math
from robocasa.scripts.collect_demos import collect_human_trajectory
import cv2
import memory_profiler

from cognitive_bt_framework.src.vision.object_detection.yolo import ObjectDetection
itern = 0

class RobosuiteSim(object):
    def __init__(self, task='PnPCounterToCab', layout="0", style="5",
                 robot='B1Z1Floating', renderer='mjviewer', arm_ctl="", turn_speed=1.0):
        self.config = {
            "env_name": task,
            "robots": robot,
            "controller_configs": load_composite_controller_config(robot=robot),
            "layout_ids": layout,
            "camera_depths": True,
            "style_ids": style,
            "translucent_robot": False,
            "camera_names": ["leg0_robotview"]
        }
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
        self.object_detection = ObjectDetection()
        self.last_obs = self.env.reset()

        # Start the simulation thread
        self.action_fn_from_str = {
            "walk_to_object": self.navigate_to_object,
            "walk": self.navigate_to_object,
            "grab": self.grab_object,
            "turnleft": self.turn_left,
            "turnright": self.turn_right,
            "put": self.place_object,
            "lookup": self.look_up
        }

    def start(self):
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True  # Allow thread to exit when main program exits
        self.simulation_thread.start()
    
    def cmd_to_action_floating_base(self, base_vx, base_vy, base_omega, gripper_pos, close_gripper=False):
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
            return obs
    
    def set_last_obs(self, obs):
        with self.obs_mutex:
            self.last_obs = deepcopy(obs)
    
    def navigate_to_object(self, obj):
        pass

    def grab_object(self, obj):
        state = self.get_state()
        pass

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

    def look_up(self, rads=np.pi/6):
        pass


    def get_state(self):
        global itern
        obs = self.get_last_obs()
        
        rgb_frames = []
        depth_frames = []
        detections = []
        object_positions = []
        for idx, camera_name in enumerate(self.config["camera_names"]):
            
            bgr_frame = obs["leg0_robotview_image"]
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_RGB2BGR)
            depth_frame = obs["leg0_robotview_depth"]
            camera_detections = self.object_detection.detect_objects(rgb_frame)
            print(f"CAMERA_DETECTIONS: {[(det['name'], det['conf']) for det in camera_detections]}")
            camera_id = self.env.sim.model.camera_name2id(camera_name)
            fovy = self.env.sim.model.cam_fovy[camera_id]
            img_height, img_width = rgb_frame.shape[:2]
            f = 0.5 * img_height / np.tan(fovy * np.pi / 360)
            fx = f
            fy = f
            cx = img_width / 2
            cy = img_height / 2

            # Get camera extrinsics
            cam_pose = self.env.sim.data.get_camera_xmat(camera_name)  # rotation matrix (3x3)
            cam_pos = self.env.sim.data.get_camera_xpos(camera_name)  # position (3,)
            for detection in camera_detections:
                bbox = detection['bbox']
                # Get bounding box coordinates
                x_min, y_min, x_max, y_max = bbox
                # Compute center of the bounding box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                # Round to integer pixel coordinates
                x_center = int(round(x_center))
                y_center = int(round(y_center))
                # Ensure coordinates are within image bounds
                x_center = min(max(0, x_center), img_width - 1)
                y_center = min(max(0, y_center), img_height - 1)
                # Get depth value at the center pixel
                depth = depth_frame[y_center, x_center]
                if depth <= 0:
                    continue  # Skip if depth is invalid
                # Back-project pixel to camera coordinates
                x_cam = (x_center - cx) * depth / fx
                y_cam = (y_center - cy) * depth / fy
                z_cam = depth
                centroid_cam = np.array([x_cam, y_cam, z_cam])
                # Transform centroid to world frame
                centroid_world = cam_pose @ centroid_cam + cam_pos
                # Add 3D position to detection
                detection['position'] = centroid_world
                object_positions.append(detection)
        return object_positions

    def simulation_loop(self):
        global itern
        count = 0
        max_fr = 20
        while True:
            start = time.time()
            # Pop action from the queue if available
            action = np.zeros(self.env.action_dim)
            if self.action_queue:
                act = self.action_queue.popleft()
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
                rgb_frame = obs["leg0_robotview_image"]
                
                self.env.render()
                self.set_last_obs(obs)
                # Wait for the next control step
                if max_fr is not None:
                    elapsed = time.time() - start
                    diff = 1 / max_fr - elapsed
                    if diff > 0:
                        time.sleep(diff)

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
    last_img = None
    for i in range(50):
        sim.add_action(np.random.uniform(-1,1, sim.env.action_dim), 1)
        obs = sim.get_last_obs()
        sim.get_state()
        # rgb_frame = obs["leg0_robotview_image"]
        # cv2.imwrite(f'/home/liam/dev/zk_task_planner/cognitive_bt_framework/src/sim/robosuite/test/img{itern}.png', rgb_frame)
        # print('wrote ' + f'/home/liam/dev/zk_task_planner/cognitive_bt_framework/src/sim/robosuite/test/img{itern}.png')
        # itern+=1
        # if last_img is not None:
        #     print(not np.any(cv2.subtract(rgb_frame, last_img)))
    
    # # sim.simulation_loop()
    # for i in range(50):
    #     sim.add_action(np.random.uniform(-1,1, sim.env.action_dim), 1)
    #     obs = sim.get_last_obs()
        
    #     time.sleep(0.1)
    #     if last_img is not None:
    #         print(not np.any(cv2.subtract(rgb_frame, last_img)))
    #     last_img = rgb_frame
    # Start generating actions in the main thread or another thread

    # Keep the main thread alive if needed
    input("Press any key to exit.")

    sim.env.close()

if __name__ == "__main__":
   main()

