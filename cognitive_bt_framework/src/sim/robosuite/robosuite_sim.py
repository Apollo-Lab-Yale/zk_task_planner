import argparse
import json
import time
from collections import deque
import threading
from robosuite.utils.observables import Observable

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
            "translucent_robot": True,
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
        # self.env = VisualizationWrapper(self.env)
        # self.env.reset()
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
        # self.last_obs = None

        # Start the simulation thread
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True  # Allow thread to exit when main program exits
        self.simulation_thread.start()
        self.action_fn_from_str = {
            "walk_to_object": self.navigate_to_object,
            "walk": self.navigate_to_object,
            "grab": self.grab_object,
            "turnleft": self.turn_left,
            "turnright": self.turn_right,
            "put": self.place_object,
            "lookup": self.look_up
        }


    def cmd_to_action_floating_base(self, base_vx, base_vy, base_omega, gripper_pos, close_gripper=False):
        action = np.zeros(self.env.action_dim)
        action[-4] = base_vx
        action[-3] = base_vy
        action[-2] = base_omega
        action[:6] = gripper_pos
        action[-1] = int(close_gripper)
        return action

    def navigate_to_object(self, obj):
        pass

    def grab_object(self, obs):
        pass

    def place_object(self, obj):
        pass

    def turn_left(self, rads=np.pi/6):
        action = self.get_action(base_vx=0, base_vy=0, base_omega=self.turn_speed, gripper_pos=0)
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
        obs = self.last_obs
        rgb_frames = []
        depth_frames = []
        detections = []
        object_positions = []
        for idx, camera_name in enumerate(self.config["camera_names"]):
            rgb_frame = obs[camera_name + '_image']
            depth_frame = obs[camera_name + '_depth']
            cv2.imshow(f"camera frame", rgb_frame)
            cv2.waitKey()
            camera_detections = self.object_detection.detect_objects(rgb_frame)
            print(f"CAMERA_DETECTIONS: {[det['name'] for det in camera_detections]}")
            # print(rgb_frame.shape)
            # print(np.count_nonzero(rgb_frame))

            itern+=1
            if itern % 10 == 0:
                cv2.destroyAllWindows()
            # Get camera intrinsics
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
                mask = detection['mask']
                # Get depth values within the mask
                print(depth_frame.shape)
                print(rgb_frame.shape)
                print(mask.shape)
                mask = mask.reshape(
                    depth_frame.shape[0], depth_frame.shape[1]
                )
                masked_depth = depth_frame * mask
                ys, xs = np.where(mask > 0)
                depths = masked_depth[ys, xs]
                # Exclude zero depth values
                valid = depths > 0
                xs = xs[valid]
                ys = ys[valid]
                depths = depths[valid]
                if len(depths) == 0:
                    continue
                # Back-project pixel coordinates to camera coordinates
                x_cam = (xs - cx) * depths / fx
                y_cam = (ys - cy) * depths / fy
                z_cam = depths
                points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
                # Compute centroid in camera frame
                centroid_cam = np.mean(points_cam, axis=0)
                # Transform centroid to world frame
                centroid_world = cam_pose @ centroid_cam + cam_pos
                # Add 3D position to detection
                detection['position'] = centroid_world
                object_positions.append(detection)
            return object_positions


    def simulation_loop(self):
        self.env.reset()
        while True:
            # Pop action from the queue if available
            if self.action_queue:
                act = self.action_queue.popleft()
                print(act)
                action, n_time_steps = act
            else:
                # Use zero action if no action is available
                action = np.zeros(self.env.action_dim)
                n_time_steps = 1
            if n_time_steps < 1:
                n_time_steps = 1
            # Step the environment

            for step in range(n_time_steps):
                print('maybe here?')
                obs, reward, done, info = self.env.step(action)

                # self.env.render()
                self.last_obs = obs
                # Render the environment
                print(self.get_state())
                # Wait for the next control step
                time.sleep(self.dt)

    def add_action(self, action):
        """Add an action to the action queue."""
        self.action_queue.append(action)


    def select_grasp(self, item_pos, arm):
        pass

    def execute_action(self, action):
        pass

if __name__ == "__main__":
    # Instantiate the simulation
    sim = RobosuiteSim()


    # Function to generate and add actions
    def generate_actions(sim):
        for i in range(100):
            # if i < 3:
                # sim.turn_left()
            # # Create a random action
            action = np.random.uniform(-1, 1, size=sim.env.action_dim)
            sim.add_action((action, sim.dt))
            time.sleep(0.05)  # Sleep before adding the next action


    # Start generating actions in the main thread or another thread
    generate_actions(sim)

    # Keep the main thread alive if needed
    input("Press any key to exit.")

    sim.env.close()

