import argparse
import json
import time
from collections import deque
import threading

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import h5py
import imageio
import mujoco
import numpy as np
from robocasa.scripts.collect_demos import collect_human_trajectory


class RobosuiteSim(object):
    def __init__(self, task='PnPCounterToCab', layout="0", style="5",
                 robot='B1Z1Floating', renderer='mjviewer', arm_ctl=""):
        self.config = {
            "env_name": task,
            "robots": robot,
            "controller_configs": load_composite_controller_config(robot=robot),
            "layout_ids": layout,
            "style_ids": style,
            "translucent_robot": True,
        }
        self.renderer = renderer
        self.env = robosuite.make(
            **self.config,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera="birdview",
            ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
            renderer=renderer,
        )
        if robot == 'B1Z1':
            raise NotImplementedError("B1Z1 is not yet implemented.")
        else:
            self.get_action = self.cmd_to_action_floating_base
        # Initialize the action queue
        self.action_queue = deque()
        self.control_freq = 20  # Control frequency in Hz
        self.dt = 1.0 / self.control_freq

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
        }

    def cmd_to_action_floating_base(self, base_vx, base_vy, base_omega, gripper_pos):
        action = np.zeros(self.env.action_dim)
        action[-4] = base_vx
        action[-3] = base_vy
        action[-2] = base_omega



    def navigate_to_object(self, obj):
        pass

    def grab_object(self, obj):
        pass

    def place_object(self, obj):
        pass

    def turn_left(self, obj):
        # action =
        pass

    def turn_right(self, obj):
        pass

    def simulation_loop(self):
        while True:
            # Pop action from the queue if available
            if self.action_queue:
                action, n_time_steps = self.action_queue.popleft()
            else:
                # Use zero action if no action is available
                action = np.zeros(self.env.action_dim)
                n_time_steps = 1
            if n_time_steps < 1:
                n_time_steps = 1
            # Step the environment
            for step in range(n_time_steps):
                obs, reward, done, info = self.env.step(action)

                # Render the environment
                self.env.render()
                # Wait for the next control step
                time.sleep(self.dt)

    def add_action(self, action):
        """Add an action to the action queue."""
        self.action_queue.append(action)


    def select_grasp(self, item_pos, arm):
        pass


if __name__ == "__main__":
    # Instantiate the simulation
    sim = RobosuiteSim()


    # Function to generate and add actions
    def generate_actions(sim):
        for _ in range(100):
            # Create a random action
            action = np.random.uniform(-1, 1, size=sim.env.action_dim)
            sim.add_action(action)
            time.sleep(0.05)  # Sleep before adding the next action


    # Start generating actions in the main thread or another thread
    generate_actions(sim)

    # Keep the main thread alive if needed
    while True:
        time.sleep(1)
