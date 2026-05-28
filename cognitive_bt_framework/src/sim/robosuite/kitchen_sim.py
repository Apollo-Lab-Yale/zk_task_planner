import mujoco
from mavis_mujoco_gym.utils.create_env import create_env
from mavis_mujoco_gym.envs.mavis_base.mavis_base_env import XARM_MODEL_PATH
from mavis_mujoco_gym.utils.mujoco_utils import MujocoModelNames

import time
import numpy as np
from dm_control import mujoco as mj_ctl

ENV_NAME='Microwave-v0'
env_configs = {
    "render_fps": 50,
    "robot_noise_ratio": 0.01,
    "obs_space_type": "config_space",
    "act_space_type": "config_space",
    "action_normalization": True,
    "observation_normalization": True,
    "render_mode": "human",
    "img_width": 640,
    "img_height": 480,
    "enable_rgb": True,
    "enable_depth": False,
    "use_cheating_observation": False
}

class KitchenSim(object):
    def __init__(self, render=True):
        self.env = create_env(ENV_NAME, env_configs)
        self.env.reset()
        self.render = render
        self.grasp_arm_start_joint = 1
        self.grasp_arm_end_joint = 8
        self.grasping_point_site_id = self.env.grasping_point_site_id

    def compute_jacobian(self):
        jacp = np.zeros((3, self.env.model.nv))
        jacr = np.zeros((3, self.env.model.nv))

        grasping_point_pos = self.env.data.site_xpos[self.grasping_point_site_id]

        # Get the body ID to which the site belongs
        grasping_body_id = self.env.model.site_bodyid[self.grasping_point_site_id]

        # Compute the Jacobian at the current state of the robot
        mujoco.mj_jac(self.env.model, self.env.data, jacp, jacr, grasping_point_pos, grasping_body_id)

        return jacp, jacr

    def compute_end_effector_velocity(self, target_pos):
        # Get the current end-effector position
        current_pos = self.env.data.site_xpos[self.grasping_point_site_id]
        # Compute the error between the current position and target position
        print(current_pos)
        position_error = target_pos - current_pos
        print(position_error)
        # Compute the desired end-effector velocity
        end_effector_velocity = position_error * -1.0  # Scaling factor for the velocity

        return end_effector_velocity

    def compute_joint_velocities(self, desired_velocity):
        # Compute the Jacobian at the current configuration
        jacp, _ = self.compute_jacobian()
        print(desired_velocity)
        # Solve for the joint velocities: q_dot = J_pseudo_inv * x_dot
        # We can use the pseudoinverse of the Jacobian to solve for joint velocities
        jacp_pseudo_inv = np.linalg.pinv(jacp)
        print('here1')
        joint_velocities = np.dot(jacp_pseudo_inv, desired_velocity)
        print('here2')
        return joint_velocities

    def expand_to_full_space(self, xarm_obj):
        new_obj = np.zeros(self.env.data.qpos.shape[0])
        new_obj[self.grasp_arm_start_joint: self.grasp_arm_end_joint] = xarm_obj
        return new_obj

    def move_to_position(self, target_pos):
        success = False
        for _ in range(200):  # Number of iterations to move towards the target
            # Compute the desired end-effector velocity
            desired_velocity = self.compute_end_effector_velocity(target_pos)

            # Compute the joint velocities using the Jacobian
            joint_velocities = self.compute_joint_velocities(desired_velocity)

            # Apply the joint velocities (velocity control)

            vel_action = joint_velocities[:17]

            # Step the simulation
            self.env.step(action=vel_action)

            # Optionally render the simulation
            if self.render:
                self.env.render()

            # Check if the end-effector is close enough to the target
            current_pos = self.env.data.site_xpos[self.grasping_point_site_id]
            if np.linalg.norm(current_pos - target_pos) < 0.01:  # 1 cm tolerance
                print('success!')
                break
        print(success)

    def grab_object(self, obj):
        pass

    def place_object(self, obj):
        pass

if __name__ == "__main__":

    sim = KitchenSim(render=True)
    sim.move_to_position(target_pos=np.array([0., 0., 0.]))

