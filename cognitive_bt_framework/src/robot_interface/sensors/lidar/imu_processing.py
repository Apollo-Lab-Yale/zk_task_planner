import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque


# IMU processing class based on the provided C++ example
class ImuProcess:
    def __init__(self):
        # Initialize parameters
        self.mean_acc = np.array([0, 0, -1.0])  # Default mean acceleration
        self.mean_gyr = np.array([0, 0, 0])  # Default mean gyro
        self.gravity_ = np.array([0, 0, -9.81])  # Default gravity vector (Earth's gravity)

        self.imu_en = True
        self.imu_need_init_ = True
        self.b_first_frame_ = True
        self.gravity_align_ = False
        self.init_iter_num = 1

        self.MAX_INI_COUNT = 100  # Maximum initialization count for IMU

    def reset(self):
        """Reset the IMU process."""
        print("Resetting IMU Process...")
        self.mean_acc = np.array([0, 0, -1.0])
        self.mean_gyr = np.array([0, 0, 0])
        self.imu_need_init_ = True
        self.init_iter_num = 1

    def imu_init(self, imu_data_queue):
        """Initialize IMU by calculating mean acceleration and gyro bias."""
        print(f"IMU Initializing: {self.init_iter_num / self.MAX_INI_COUNT * 100:.1f}%")

        if self.b_first_frame_:
            self.reset()
            self.b_first_frame_ = False
            imu_sample = imu_data_queue[0]
            self.mean_acc = np.array(imu_sample.linear_acceleration)
            self.mean_gyr = np.array(imu_sample.angular_velocity)

        for imu_data in imu_data_queue:
            cur_acc = np.array(imu_data.linear_acceleration)
            cur_gyr = np.array(imu_data.angular_velocity)

            self.mean_acc += (cur_acc - self.mean_acc) / self.init_iter_num
            self.mean_gyr += (cur_gyr - self.mean_gyr) / self.init_iter_num
            self.init_iter_num += 1

        if self.init_iter_num >= self.MAX_INI_COUNT:
            print("IMU Initialization Complete")
            self.imu_need_init_ = False

    def process(self, imu_data_queue):
        """Main IMU processing logic to handle point cloud and IMU alignment."""
        if not self.imu_en or not imu_data_queue:
            return

        if self.imu_need_init_:
            # Initialize IMU before processing data
            self.imu_init(imu_data_queue)
            return

        if not self.gravity_align_:
            print("Gravity alignment complete.")
            self.gravity_align_ = True
        # Here we can apply gravity alignment or other transformations to the point cloud
        # For now, we simply return the point cloud.


    def set_init(self, tmp_gravity, rot_matrix):
        """Set the initial gravity and rotation matrix."""
        hat_grav = np.array([[0, self.gravity_[2], -self.gravity_[1]],
                             [-self.gravity_[2], 0, self.gravity_[0]],
                             [self.gravity_[1], -self.gravity_[0], 0]])

        align_norm = np.linalg.norm(np.dot(hat_grav, tmp_gravity)) / (
                    np.linalg.norm(tmp_gravity) * np.linalg.norm(self.gravity_))
        align_cos = np.dot(self.gravity_.T, tmp_gravity) / (np.linalg.norm(self.gravity_) * np.linalg.norm(tmp_gravity))

        if align_norm < 1e-6:
            if align_cos > 1e-6:
                rot_matrix = np.eye(3)
            else:
                rot_matrix = -np.eye(3)
        else:
            align_angle = np.dot(hat_grav, tmp_gravity) / np.linalg.norm(np.dot(hat_grav, tmp_gravity)) * np.arccos(
                align_cos)
            rot_matrix = R.from_rotvec(align_angle).as_matrix()
