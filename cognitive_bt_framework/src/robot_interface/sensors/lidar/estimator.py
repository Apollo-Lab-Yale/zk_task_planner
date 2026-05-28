import numpy as np
from scipy.linalg import block_diag
from collections import deque


# Helper functions for rotation matrices and skew-symmetric matrices
def skew_symmetric(v):
    """Generate a skew-symmetric matrix from a vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rotation_matrix_from_quaternion(q):
    """Convert a quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])


class Estimator:
    def __init__(self):
        # State variables
        self.state_input = np.zeros(24)
        self.state_output = np.zeros(30)

        # Covariance matrices for process noise
        self.process_noise_cov_input = np.zeros((24, 24))
        self.process_noise_cov_output = np.zeros((30, 30))

        # Measurement noise covariance matrix
        self.laser_point_cov = np.eye(3) * 0.01  # Example value

        # EKF variables
        self.kf_input = np.zeros((24, 24))  # Kalman filter for input state
        self.kf_output = np.zeros((30, 30))  # Kalman filter for output state

        # For storing point clouds and nearest neighbors
        self.point_cloud_queue = deque()
        self.normvec = deque()
        self.crossmat_list = deque()

        # Some additional flags and parameters
        self.gravity = np.array([0, 0, -9.81])  # Gravity in the global frame
        self.Lidar_T_wrt_IMU = np.zeros(3)
        self.Lidar_R_wrt_IMU = np.eye(3)
        self.point_selected_surf = np.zeros(100000, dtype=bool)
        self.Nearest_Points = []  # Placeholder for nearest neighbor search results

    def generate_process_noise_cov_input(self):
        """Generate the process noise covariance matrix for input."""
        cov = np.zeros((24, 24))
        cov[3:6, 3:6] = np.eye(3) * 0.03  # Gyro noise
        cov[12:15, 12:15] = np.eye(3) * 0.01  # Accelerometer noise
        return cov

    def generate_process_noise_cov_output(self):
        """Generate the process noise covariance matrix for output."""
        cov = np.zeros((30, 30))
        cov[12:15, 12:15] = np.eye(3) * 0.02  # Velocity noise
        cov[15:18, 15:18] = np.eye(3) * 0.02  # Gyro noise
        return cov

    def predict(self, state, control_input, dt):
        """EKF predict step: updates the state with the process model."""
        F = self.get_f_jacobian(state, control_input)  # Jacobian of the system model
        state_pred = self.get_f(state, control_input) * dt  # Prediction step
        state_cov_pred = F @ self.kf_input @ F.T + self.generate_process_noise_cov_input()  # Fixed here
        return state_pred, state_cov_pred

    def update(self, state_pred, state_cov_pred, measurement):
        """EKF update step: updates the state with the measurement model."""
        H = self.get_h_jacobian(state_pred)  # Jacobian of the measurement model
        K = state_cov_pred @ H.T @ np.linalg.inv(H @ state_cov_pred @ H.T + self.laser_point_cov)  # Kalman gain
        state_updated = state_pred + K @ (measurement - self.get_h(state_pred))
        state_cov_updated = (np.eye(len(state_pred)) - K @ H) @ state_cov_pred
        return state_updated, state_cov_updated

    def get_f(self, state, control_input):
        """System model: describes how the state evolves without measurement."""
        # Update velocity, angular velocity, and accelerations using input
        omega = control_input.angular_velocity  # Gyro data
        acc_inertial = np.dot(state[6:9], control_input.linear_acceleration) + self.gravity
        state_dot = np.zeros(24)
        state_dot[:3] = state[12:15]  # velocity
        state_dot[12:15] = acc_inertial
        return state_dot

    def get_h(self, state):
        """Measurement model: relates state to the measurements."""
        return state[:3]  # Return position (for example, to compare to LiDAR points)

    def get_f_jacobian(self, state, control_input):
        """Jacobian of the system model (partial derivatives)."""
        F = np.zeros((24, 24))
        F[:3, 12:15] = np.eye(3)  # Derivative of position with respect to velocity
        if type(control_input) is list:
            control_input = control_input[0]
        F[12:15, 3:6] = skew_symmetric(control_input.linear_acceleration)  # Acceleration with respect to orientation
        return F

    def get_h_jacobian(self, state):
        """Jacobian of the measurement model (partial derivatives)."""
        H = np.zeros((3, 24))
        H[:3, :3] = np.eye(3)  # Measurement relates to position
        return H

    def process_measurement(self, point_cloud, imu_data):
        """Process LiDAR and IMU data to update the state."""
        # Predict step with IMU data (gyro and accel)
        state_pred, state_cov_pred = self.predict(self.state_input, imu_data, dt=0.01)

        # Extract LiDAR points from the point cloud (e.g., nearest neighbors)
        lidar_measurements = self.get_lidar_measurements(point_cloud)

        # Update step with LiDAR data
        state_updated, state_cov_updated = self.update(state_pred, state_cov_pred, lidar_measurements)

        # Save updated state
        self.state_input = state_updated
        self.kf_input = state_cov_updated

    def get_lidar_measurements(self, point_cloud):
        """Extract nearest neighbors from the point cloud."""
        # Placeholder: actual nearest neighbor search to be implemented
        return np.zeros(3)

    def point_body_to_world(self, point_body):
        """Convert point from body frame to world frame."""
        point_world = self.Lidar_R_wrt_IMU @ point_body + self.Lidar_T_wrt_IMU
        return point_world


# Example usage
if __name__ == "__main__":
    estimator = Estimator()

    # Simulated control inputs (IMU data)
    imu_data = np.array([0.01, 0.01, 0.02, 0.1, 0.1, 0.2])  # Gyro and accel data

    # Simulated point cloud data (LiDAR)
    point_cloud = np.random.rand(100, 3)  # Random point cloud data

    # Process the LiDAR and IMU data through the EKF
    estimator.process_measurement(point_cloud, imu_data)

    # Updated state
    print(f"Updated state: {estimator.state_input}")
