import threading
import time
import numpy as np
import signal
import matplotlib.pyplot as plt
import open3d as o3d

from cognitive_bt_framework.src.robot_interface.sensors.lidar.imu_processing import ImuProcess
from cognitive_bt_framework.src.robot_interface.sensors.lidar.estimator import Estimator
from cognitive_bt_framework.src.robot_interface.sensors.lidar.preprocess import Preprocess
from cognitive_bt_framework.src.robot_interface.sensors.lidar.lidar_subscriber_udp import UnitreeUDPServer


# SLAM system in Python
class SLAMSystem:
    def __init__(self, lidar_ip="127.0.0.1", lidar_port=12345, max_clouds=10):
        # Initialize components for processing IMU and LiDAR
        self.server = UnitreeUDPServer(ip=lidar_ip, port=lidar_port, max_clouds=max_clouds)
        self.preprocessor = Preprocess(lidar_type="OUST64", blind=0.01, point_filter_num=1)
        self.imu_processor = ImuProcess()
        self.estimator = Estimator()

        # For storing synced sensor data
        self.lidar_buffer = []
        self.imu_queue = []
        self.lock = threading.Lock()

        # Flag to manage thread shutdowns
        self.running = True
        self.process_increments = 0

        # Threads for LiDAR and IMU data synchronization
        self.lidar_thread = threading.Thread(target=self.process_lidar_data)
        self.imu_thread = threading.Thread(target=self.process_imu_data)
        self.visualization_thread = threading.Thread(target=self.visualize_data)


        # To handle sensor data synchronization
        self.sync_event = threading.Event()

        # Initialize data for odometry publishing
        self.lidar_data = None
        self.imu_data = None
        self.lidar_end_time = 0
        # Data for visualization
        self.path = []  # To store robot positions over time
        self.lidar_points = None

        # Visualization setup with Open3D
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()  # Point cloud object for LiDAR points
        self.path_line_set = o3d.geometry.LineSet()  # Line set for robot's path

        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.path_line_set)

    def start(self):
        """Start the SLAM system."""
        self.server.start()
        self.lidar_thread.start()
        self.imu_thread.start()
        self.visualization_thread.start()

    def process_lidar_data(self):
        """Process incoming LiDAR data in a separate thread."""
        while self.running:
            with self.lock:
                self.lidar_data = self.server.get_last_n_data()[0]  # Get LiDAR data from the UDP server
                if self.lidar_data:
                    # Preprocess the LiDAR data
                    downsampled_data = self.preprocessor.filter_and_downsample(self.lidar_data)
                    self.lidar_buffer.append(downsampled_data)
                    # Notify IMU thread for synchronization
                    self.sync_event.set()
            time.sleep(0.01)

    def process_imu_data(self):
        """Process incoming IMU data in a separate thread."""
        while self.running:
            with self.lock:
                self.imu_data = self.server.get_last_n_data()[1]  # Get IMU data
                if self.imu_data:
                    # Process IMU measurements
                    # self.imu_data = self.imu_data[-1]
                    self.imu_processor.process(self.imu_data)
                    # Sync the data with the latest LiDAR data
                    self.imu_queue.append(self.imu_data)
                    self.sync_lidar_imu()
            self.sync_event.wait(timeout=0.1)

    def sync_lidar_imu(self):
        """Sync IMU and LiDAR data, then perform estimation."""
        if self.lidar_buffer and self.imu_queue:
            # Process SLAM estimation using synced LiDAR and IMU data
            lidar_data = self.lidar_buffer.pop(0)
            imu_data = self.imu_queue.pop(0)
            # Run estimation and get the new position
            imu_data = imu_data[0]
            new_position = self.estimator.process_measurement(lidar_data, imu_data)
            self.path.append(new_position)
            self.lidar_points = lidar_data  # Store the LiDAR points for visualization

            # Run estimation
            # self.estimator.process_measurement(lidar_data, imu_data)
            print(f"Processed synced LiDAR and IMU data, updated state.")

    def visualize_data(self):
        """Thread to visualize the robot path and LiDAR point cloud."""
        while self.running:
            with self.lock:
                if self.lidar_points:
                    # Update the LiDAR point cloud in Open3D
                    lidar_xyz = np.array([point for point in self.lidar_points.points])
                    self.pcd.points = o3d.utility.Vector3dVector(lidar_xyz)

                    # If we have a path, update the path line set
                    if len(self.path):
                        path_xyz = np.array(self.path)
                        print(path_xyz)
                        points = o3d.utility.Vector3dVector(path_xyz)
                        lines = [[i, i + 1] for i in range(len(path_xyz) - 1)]
                        self.path_line_set.points = points
                        self.path_line_set.lines = o3d.utility.Vector2iVector(lines)

                    # Update the Open3D visualization
                    self.vis.update_geometry(self.pcd)
                    self.vis.update_geometry(self.path_line_set)
                    self.vis.poll_events()
                    self.vis.update_renderer()

            time.sleep(0.1)

    def stop(self):
        """Stop the SLAM system."""
        self.running = False
        self.lidar_thread.join()
        self.imu_thread.join()
        self.visualization_thread.join()
        self.vis.destroy_window()

    def signal_handler(self, sig, frame):
        """Handle system shutdown signal."""
        self.stop()
        print("SLAM system shutting down.")


if __name__ == '__main__':
    slam_system = SLAMSystem()
    signal.signal(signal.SIGINT, slam_system.signal_handler)  # Handle SIGINT (Ctrl+C)
    slam_system.start()

    # Keep the main thread alive to allow processing to continue
    try:
        while slam_system.running:
            time.sleep(1)
    except KeyboardInterrupt:
        slam_system.stop()
