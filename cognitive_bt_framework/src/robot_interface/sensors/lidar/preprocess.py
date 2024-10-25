import numpy as np
import open3d as o3d

# Preprocessing Class
class Preprocess:
    def __init__(self, lidar_type="OUST64", blind=0.01, point_filter_num=1):
        self.lidar_type = lidar_type
        self.blind = blind
        self.point_filter_num = point_filter_num

    def filter_and_downsample(self, scan_msg):
        points = np.array([[p.x, p.y, p.z] for cloud in scan_msg for p in cloud.points])

        distances = np.linalg.norm(points, axis=1)
        filtered_points = points[distances > self.blind]

        filtered_points = filtered_points[::self.point_filter_num]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        voxel_size = 0.1
        downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

        return downsampled_cloud