import socket
import struct
import threading
from collections import deque

# Point Type
class PointUnitree:
    def __init__(self, x, y, z, intensity, time, ring):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity
        self.time = time
        self.ring = ring

# Scan Type
class ScanUnitree:
    def __init__(self, stamp, id, validPointsNum, points):
        self.stamp = stamp
        self.id = id
        self.validPointsNum = validPointsNum
        self.points = points

# IMU Type
class IMUUnitree:
    def __init__(self, stamp, id, quaternion, angular_velocity, linear_acceleration):
        self.stamp = stamp
        self.id = id
        self.quaternion = quaternion
        self.angular_velocity = angular_velocity
        self.linear_acceleration = linear_acceleration

# Main UDP server class to handle the UDP communication and maintain the queue
class UnitreeUDPServer:
    def __init__(self, ip="127.0.0.1", port=12345, max_clouds=10):
        self.UDP_IP = ip
        self.UDP_PORT = port
        self.max_clouds = max_clouds
        self.point_cloud_queue = deque(maxlen=self.max_clouds)
        self.imu_queue = deque(maxlen=self.max_clouds)
        self.lock = threading.Lock()

        # Calculate Struct Sizes
        self.imuDataStr = "=dI4f3f3f"
        self.imuDataSize = struct.calcsize(self.imuDataStr)

        self.pointDataStr = "=fffffI"
        self.pointSize = struct.calcsize(self.pointDataStr)

        self.scanDataStr = "=dII" + 120 * "fffffI"
        self.scanDataSize = struct.calcsize(self.scanDataStr)

        self.server_thread = threading.Thread(target=self.udp_server, daemon=True)

    def start(self):
        """Start the server in a separate thread."""
        self.server_thread.start()
        print(f"UDP server started on {self.UDP_IP}:{self.UDP_PORT}")

    def udp_server(self):
        """Run the UDP server to listen for incoming Unitree messages."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.UDP_IP, self.UDP_PORT))
        print(f"Listening for UDP data on {self.UDP_IP}:{self.UDP_PORT}")

        while True:
            data, addr = sock.recvfrom(10000)
            msgType = struct.unpack("=I", data[:4])[0]

            if msgType == 102:  # Scan Message
                self.parse_and_store_scan(data)
            if msgType == 101:
                self.parse_and_store_imu(data)

    def parse_and_store_scan(self, data):
        """Parse the ScanUnitree data and store it in the queue."""
        length = struct.unpack("=I", data[4:8])[0]
        stamp = struct.unpack("=d", data[8:16])[0]
        id = struct.unpack("=I", data[16:20])[0]
        validPointsNum = struct.unpack("=I", data[20:24])[0]
        scanPoints = []
        pointStartAddr = 24

        for i in range(validPointsNum):
            pointData = struct.unpack(self.pointDataStr, data[pointStartAddr: pointStartAddr + self.pointSize])
            pointStartAddr += self.pointSize
            point = PointUnitree(*pointData)
            scanPoints.append(point)

        scanMsg = ScanUnitree(stamp, id, validPointsNum, scanPoints)

        with self.lock:
            self.point_cloud_queue.append(scanMsg)
            # print(f"Stored scan message {scanMsg.id} with {scanMsg.validPointsNum} points.")

    def parse_and_store_imu(self, data):
        length = struct.unpack("=I", data[4:8])[0]
        imuData = struct.unpack(self.imuDataStr, data[8:8 + self.imuDataSize])
        imuMsg = IMUUnitree(imuData[0], imuData[1], imuData[2:6], imuData[6:9], imuData[9:12])
        with self.lock:
            self.imu_queue.append(imuMsg)
            # print(f"Stored IMU data massage: {imuMsg.id}")

    def get_last_n_data(self):
        """Retrieve the last N point clouds stored in the queue."""
        with self.lock:
            return list(self.point_cloud_queue), list(self.imu_queue)

# Example usage
if __name__ == '__main__':
    # Start the server
    unitree_server = UnitreeUDPServer()
    unitree_server.start()

    # Example function that could be run in another thread to access the point clouds
    def process_point_clouds():
        while True:
            last_clouds = unitree_server.get_last_n_data()
            if last_clouds:
                print(f"Processing {len(last_clouds)} point clouds...")
                # Do something with the point clouds
            # Add a small sleep to avoid busy waiting
            threading.Event().wait(1)
    process_point_clouds()
    # Start processing thread
    # processing_thread = threading.Thread(target=process_point_clouds, daemon=True)
    # processing_thread.start()
    # while True:

    input()
