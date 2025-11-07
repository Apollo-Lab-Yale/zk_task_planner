from xarm.wrapper import XArmAPI
import time

joints = [0,
        -1.694867,
        -0.663357,
        0.862007,
        -0.501063,
        1.736574,
        -0.617966]

robot = XArmAPI('192.168.1.224', is_radian=True)


robot.clean_error()
robot.clean_warn()
robot.motion_enable(True)
time.sleep(1)
robot.set_mode(1)
time.sleep(1)
robot.set_collision_sensitivity(5)
time.sleep(1)
robot.set_state(state=0)
time.sleep(1)
robot.set_mode(1)
time.sleep(1)

print(robot.set_servo_angle(angle=joints, is_radian=True))