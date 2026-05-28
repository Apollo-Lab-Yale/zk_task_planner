import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '…/…/…'))

from xarm.wrapper import XArmAPI
from configparser import ConfigParser
parser = ConfigParser()
parser.read('…/robot.conf')
try:
    ip = parser.get('xArm', 'ip')
except:
    ip = input('Please input the xArm ip address[10.1.10.199]:')
    if not ip:
        ip = '192.168.1.224'

arm = XArmAPI(ip)
time.sleep(0.5)

print('Cleaning warnings...')
code = arm.clean_warn()
print('clean_warn, code={}'.format(code))

print('Cleaning errors...')
code = arm.clean_error()
print('clean_error, code={}'.format(code))

print('Enabling motion...')
code = arm.motion_enable(True)
print('motion_enable, code={}'.format(code))

print('Enabling gripper...')
code = arm.set_gripper_enable(True)
print('set_gripper_enable, code={}'.format(code))

print('Setting mode to 0...')
code = arm.set_mode(0)
print('set_mode, code={}'.format(code))

print('Setting state to 0...')
code = arm.set_state(0)
print('set_state, code={}'.format(code))

print('Opening gripper...')
code = arm.set_gripper_position(800, wait=True)
print('set_gripper_position (open), code={}'.format(code))
time.sleep(1)

print('Closing gripper...')
code = arm.set_gripper_position(0, wait=True)
print('set_gripper_position (close), code={}'.format(code))
time.sleep(1)

