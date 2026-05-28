from Go2Py.sim.mujoco import Go2Sim
from Go2Py.robot.model import Go2Model

robot = Go2Sim('highlevel')
model = Go2Model()
print("sup")
robot.standUpReset()
print("sup done")
running = True
print(robot.data.qpos)
print(len(robot.data.qpos))
while running:
   robot.step(0.0,0,0)
