
from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
from cognitive_bt_framework.src.sim.ai2_thor.utils import StateGraphConverter

'''
trooms = [4, 7, 9, 11, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28]
'''

#
#
# GoToObject(robots[0], 'Fridge')
sim = AI2ThorSimEnv(28)
print(sim.get_graph())
state = sim.get_graph()
print(StateGraphConverter(state).convert_to_vhome_graph())