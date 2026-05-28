from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
def get_put_apple_in_fridge_goal(sim : AI2ThorSimEnv):
    graph = sim.get_graph()
    apple = [node for node in graph["objects"] if "Apple" in node["objectId"]][0]
    fridge = [node for node in graph["objects"] if "Fridge" in node["objectId"]][0]
    print("graph expanded")
    print(fridge)
    goals = [f"INSIDE {apple['objectId']} {fridge['objectId']}"]
    return goals, f"put the {apple['objectId']} in the {fridge['objectId']}."

def get_slice_bread(sim):
    graph = sim.get_graph()
    bread = [node for node in graph["objects"] if "Bread" in node["objectId"]][0]
    goals = [f"SLICED {bread['objectId']}"]
    return goals, f"slice the {bread['objectId']}."

def get_wash_mug_in_sink_goal(sim):
    graph = sim.get_graph()
    cup = [node for node in graph["objects"] if "Mug" in node["objectId"]][0]
    sink = [node for node in graph["objects"] if "SinkBasin" in node["objectId"]][0]
    faucet = [node for node in graph["objects"] if "Faucet" in node["objectId"]][0]
    sim.make_object_dirty(cup)
    goals = [f"WASHED_SINK {cup['objectId']} {sink['objectId']} {faucet['objectId']}"]
    return goals, f"Wash the {cup['objectId']} with {faucet['objectId']} in the {sink['objectId']}"

def get_make_coffee(sim):
    graph = sim.get_graph()
    cup = [node for node in graph["objects"] if "Mug" in node["objectId"] and node['canFillWithLiquid']][0]
    coffee_maker = [node for node in graph["objects"] if "CoffeeMachine" in node["objectId"]][0]
    goals = [f"ON {coffee_maker['objectId']}", f"ON {cup['objectId']} {coffee_maker['objectId']}"]
    return ""



def get_make_toast(sim):
    graph = sim.get_graph()
    bread = [node for node in graph["objects"] if "Bread" in node["objectId"]][0]
    toaster = [node for node in graph["objects"] if "Toaster" in node["objectId"]][0]
    counter = [node for node in graph["objects"] if "counter" in node["objectId"].lower()][0]
    goals = [f"SLICED {bread['objectId']}", f"COOKED {bread['objectId']}|BreadSliced_1 {toaster['objectId']}"]
    return goals, f"slice the {bread['objectId']} on the {counter['objectId']}. And cook {bread['objectId']}|BreadSliced_1 in the {toaster['objectId']}"