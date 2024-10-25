from cognitive_bt_framework.utils.bt_utils import Node, Sequence, Selector, Action, Condition, parse_node
from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_ACTIONS, AI2THOR_PREDICATES

class BTCorrector(object):
    def __init__(self):
        pass

OBJ_PH = '<OBJ>'
ACT_PH = '<ACT>'
NODE_PH = '<NODE>'



action_effects = {
    "walk_to_object": ["isClose <OBJ>", "visible <OBJ>"],
    "scanroom": ["visible <OBJ>"],

}

LOCATE_OBJ_SUBTREE = '''
    <Selector>
        <Condition name="visible" target="<OBJ>" val=1/>
        <Action name="scanroom" target="<OBJ>" />
        <Sequence name="locate stored object"/>
            <Action name="getPossibleContainers" target="<OBJ>"/>
            <Action name="searchPossibleContainers" target="<OBJ>"/>
        <Sequence/>
    <Selector/>
'''

INTERACT_OBJ_SUBTREE = f'''
        <Sequence>
            {LOCATE_OBJ_SUBTREE}
            <Selector>
                <Condition name="isClose" target="<OBJ>" val=1/>
                <Action name="walk_to_object" target="<OBJ>"/>
            <Selector/>
            <Action name="queryObjAffordance" target="<OBJ>" recipient="<ACT>" val=1/>
            <Action name="<ACT>" target="<OBJ>"/>
        <Sequence/>
'''

def generate_enable_interact_subtree(obj, action):
    xml_tree = INTERACT_OBJ_SUBTREE
    xml_tree = xml_tree.replace(OBJ_PH, obj).replace(ACT_PH, action)

    xml_tree = xml_tree.replace(NODE_PH, "Action")
    xml_tree = f'''
        <Sequence>
            <Action name="search" target="{obj}"/>
            <Action name="{action}" target="{obj}"/>
        <Sequence/>
    '''
    return parse_node(xml_tree, AI2THOR_ACTIONS, AI2THOR_PREDICATES)

def generate_locate_subtree(obj, action, isCond=True):
    xml_tree = LOCATE_OBJ_SUBTREE
    xml_tree = xml_tree.replace(OBJ_PH, obj).replace(ACT_PH, action)
    if isCond:
        xml_tree += f"\n<Condition> name={action} target={obj}</Condition>\n"
    return parse_node(xml_tree, AI2THOR_ACTIONS, AI2THOR_PREDICATES)

def apply_action_effects(action, target, state):
    pass

def navigate_tree(root, state, interface, conditions, actions):
    """
    Traverses the behavior tree starting from the root node, hypothetically simulating the results of executing actions
    and updating the world state according to the effects of those actions.

    Args:
        root: The root node of the behavior tree.
        state: The current world state (a dictionary of predicates with values for objects).Action
        interface: The interface for interacting with the environment.
        conditions: A list of current state conditions (predicates) in the world.
        actions: A list of available actions.

    Returns:
        A tuple (success, state) where success is a boolean indicating if the tree execution was successful,
        and state is the updated world state.
    """
    isSeq = isinstance(root, Sequence)
    isSelector = isinstance(root, Selector)

    # Recursively navigate the tree
    for i in range(len(root.children)):
        child = root.children[i]
        if isinstance(child, Selector) or isinstance(child, Sequence):
            actions, conditions = navigate_tree(child, state, interface, conditions, actions)
            continue
        name = child.name
        isCond = isinstance(child, Condition)
        target = child.target
        isVisible = any(f"visible {target}" in cond for cond in conditions)
        isClose = any(f"close {target}" in cond for cond in conditions)
        if name == "search":
            conditions += [f"visible {target}", f"isClose {target}"]
            continue
        if isCond and not isVisible:
            root.children[i] = generate_locate_subtree(target, name, isCond)
            conditions.append(f"visible {target}")
        elif not isVisible or not isClose and not isCond:
            root.children[i] = generate_enable_interact_subtree(target, name)
            conditions.append(f"isClose {target}")
            conditions.append(f"visible {target}")


