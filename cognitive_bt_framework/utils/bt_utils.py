import xml.etree.ElementTree as ET
from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_NO_TARGET, AI2THOR_ACTIONS, AI2THOR_PREDICATES

BASE_EXAMPLE = ''
FORMAT_EXAMPLE = ''
DOMAIN_DEF = ''

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def execute(self, state, interface, memory):
        raise NotImplementedError("This method should be overridden by subclasses")

    def to_xml(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class Action(Node):
    def __init__(self, name, target, xml_str):
        self.name = name

        self.target = target
        if target is not None:
            self.target = self.target#.replace('_', '')

    def execute(self, state, interface, memory):

        # Logic to execute the action
        print(f"Executing action: {self.name}")
        ret = interface.execute_actions([f"{self.name} {self.target}"], memory)
        if not ret[0]:
            print(f"Action {self.name} {self.target} failed to execute")
            if 'visibility' in ret[1].lower():
                ret = list(ret)
                ret[1] += f' Try search actions like <Action name=scannroom, target="{self.target}"/>'
            if 'with something if agent rotates' in ret[1].lower():
                ret = list(ret)
                ret[1] = ret[1].replace('with something if agent rotates', 'will COLLIDE with something if agent rotates')

        return *ret, self.to_xml()# Modify and return the new state

    def to_xml(self):
        return f'<Action name="{self.name}" target="{self.target}"/>'

class Condition(Node):
    def __init__(self, name, target, recipient, value, xml_str):
        super().__init__(name)
        self.target = target
        self.recipient = recipient
        self.value = value
        if target is not None:
            self.target = self.target#.replace('_', '')
        self.xml_str = xml_str

    def execute(self, state, interface, memory):
        # Evaluate the Conditionagainst the state
        print(f'Checking {self.to_xml()}')
        success, msg = interface.check_condition(self.name, self.target, self.recipient, memory, value = self.value)

        if not success:
            print(f'Condition {self.name}, {self.target}, value={self.value} returned FALSE')
        if 'These objects satisfy the condition' in msg:
            msg = f"{self.name} is FALSE for object {self.target}. " + msg
        return success, msg, self.to_xml()

    def to_xml(self):
        return self.xml_str

class Selector(Node):
    def __init__(self, name, xml_str):
        super().__init__(name)
        self.failures = []
        self.xml_str = xml_str

    def execute(self, state, interface, memory):
        for child in self.children:
            success, msg, child_xml = child.execute(state, interface, memory)
            if success:
                return success, msg, ""
            msg = msg if type(child).__name__ in ["Selector", "Sequence"] else f"Selector exited due to failure: {msg}"
            bt = child_xml if type(child).__name__ in ["Selector", "Sequence"] else self.to_xml()
        print(f"{self.failures}")
        return False, msg, bt

    def to_xml(self):
        # children_xml = "\n".join(child.to_xml() for child in self.children)
        return self.xml_str


class Sequence(Node):
    def __init__(self, name, xml_str):
        super().__init__(name)
        self.xml_str = xml_str

    def execute(self, state, interface, memory):
        for child in self.children:
            success, msg, child_xml = child.execute(state, interface, memory)
            if not success:
                print(f"{msg}")
                ret_xml = child_xml
                if type(child) not in [Sequence, Condition]:
                    ret_xml = self.to_xml()
                full_msg = f"""Sequence failed to execute all children due to failure: {msg}
                                If you expected further execution after this condition check consider replacing
                                sequence with selector node"""
                msg = msg if type(child).__name__ in ["Selector", "Sequence"] else full_msg
                bt = child_xml if type(child).__name__ in ["Selector", "Sequence"] else self.to_xml()
                return False, msg, bt
        return True, "", ""

    def to_xml(self):
        # children_xml = "\n".join(child.to_xml() for child in self.children)
        return self.xml_str



def parse_bt_xml(xml_content, actions, conditions):
    print(xml_content)
    root = ET.fromstring(xml_content)
    return parse_node(root, actions, conditions)

def parse_node(element, actions, conditions):
    node_type = element.tag.capitalize()
    xml_str = ET.tostring(element)
    if node_type == "Action":
        target = element.get('target', None)
        recipient = element.get('recipient', None)
        receptacle = element.get('receptacle', None)

        if target is None and element.get('name') not in actions:
            target = "TARGET MISSING"
            raise Exception(f"ERROR: Failed to parse <{node_type.lower()} name={element.get('name')} target={target}> DUE TO MISSING TARGET")
        if not any(element.get('name') in action for action in actions):
            raise Exception(f"ERROR: {element.get('name')} is not a valid action.")
        if element.get('name').lower() in ['put', 'putin']:
            if recipient is not None:
                target = recipient
            if receptacle is not None:
                target = receptacle
        node = Action(element.get('name'), target, xml_str)
        return node
    elif node_type == "Condition":
        target = element.get('target', None)
        recipient = element.get('recipient', None)
        value = element.get('value', '1') == '1'
        if target is None:
            target = "TARGET MISSING"
            raise Exception(f"ERROR: Failed to parse <{node_type.lower()} name={element.get('name')} target={target}> DUE TO MISSING TARGET")
        if element.get('name') not in conditions:
            raise Exception(f"ERROR: {element.get('name')} is not a valid condition.")

        return Condition(element.get('name'), target, recipient, value, xml_str)
    elif node_type in ["Selector", "Sequence"]:
        node_class = globals()[node_type]
        node = node_class(element.get('name'), xml_str)
        for child_elem in element:
            child = parse_node(child_elem, actions, conditions)
            node.children.append(child)
        return node
    elif node_type == "Root":
        node = Selector('root', xml_str)
        for child_elem in element:
            child = parse_node(child_elem, actions, conditions)
            node.children.append(child)
        return node
    raise ValueError(f"Unsupported node type: {node_type}")

