import xml.etree.ElementTree as ET
from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_NO_TARGET

BASE_EXAMPLE = '''
<?xml version="1.0"?>
<root>
    <Selector>
        <Sequence name="Water Filling Task">
            <!-- Navigate to the room with the glass -->
            <Action name="walk_to_room" target="kitchen"/>
            <!-- Locate the glass -->
            <Selector name="Locate Glass">
                <!-- Check if the glass is visible -->
                <Condition name="visible" target="glass"/>
                <!-- If not visible, scan the room -->
                <Sequence name="Adjust and Scan Glass">
                    <Action name="scanroom" target="glass"/>
                    <Condition name="visible" target="glass"/>
                </Sequence>
            </Selector>
            <!-- Navigate to the glass -->
            <Action name="walk_to_object" target="glass"/>
            <!-- Pick up the glass -->
            <Action name="grab" target="glass"/>
            <!-- Locate the faucet -->
            <Selector name="Locate Faucet After Grabbing Glass">
                <!-- Check if the faucet is visible -->
                <Condition name="visible" target="faucet"/>
                <!-- If not visible, scan the room -->
                <Sequence name="Adjust and Scan Faucet">
                    <Action name="scanroom" target="faucet"/>
                    <Condition name="visible" target="faucet"/>
                </Sequence>
            </Selector>
            <!-- Navigate to the faucet -->
            <Action name="walk_to_object" target="faucet"/>
            <!-- Place the glass in the sink basin -->
            <Action name="put" target="sinkBasin"/>
            <!-- Check if faucet is off before turning it on -->
            <Selector name="Ensure Faucet Off">
                <Condition name="isToggled" target="faucet"/>
                <!-- Turn on the faucet if it is not toggled-->
                <Action name="switchon" target="faucet"/>
            </Selector>
            <!-- Fill the glass with water -->
            <Sequence name="Fill Glass">
                <!-- Check if the glass can be filled -->
                <Condition name="canFillWithLiquid" target="glass"/>
                <!-- Check if the glass is filled with liquid -->
                <Condition name="isFilledWithLiquid" target="glass"/>
            </Sequence>
            <!-- Turn off the faucet -->
            <Action name="switchoff" target="faucet"/>
            <!-- Pick up the filled glass from the sink basin -->
            <Action name="grab" target="glass"/>
            <!-- Place the filled glass on the countertop -->
            <Action name="put" target="CounterTop"/>
        </Sequence>
    </Selector>
</root>
'''

FORMAT_EXAMPLE = '''
<?xml version="1.0"?>
<root>
  <Selector>
    <!-- Subtask 1: Define the first Sequence and the conditions/actions with fallback options described in comments -->
    <Sequence name="Subtask1">
      <Selector>
        <!-- Primary conditions and Action for the first Action -->
        <Sequence>
          <Condition name="condition1_for_action1" target="target1_for_action1"/>
          <Condition name="condition2_for_action1" target="target2_for_action1"/>
          <Action name="action1"/>
        </Sequence>
        <!-- Fallback options if the conditions above fail (described in comments) -->
        <!-- Fallback conditions and Action for the first Action -->
        <Sequence>
          <Condition name="alternative_condition1_for_action1" target="alternative_target1_for_action1"/>
          <Action name="action2"/>
        </Sequence>
      </Selector>
      
      <!-- Repeat the pattern for the second Action with fallback described in comments -->
      <Selector>
        <Sequence>
          <Condition name="condition1_for_action2" target="target1_for_action2"/>
          <Condition name="condition2_for_action2" target="target2_for_action2"/>
          <Action name="action3"/>
        </Sequence>
        <!-- Fallback conditions and Action for the second Action -->
        <Sequence>
          <Condition name="alternative_condition1_for_action2" target="alternative_target1_for_action2"/>
          <Action name="action4"/>
        </Sequence>
      </Selector>
    </Sequence>
    
    <!-- Subtask 2: Define the second Sequence and the conditions/actions with fallback options described in comments -->
    <Sequence name="Subtask2">
      <Selector>
        <Sequence>
          <Condition name="condition1_for_action3" target="target1_for_action3"/>
          <Condition name="condition2_for_action3" target="target2_for_action3"/>
          <Action name="action5"/>
        </Sequence>
        <!-- Fallback conditions and Action for the third Action -->
        <Sequence>
          <Condition name="alternative_condition1_for_action3" target="alternative_target1_for_action3"/>
          <Action name="action6"/>
        </Sequence>
      </Selector>
      
      <!-- Repeat the pattern for the fourth Action with fallback described in comments -->
      <Selector>
        <Sequence>
          <Condition name="condition1_for_action4" target="target1_for_action4"/>
          <Condition name="condition2_for_action4" target="target2_for_action4"/>
          <Action name="action7"/>
        </Sequence>
        <!-- Fallback conditions and Action for the fourth Action -->
        <Sequence>
          <Condition name="alternative_condition1_for_action4" target="alternative_target1_for_action4"/>
          <Action name="action8"/>
        </Sequence>
      </Selector>
    </Sequence>
  </Selector>
</root>

'''

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
            self.target = self.target.replace('_', '')

    def execute(self, state, interface, memory):

        # Logic to execute the action
        print(f"Executing action: {self.name}")
        return *interface.execute_actions([f"{self.name} {self.target}"], memory), self.to_xml()# Modify and return the new state

    def to_xml(self):
        return f'<Action name="{self.name}" target="{self.target}"/>'

class Condition(Node):
    def __init__(self, name, target, xml_str):
        super().__init__(name)
        self.target = target
        if target is not None:
            self.target = self.target.replace('_', '')
        self.xml_str = xml_str

    def execute(self, state, interface, memory):
        # Evaluate the Conditionagainst the state
        success, msg = interface.check_condition(self.name, self.target, memory)
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
            success, msg, _ = child.execute(state, interface, memory)
            if success:
                return success, msg, ""
            self.failures.append(msg)
        print(f"{self.failures}")
        return False, f"{self.failures[-1]}", self.to_xml()

    def to_xml(self):
        children_xml = "\n".join(child.to_xml() for child in self.children)
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
                return False, f"{msg}.", self.to_xml()
        return True, "", ""

    def to_xml(self):
        children_xml = "\n".join(child.to_xml() for child in self.children)
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
        if target is None and element.get('name') not in AI2THOR_NO_TARGET:
            target = "TARGET MISSING"
            raise Exception(f"ERROR: Failed to parse <{node_type.lower()} name={element.get('name')} target={target}> DUE TO MISSING TARGET")
        if element.get('name') not in actions:
            raise Exception(f"ERROR: {element.get('name')} is not a valid action.")
        node = Action(element.get('name'), element.get('target'), xml_str)
        return node
    elif node_type == "Condition":
        target = element.get('target', None)
        if target is None:
            target = "TARGET MISSING"
            raise Exception(f"ERROR: Failed to parse <{node_type.lower()} name={element.get('name')} target={target}> DUE TO MISSING TARGET")
        if element.get('name') not in conditions:
            raise Exception(f"ERROR: {element.get('name')} is not a valid condition.")
        return Condition(element.get('name'), target, xml_str)
    elif node_type in ["Selector", "Sequence"]:
        node_class = globals()[node_type]
        node = node_class(element.get('name'), xml_str)
        for child_elem in element:
            child = parse_node(child_elem, actions, conditions)
            node.children.append(child)
        return node
    elif node_type == "Root":
        node = Sequence('root', xml_str)
        for child_elem in element:
            child = parse_node(child_elem, actions, conditions)
            node.children.append(child)
        return node
    raise ValueError(f"Unsupported node type: {node_type}")


DOMAIN_DEF = """
(define (domain virtual-home)
(:requirements :strips :typing :equality :disjunctive-preconditions :conditional-effects)
(:types Character
      Object
      Room
      Surface - Object
      Container - Object
)

  (:predicates
    ; States
    (open ?obj - Object)
    (closed ?obj - Object)
    (active ?obj)
    (off ?obj - Object)
    (breakable ?obj - Object)
    (can_open ?obj - Object)
    (grabbable ?obj - Object)
    (sittable ?obj - Object)
    (lieable ?obj - Object)
    (has_paper ?obj)
    (filled ?obj)
    (has_plug ?obj)
    (lookable ?obj - Object)
    (readable ?obj - Object)
    (eatable ?obj - Object)
    (clothes ?obj - Object)
    (containers ?obj - Object)
    (cuttable ?obj - Object)
    (drinkable ?obj - Object)
    (between ?obj1 - Object ?obj2 - Object)
    (hangable ?obj - Object)
    (surfaces ?obj - Object)
    (cover_object ?obj - Object)
    (cream ?obj - Object)
    (pourable ?obj - Object)
    (moveable ?obj - Object)
    (toggleable ?obj - Object)
    (recipient ?obj - Object)
    (receptacle ?obj - Object)
    (pickupable ?obj - Object)
    (dirty ?obj)
    (holds ?obj)
    (water_source ?obj)
    (cleaning_target ?obj)
    (on ?obj1 - Object ?obj2 - Object)
    (inside ?obj1 ?obj2)
    (in ?obj1 ?obj2)

    (facing ?obj1 - Object ?obj2 - Object)
    (holds_rh ?character - Character ?obj - Object)
    (holds_lh ?character - Character ?obj - Object)
    (sitting ?character)
    (close ?character ?object)
    (standing ?character)
    (has_switch ?object)
    (visible ?object)
    (hands_full ?character)
    (can_cook ?cont)
    (cooked ?obj ?cont)
    (in_room ?character ?room)
    (sliceable ?obj)
    (sliced ?obj)
  )

  ; Actions
    (:action walk_to_object
        :parameters (?character - Character ?obj1 - Object)
        :precondition (and (standing ?character) (visible ?obj1))
        :effect (and (close ?character ?obj1)
                    (forall (?obj - object) (when (not(close ?obj1 ?obj)) (not (visible ?obj))))
                    (visible ?obj1))
             )

    (:action walk_to_room
        :parameters (?character - Character ?room - Room)
        :precondition (standing ?character)
        :effect (and
                    (forall (?r - Room)
                        (when (inside ?character ?r) (not(inside ?character ?r))))
                   (inside ?character ?room)
                   (close ?character ?room)
                   (forall (?obj - Object)
                            (when (not(inside ?obj ?room)) (not (visible ?obj))))
                 )
                )

    (:action grab
        :parameters (?character - Character ?object - Object)
        :precondition (and (grabbable ?object) (close ?character ?object) (visible ?object) (not (hands_full ?character)))
        :effect (and (holds_rh ?character ?object)
                      (hands_full ?character)
                      (forall (?cont - Object)(when (inside ?object ?cont) (not (inside ?object ?cont))))
                )
        )

    (:action switchon
        :parameters (?character - Character ?object - Object)
        :precondition (and (has_switch ?object) (off ?object) (close ?character ?object) (visible ?object) (not (hands_full ?character)))
        :effect (active ?object))

    (:action switchoff
        :parameters (?character - Character ?object - Object)
        :precondition (and (has_switch ?object) (active ?object) (close ?character ?object) (visible ?object) (not (hands_full ?character)))
        :effect (off ?object))

    (:action put
        :parameters (?character - Character ?object - Object ?target - Object)
        :precondition (and (holds_rh ?character ?object) (close ?character ?target))
        :effect (and (not (holds_rh ?character ?object)) (on ?object ?target) (not (hands_full ?character))) )

    (:action putin
        :parameters (?character - Character
                      ?object - Object
                      ?container - Object)
        :precondition (and (holds_rh ?character ?object) (close ?character ?container) (open ?container))
        :effect (and (not (holds_rh ?character ?object)) (inside ?object ?container) (not (hands_full ?character))))

    (:action scanroom
        :parameters (?character - Character ?obj - Object ?room - Room)
        :precondition (and (inside ?character ?room))
        :effect (and

            (when (inside ?obj ?room) (visible ?obj))
        )
    )


    (:action open
        :parameters (?character - Character
                     ?container)
        :precondition (and (close ?character ?container)
                        (closed ?container)
                        (visible ?container)
                        (not(active ?container))
                       )
        :effect (and (open ?container)
                    (forall (?obj - Object)
                        (when (inside ?obj ?container)
                            (visible ?obj)
                        )
                    )
                    (not (closed ?container))
                )
     )

    (:action close
        :parameters (?character - Character
                     ?container)
        :precondition (and (close ?character ?container) (open ?container) (visible ?container))
        :effect (and (closed ?container)
                    (not (open ?container))
                    (forall (?obj - Object)
                                (when (inside ?obj ?container)
                                    (not(visible ?obj))
                                )
                    )
         )
    )

"""