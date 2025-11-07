from ai2thor.controller import Controller
import numpy as np
from typing import Tuple
import time
import prior
import random
import base64
import time
import math
import threading
import cv2
import re
from cognitive_bt_framework.src.sim.ai2_thor.utils import get_visible_objects, get_predicates, CLOSE_DISTANCE, find_closest_position, is_in_room, get_yaw_angle, get_vhome_to_thor_dict, get_inf_floor_polygon, \
    NO_VALID_PUT, Event, PUT_COLLISION, AI2THOR_PREDICATES, closest_node, distance_pts
from cognitive_bt_framework.src.sim.ai2_thor.image_saver import SaveImagesThread

def parse_instantiated_predicate(predicate_str):
    parts = predicate_str.split()
    predicate_name = parts[0]
    target = parts[1] if len(parts) > 1 else None
    return predicate_name, parts[1:]

def get_ithor_scene_single_room(room, index = -1):
    broken_rooms = [2, 5, 17, 22]
    if index > 0:
        return f"FloorPlan{index}"
    # kitchens = [f"FloorPlan{i}" for i in range(1, 31) if i not in broken_rooms]
    kitchens = [f"FloorPlan{i}" for i in [4, 7, 9, 11, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28] if i not in broken_rooms]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]
    rooms = []
    if "kitchen" in room.lower():
        rooms = kitchens
        # rooms = [4, 6]
    if "livingroom" in room.lower().replace("_", ""):
        rooms = living_rooms
    if "bedroom" in room.lower():
        rooms = bedrooms
    if "bathroom" in room.lower():
        rooms = bathrooms
    assert len(rooms) > 0
    return np.random.choice(rooms)
action_queue = []




class AI2ThorSimEnv:
    def __init__(self, scene_index=-1, width=600, height=600, gridSize=0.25, visibilityDistance=20, single_room='kitchen', save_video=False, use_find = False):
        self.single_room = single_room
        self.scene = None
        self.controller = None
        self.comm = self
        self.image_saver = None
        self.save_video = save_video
        self.use_find = use_find
        self.reachable_positions = None
        self.reset_pose = None
        self.navigating = False
        self.reset(scene_index, width, height, gridSize, visibilityDistance, single_room, save_video)
        self.valid_positions = self.get_valid_positions()
        self.object_waypoints = {}
        self.object_names = self.get_object_names()
        self.action_fn_from_str = {
            "walk_to_room": self.navigate_to_room,
            "walk_to_object": self.navigate_to_object,
            "walk": self.navigate_to_object,
            "grab": self.grab_object,
            "turnleft": self.turn_left,
            "turnright": self.turn_right,
            "put": self.place_object,
            "putin": self.place_object,
            "putback": self.place_object,
            "open": self.open_object,
            "close": self.close_object,
            "cook": self.cook_object,
            "clean": self.clean_object,
            "switchon": self.turn_object_on,
            "switchoff": self.turn_object_off,
            "slice": self.slice_object,
            "find": self.cheat_and_find_object,
            "give-up": self.give_up,
            "cut": self.slice_object,
            "moveforward": self.move_forward,
            "movebackward": self.move_back,
            "turnaround": self.turn_around,
            "lookup": self.look_up,
            "lookdown": self.look_down,
            "scanroom": self.handle_scan_room,
            "break": self.break_obj,

        }
        self.room_names = ['kitchen']#, 'livingroom', 'bedroom', 'bathroom']
        self.nav_thread = None

        self.vhome_obj_ids = {}


    def exec_actions(self):
        c = self.controller
        while self.navigating:
            if len(action_queue) > 0:
                try:
                    act = action_queue[0]
                    if act['action'] == 'ObjectNavExpertAction':
                        multi_agent_event = c.step(
                            dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                        next_action = multi_agent_event.metadata['actionReturn']

                        if next_action != None:
                            multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=False)

                    elif act['action'] == 'RotateLeft':
                        multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])

                    elif act['action'] == 'RotateRight':
                        multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'],
                                                   agentId=act['agent_id'])

                except Exception as e:
                    print(e)
                    if 'write to closed file' in str(e).lower():
                        self.navigating = False
                        self.reset(self.scene_index)

                action_queue.pop(0)

            time.sleep(0.2)

    def reset(self, scene_index=-1, width=600, height=600, gridSize=0.25, visibilityDistance=10, single_room='kitchen', save_video=False):
        scene = None
        self.scene_index = scene_index
        self.vhome_obj_ids = {}
        self.save_video = save_video
        if single_room is not None and single_room != '':
            scene = get_ithor_scene_single_room(single_room, index=scene_index)
            self.scene_id = scene
        else:
            self.scene_id = f"FloorPlan{scene_index}"
            dataset = prior.load_dataset("procthor-10k")
            scene = dataset["train"][scene_index]
        self.scene = scene
        if self.controller is None:
            self.controller = Controller(scene=scene, agentMode="default",
                                         visibilityDistance=visibilityDistance, width=width,
                                         height=height, allowAutoSimulation=True, gridSize=gridSize)
        self.controller.reset(scene)
        self.get_vhome_obj_ids(self.get_graph())
        self.reset_pose = self.controller.last_event.metadata['agent']['position']
        reachable_positions_ = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        self.reachable_positions =  [(p["x"], p["y"], p["z"]) for p in reachable_positions_]
        self.scene = {"rooms": [{"roomType": single_room, "floorPolygon": get_inf_floor_polygon(), "name":scene}]}
        self.object_names = self.get_object_names()
        print(self.scene)
        if self.save_video:
            if self.image_saver is not None:
                self.image_saver.keep_running = False
            self.image_saver = SaveImagesThread(controller=self.controller)
            self.image_saver.start()

    def end_sim(self):
        self.controller.stop()
        if self.image_saver is not None:
            self.image_saver.keep_running = False

    def get_object_location(self, object_type):
        objects = self.controller.last_event.metadata['objects']
        for obj in objects:
            if obj['objectType'] == object_type:
                return obj['position']
        return None

    def turn_around(self, hold=None):
        act = np.random.choice([self.turn_left, self.turn_right])
        return act(180)

    def get_state(self, goal_objs=None):
        event = self.controller.last_event.metadata
        state = {}
        state["objects"] = event["objects"]
        state["objects"] = get_visible_objects(event["objects"])
        for object in state["objects"]:
            if 'surface' in object['name']:
                object['onTop'] = object['parentReceptacles']
        state["predicates"] = get_predicates(state["objects"])
        # state["room_names"] = [room['roomType'] for room in self.scene['rooms']]
        state["room_names"] = self.room_names
        state["robot_state"] = event['agent']
        state["memory_dict"] = {}
        return state

    def get_robot_state(self, state, robot = "character", relations=None):
        state = self.get_state()
        nl_state = ''
        agent_pose = state["robot_state"]["position"]
        robot_room = None
        for room in state["room_names"]:
            polygon = [room_obj for room_obj in self.scene["rooms"] if room_obj["roomType"]==room][0]["floorPolygon"]
            if is_in_room(agent_pose, polygon):
                robot_room = room

        holding = [obj['name'] for obj in state["objects"] if obj["isPickedUp"]]
        # nl_state = f"I am in the {robot_room}."
        if len(holding) == 0:
            nl_state+= " I am not holding anything."
        else:
            holding_str = ", ".join(holding)
            nl_state+=f" I am holding the {holding_str}."
        return nl_state, robot_room

    def cheat_and_find_object(self, object):
        print(object)
        try:
            obj = [obj for obj in self.get_graph()['objects'] if object.lower() in obj['name'].lower()][0]
            if self.use_find:
                self.navigate_to_object(obj)
            self.handle_scan_room(obj['name'], memory=None, pause_time=0)
        except Exception as e:
            print(f"Failed to find object {object}")

        return self.controller.last_event

    def done(self):
        pass
        # return self.controller.step(action="Done")

    def turn_left(self, degrees=90):
        if type(degrees) is not float:
            degrees = float(90)
        return self.controller.step("RotateLeft", degrees=degrees)

    def turn_right(self, degrees=90):
        if type(degrees) is not float:
            degrees = float(90)
        return self.controller.step("RotateRight", degrees=degrees)

    def look_up(self, degrees=10):
        return self.controller.step("LookUp", degrees=degrees)

    def look_down(self, degrees=10):
        return self.controller.step("LookDown", degrees=degrees)

    def move_forward(self, meters=1):
        if type(meters) != int and type(meters) != float:
            meters = 1
        return self.controller.step("MoveAhead", moveMagnitude=meters)

    def move_back(self, meters=0.25):
        return self.controller.step("MoveBack", moveMagnitude=meters)

    def break_obj(self, obj):
        return self.controller.step("BreakObject", objectId=obj['objectId'])

    def give_up(self):
        print("womp womp")
        return self.navigate_to_room(target_room_str=self.single_room)

    def wash(self, object):
        return not object['isDirty'], '' if not object['isDirty'] else (f'Failed to wash '
                                                                        f'{object["objectId"]}, '
                                                                        f'the object must be rinsed under water to wash.')

    def get_object_names(self):
        objects = self.get_graph()['objects']
        return [obj['name'].lower() for obj in objects]

    def navigate_to_room(self, target_room_str="bedroom"):
        return True, ""
        target_room = None
        for room in self.scene["rooms"]:
            if room["roomType"].lower() == target_room_str.lower():
                target_room = room
        assert target_room is not None
        positions = [pose for pose in self.valid_positions if is_in_room(pose, target_room["floorPolygon"])]
        teleport_pose = np.random.choice(positions)
        return self.controller.step(
            action="Teleport",
            position=teleport_pose,
            rotation={'x': 0.0, "y": np.random.uniform(0, 360), "z": 0.0},
            forceAction=False
        )

    def grab_object(self, object):
        if object["distance"] > CLOSE_DISTANCE:
            ret = Event()
            ret.metadata['lastActionSuccess'] = False
            ret.metadata['errorMessage'] = f'Failed to grab object {object["objectId"]}, agent is too far away navigate closer to interact.'
            return ret
        if object['isPickedUp']:
            ret = Event()
            ret.metadata['lastActionSuccess'] = True
            ret.metadata['errorMessage'] = ""
            print(f'Already holding {object["name"]}')
            return ret

        ret =  self.controller.step("PickupObject",
                             objectId=object["objectId"],
                             forceAction=False,
                             manualInteract=True)
        success = object['isPickedUp']
        state = self.get_state()
        cam_horizon = state['robot_state']['cameraHorizon']
        if abs(cam_horizon) > 0:
            if cam_horizon < 0:
                self.look_down(abs(cam_horizon))
            else:
                self.look_up(abs(cam_horizon) )
        self.controller.step("MoveHeldObjectUp", moveMagnitude=0.1, forceVisible=True)
        for i in range(20):
            self.controller.step("MoveHeldObjectUp", moveMagnitude=0.01, forceVisible=True)
            self.controller.step("MoveHeldObjectBack", moveMagnitude=0.01, forceVisible=True)
        return ret

    def place_object(self, target):
        held_obj = [obj for obj in self.get_graph()['objects'] if obj['isPickedUp']]
        if len(held_obj) == 0:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f'Cannot place onto {target["objectId"]} because no object is held'
            return fail_event
        if target['distance'] > CLOSE_DISTANCE and 'counter' not in target['name']:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = (f"Not close enough to {target['name'].lower()} to put the heald object. "
                                                   f"NAVIGATE CLOSER to {target['name'].lower()} to interact.")
            return fail_event
        held_obj = held_obj[0]
        ret = self.controller.step(
            action="PutObject",
            objectId=target["objectId"],
            forceAction=True,
            placeStationary=True
        )
        if ret.metadata['lastActionSuccess']:
            self.handle_scan_room(held_obj['name'], None)
        return ret

    def get_interactable_poses(self, object):
        positions = self.controller.step(
                action="GetInteractablePoses",
                objectId=object['name']
            ).metadata['actionReturn']
        return positions


    def exec_action(self):
        c = self.controller
        if len(action_queue) > 0:
            try:
                act = action_queue[0]
                if act['action'] == 'ObjectNavExpertAction':
                    multi_agent_event = c.step(
                        dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                    next_action = multi_agent_event.metadata['actionReturn']

                    if next_action != None:
                        multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=False)

                elif act['action'] == 'RotateLeft':
                    multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])

                elif act['action'] == 'RotateRight':
                    multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'],
                                               agentId=act['agent_id'])
                if not multi_agent_event.metadata['lastActionSuccess']:
                    c.step(action="ObjectNavExpertAction", position=self.reset_pose, agentId=act['agent_id'])
            except Exception as e:
                print(e)
                if 'write to closed file' in str(e).lower():
                    self.navigating = False
                    self.reset(self.scene_index)

            action_queue.pop(0)

        time.sleep(0.2)


    def GoToObject(self, dest_obj):
        print("Going to ", dest_obj)
        robots = [{'name': 'agent', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}]
        c = self.controller
        # check if robots is a list
        # reachable_positions_ = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        # self.reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]
        reachable_positions = self.reachable_positions
        if not isinstance(robots, list):
            # convert robot to a list
            robots = [robots]
        no_agents = len(robots)
        # robots distance to the goal
        dist_goals = [10.0] * len(robots)
        prev_dist_goals = [10.0] * len(robots)
        count_since_update = [0] * len(robots)
        clost_node_location = [0] * len(robots)

        # list of objects in the scene and their centers
        objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
        objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
        agent_id = 0
        # look for the location and id of the destination object
        for idx, obj in enumerate(objs):
            match = re.match(dest_obj, obj)
            if match is not None:
                dest_obj_id = obj
                dest_obj_center = objs_center[idx]
                break  # find the first instance

        dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]

        # closest reachable position for each robot
        # all robots cannot reach the same spot
        # differt close points needs to be found for each robot
        print(reachable_positions)
        crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

        goal_thresh = 0.3
        # at least one robot is far away from the goal
        max_iter = 30
        iter = 0
        while all(d > goal_thresh for d in dist_goals):
            if iter > max_iter:
                return False
            iter += 1

            for ia, robot in enumerate(robots):
                robot_name = robot['name']
                agent_id = 0

                # get the pose of robot
                metadata = c.last_event.events[agent_id].metadata
                location = {
                    "x": metadata["agent"]["position"]["x"],
                    "y": metadata["agent"]["position"]["y"],
                    "z": metadata["agent"]["position"]["z"],
                    "rotation": metadata["agent"]["rotation"]["y"],
                    "horizon": metadata["agent"]["cameraHorizon"]}

                prev_dist_goals[ia] = dist_goals[ia]  # store the previous distance to goal
                dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])

                dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
                print(ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
                if dist_del < 0.2:
                    # robot did not move
                    count_since_update[ia] += 1
                else:
                    # robot moving
                    count_since_update[ia] = 0

                if count_since_update[ia] < 15:
                    action_queue.append(
                        {'action': 'ObjectNavExpertAction', 'position': dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]),
                         'agent_id': agent_id})
                    self.exec_action()
                else:
                    # updating goal
                    clost_node_location[ia] += 1
                    count_since_update[ia] = 0
                    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

                # time.sleep(0.2)

        # align the robot once goal is reached
        # compute angle between robot heading and object
        metadata = c.last_event.events[agent_id].metadata
        robot_location = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"]}

        robot_object_vec = [dest_obj_pos[0] - robot_location['x'], dest_obj_pos[2] - robot_location['z']]
        y_axis = [0, 1]
        unit_y = y_axis / np.linalg.norm(y_axis)
        unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)

        angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
        angle = 360 * angle / (2 * np.pi)
        angle = (angle + 360) % 360
        rot_angle = angle - robot_location['rotation']

        print("Reached: ", dest_obj)
        return True

    def exec_acts(self):
        global action_queue
        c = self.controller
        multi_agent_event = self.controller.last_event
        while len(action_queue) > 0:
            act = action_queue[0]
            if act['action'] == 'ObjectNavExpertAction':
                multi_agent_event = c.step(
                    dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                next_action = multi_agent_event.metadata['actionReturn']

                if next_action != None:
                    multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=False)

            elif act['action'] == 'RotateLeft':
                multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])

            elif act['action'] == 'RotateRight':
                multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'],
                                           agentId=act['agent_id'])
            if not multi_agent_event.metadata['lastActionSuccess']:
                multi_agent_event = self.controller.step(action='Teleport', position=self.reset_pose, forceAction=True)
                action_queue = []
                return multi_agent_event
            action_queue.pop(0)
            time.sleep(0.2)
        return multi_agent_event


    def navigate_to_object(self, object):
        # # self.navigating = True
        # # self.nav_thread = threading.Thread(target=self.exec_actions)
        # self.nav_thread.start()
        obj = object
        if type(object) != str:
            obj = object['name']
        nav_obj = obj.split('_')[0]
        success = self.GoToObject(nav_obj)
        if not success:
            res = self.controller.step(action='Teleport', position=self.reset_pose)
            print("545"*100)
            print(res)
            return False, "failed to navigate to object"
        self.exec_acts()

        ret = self.controller.last_event
        self.navigating = False
        self.handle_scan_room(object['name'], memory=None)
        return ret
        positions = self.get_valid_positions()
        print('navigate to object')
        print(type(object))
        teleport_pose = None
        for i in range(50):
            teleport_pose = find_closest_position(object["position"], object["rotation"], positions, 1.5, facing=False)
            if teleport_pose is not None:
                break
        # teleport_pose = np.random.choice(positions)
        if teleport_pose is None:
            evt = Event()
            evt.metadata['lastActionSuccess'] = False
            evt.metadata['errorMessage'] = f"Failed to navigate to {object['name'].lower()} to interact."
            return evt
        event = self.controller.step(
            action="Teleport",
            position=teleport_pose,
            rotation={'x': 0.0, 'y': get_yaw_angle(teleport_pose,object['rotation'], object['position']), 'z': 0.0},
            forceAction=False
        )
        print("navigated to object scanning")
        self.handle_scan_room(object['name'], None)
        return event

    def open_object(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['name']} to open object. Navigate to object."
            return fail_event
        event = self.controller.step(
            action="OpenObject",
            objectId=object['objectId'],
            openness=1,
            forceAction=False
        )

        # Image.fromarray(event.frame).show("str")
        return event

    def close_object(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['name']} to close object. Navigate to object."
            return fail_event
        return self.controller.step(
            action="CloseObject",
            objectId=object['objectId'],
            forceAction=False
        )

    def cook_object(self, object):
        return self.controller.step(
            action="CookObject",
            objectId=object['objectId'],
            forceAction=False
        )

    def get_context(self, num_images=5):
        rotation = 360 / num_images
        pause_time = 0.1
        img = None
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 100]
        _, frame = cv2.imencode('.png', self.controller.last_event.cv2img)
        img = base64.b64encode(frame.tobytes()).decode('utf-8')
        images = [img]
        states = [self.get_state()]
        for i in range(num_images + 1):
            evt = self.turn_left(degrees=int(rotation))
            time.sleep(pause_time)
            self.done()
            _, frame = cv2.imencode('.png', self.controller.last_event.cv2img)
            img = base64.b64encode(frame.tobytes()).decode('utf-8')
            images.append(img)
            states.append(self.get_state())
        return images, states

    def slice_object(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['name']} to slice object.. Navigate to object."
            return fail_event
        return self.controller.step(
            action="SliceObject",
            objectId=object["objectId"],
            forceAction=False
        )

    def turn_object_on(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['name']} to switch object on.. Navigate to object."
            return fail_event
        if object['isToggled']:
            evt = Event()
            evt.metadata['lastActionSuccess'] = True
            evt.metadata['errorMessage'] = ''
            return evt
        return self.controller.step(
            action="ToggleObjectOn",
            objectId=object["objectId"],
            forceAction=False
        )

    def turn_object_off(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['name']} switch it off.. Navigate to object."
            return fail_event
        return self.controller.step(
            action="ToggleObjectOff",
            objectId=object["objectId"],
            forceAction=False
        )

    def get_valid_positions(self):
        return self.controller.step(
                    action="GetReachablePositions").metadata["actionReturn"]

    def make_object_dirty(self, object):
        return  self.controller.step(
            action="DirtyObject",
            objectId=object["objectId"],
            forceAction=True
        )

    def clean_object(self, object):

        return self.controller.step(
            action="CleanObject",
            objectId=object["objectId"],
            forceAction=True
        )

    def fill_with_liquid(self, object, liquid="water"):
        return self.controller.step(
            action="FillObjectWithLiquid",
            objectId=object["objectId"],
            fillLiquid=liquid,
            forceAction=False
        )

    def empty_liquid(self, object):
        return self.controller.step(
            action="EmptyLiquidFromObject",
            objectId=object["objectId"],
            forceAction=False
        )

    def handle_fry_in_pan(self, object):
        pass

    def drop_object(self):
        return self.controller.step(action="DropHandObject",
                             forceAction=False)

    def handle_scan_room(self, goal_obj, memory, pause_time = 0.2, is_goal=False):
        print(f"scanning for object {goal_obj}")
        action = random.choice([self.turn_left, self.turn_left, self.turn_right])
        action = self.turn_left
        goal_obj_name = goal_obj.split('_')[0]
        for j in range(3):
            state = self.get_state()
            if any([self.check_same_obj(goal_obj.lower(), object["name"].lower()) for object in
                    state['objects']]) or goal_obj.lower() in self.room_names:
                print(f"{goal_obj} found!")
                # if j > 0:
                #     fn_inv(45)
                return True, ""
            fn, fn_inv = None, None
            if j > 0:
                fn = self.look_down if j == 1 else self.look_up
                fn_inv = self.look_up if j == 1 else self.look_down
                fn(45.0)
            for i in range(12):
                evt = action(degrees=30.0)
                if not evt.metadata['lastActionSuccess']:
                    action = self.turn_right
                    evt = action(degrees=30.0)
                time.sleep(pause_time)
                self.done()
                state = self.get_state()
                # Bulk update all objects in the current state
                if memory is not None:
                    episode_id = memory.current_episode
                    objects_to_update = [(obj['name'], obj) for obj in state['objects']]
                    memory.store_multiple_object_states(objects_to_update, episode_id, self.scene_id)
                if not self.controller.last_event.metadata["lastActionSuccess"]:
                    return self.controller.last_event.metadata["lastActionSuccess"], self.controller.last_event.metadata[
                        "errorMessage"]
                if any([self.check_same_obj(goal_obj.lower(), object["name"].lower()) for object in self.get_state()['objects']]) or goal_obj.lower() in self.room_names:
                    print(f"{goal_obj} found!")
                    print([obj for obj in self.get_state()['objects'] if obj['name'].lower() == goal_obj.lower()])
                    # if j > 0:
                    #     fn_inv(45)
                    return True, ""
            if j > 0:
                fn_inv(45.0)
        return False, (f"Failed to find object {goal_obj} during scan_room, it may not be visible from my"
                       f" current position try movement actions or add sequences to search likely containers that {goal_obj} might be inside of.")

    def translate_action_for_sim(self, action, state):
        return [action]

    def check_same_obj(self, obj1, obj2):
        # return obj1.lower().split("_")[0] == obj2.lower().split("_")[0]
        return obj1.lower() == obj2.lower()

    def add_object_waypoint(self, x, y):
        pass

    def get_graph(self):
        return self.controller.last_event.metadata

    def environment_graph(self):
        graph = self.get_graph()
        graph['nodes'] = graph['objects']
        for node in graph['nodes']:
            node['class_name'] = node['objectType']
            node['id'] = '|'.join(node['name'])
        return True, graph

    def check_cooked(self):
        pass




    def check_condition(self, cond, target, recipient, memory, value=1):
        target = target.lower()
        current_state = self.get_state()  # Get the current sensor state
        episode_id = memory.current_episode  # Assumes there's a method to get the current episode ID
        if "inroom" in cond.lower():
            return True, ""
        # Bulk update all objects in the current state
        objects_to_update = [(obj['name'], obj) for obj in current_state['objects']]
        memory.store_multiple_object_states(objects_to_update, episode_id, self.scene_id)
        if 'handsfull' in cond.lower():
            hands_full = any(obj['isPickedUp'] for obj in self.get_graph()['objects'])
            return  hands_full== value, f'Condition handsFull is {hands_full}'
        if not any(self.check_same_obj(target, obj) for obj in self.object_names):
            return False, "There is no object named {} in this env to check condition.".format(target)
        if recipient is not None and not any(self.check_same_obj(recipient, obj) for obj in self.object_names):
            return False, "There is no object named {} in this env to check condition.".format(recipient)
        # Find specific target object
        object_state = next((obj for obj in current_state['objects'] if target in obj['name'].lower()), None)
        if object_state is None:
            return False, f'Cannot check condition {cond} for {target} because the state of {target} is unknown. Try search actions like <action name="scanroom" target={target}/>'
        # if object_state is None:
        #     # Fallback to memory
        #     object_state, _ = memory.retrieve_object_state(target)
        if "inroom" in cond.lower():
            return True, ""

        if "visible" in cond.lower():
            isVisible = object_state is not None or target.lower() in self.room_names
            print(object_state)
            msg = "" if isVisible else f"{target} is not currently visible./>."

            return isVisible, msg
        if "isclose" in cond.lower():
            isClose = object_state is not None and object_state['distance'] < CLOSE_DISTANCE
            msg = f"isClose {object_state['name'].lower()} is {isClose}"
            return isClose, msg
        if "isontop" in cond.lower() or "isinside" in cond.lower():
            current_state = self.get_graph()
            object_state = next((obj for obj in current_state['objects'] if target in obj['name'].lower()), None)
            if recipient is None:
                return False, (f"I failed to check {cond} target={target} because recipient is None. please provide a "
                               f"recipient in the following format "
                               f"<Condition name={cond} target={target} recipient=<object to check if {target} satisfies {cond}>> ")
            surfCont = [obj for obj in self.get_graph()['objects'] if recipient.lower() in obj['name'].lower()]
            if len(surfCont) == 0:
                surfCont, _ = memory.retrieve_object_state(recipient)
                surfCont = surfCont
            else:
                surfCont = surfCont[0]
            if surfCont is None:
                return False, (f'I failed to check <condition name={cond} target={target} recipient={recipient}/> '
                               f'because the object state is unknown. Try search actions like '
                               f'<action name=scanroom target={target}/> or look for {target} inside recepticals.')
            # if surfCont['receptacleObjectIds'] is None or not surfCont['receptacle']:
            #     return False, f"{recipient} is not a receptical so {target} cannot be {cond}."
            print(surfCont)
            isInOn = False
            if surfCont['receptacleObjectIds'] is not None:
                isInOn = any(target.lower().split('_')[0] in name.lower() for name in surfCont['receptacleObjectIds'])
            msg = "" if isInOn == value else f"<condition name={cond} target={target} recipient={recipient}/> is {isInOn}."
            return isInOn == value, msg

        if object_state and cond in object_state:
            if cond.lower() in "isopen" and not object_state['openable']:
                return True , ''
            cond_objs = []
            msg = ''
            if not object_state[cond]:
                cond_objs = [obj['name'] for obj in current_state['objects'] if obj[cond]]
                msg = f"These objects satisfy the condition {cond}: f{cond_objs}"
            return object_state[cond] == value, msg

        return False, (f'I failed to check <condition name={cond} target={target}/> '
                       f'because the object state is unknown. Try search actions like '
                       f'<action name=scanroom target={target}/> or look for {target} inside recepticals.')

    def execute_actions(self, actions, memory):
        # episode_id = memory.current_episode

        for action in actions:
            if 'walk_to_room' in action.lower():
                result = (True, "")
                continue
            current_state = self.get_state()
            print(action)

            act, target = parse_instantiated_predicate(action)

            if type(target) == list:
                target = target[0]
            if target.lower() not in self.object_names:
                return False, f"There is no object named {target} in this environemnent to execute action {act}"
            if "walk" in action.lower() and "sinkbasin" in target.lower():
                obj = [ob for ob in self.get_graph()['objects'] if "sink_" in ob['name'].lower()]
                if len(obj) == 0:
                    return False, f"There is no object named {target} in this env to execute action.".format(target)
                evt = self.navigate_to_object(object=obj[0])

                return evt.metadata['lastActionSuccess'], evt.metadata["errorMessage"]

            print(target)
            print(f"target type {type(target)}")
            if target is None or target == 'None':
                result = self.action_fn_from_str[act]()
                continue

            if target not in self.room_names and not any(self.check_same_obj(target.lower(), obj.lower()) for obj in self.object_names) and target != 'None':
                print(self.object_names)
                return False, "There is no object named {} in this env to execute action.".format(target)

            if 'scanroom' in act:
                return self.handle_scan_room(target, memory)
            # Bulk update all objects in the current state
            # objects_to_update = [(obj['name'], obj) for obj in current_state['objects']]
            # memory.store_multiple_object_states(objects_to_update, episode_id, self.scene_id)

            if target in current_state['room_names'] or target.lower() in self.room_names:
                ret = self.action_fn_from_str[act](target)
                return True, ''

            # Find specific target object
            object_state = next((obj for obj in current_state['objects'] if target.lower() in obj['name'].lower()), None)
            if object_state is None:
                return False, (f"Could not find object named {target} try search actions like "
                               f"<action name='scanroom' target={target}/> or look for {target} inside recepticals.>.")
            print(f"Action: {act}, Target: {target}")
            print(object_state)
            if target is None:
                result = self.action_fn_from_str[act]()
                continue

            try:
                result = self.action_fn_from_str[act](object_state)
            except Exception as e:
                print(object_state)
                if 'write to closed file' in f"{e}".lower():
                    self.reset(self.scene_index)
                print(f"Action <action name={act} target={target}> failed due to: {e}")
                return False, f"Failed to execute action {act}: {e}."

            if not result or not result.metadata.get("lastActionSuccess", False):
                error_message = result.metadata["errorMessage"]
                if len(error_message) > 50:
                    error_message = error_message[:error_message.find('trace:')]
                return False, f"Action <action name={act} target={target}> failed due to: {error_message}"
        self.object_names = self.get_object_names()
        return True, "Actions executed successfully."

    def get_world_predicate_set(self, graph, custom_preds=()):
        return set(get_predicates(graph['objects']))

    def check_obj_contained(self, obj, receptacle):
        state = self.get_graph()
        receptacle = [object for object in state["objects"] if object["objectId"] == receptacle][0]
        print(obj)
        print(receptacle)
        print(obj in receptacle['receptacleObjectIds'])
        return obj in receptacle['receptacleObjectIds']

    def failure_msg_to_blocking_mode(self, msg, action):
        act, params = parse_instantiated_predicate(action)
        if msg == NO_VALID_PUT or msg == PUT_COLLISION:
            return f"no-placement {params[1]}"
        return None


    def check_satisfied(self, sub_goal, memory):
        state = self.get_graph()
        to_remove = []
        success = False
        if sub_goal is None:
            raise Exception('No goal set.')
            return True, ''
        print(sub_goal)
        sub_goal = sub_goal.split(" ")
        if len(sub_goal) <= 1:
            return False, f"I don't know how to verify this subgoal: {sub_goal}"
        value = int(sub_goal[-1])
        pred = sub_goal[0]
        params = sub_goal[1:-1]
        target = params[0]
        recipient = params[1] if len(params) > 1 else None
        return self.check_condition(pred, target=target, recipient=recipient, memory=memory, value=value)
        print(f"pred, params, value: {pred},{params},{value}")
        if "WASHED" in pred:
            if "SINK" in pred:
                faucet_str = params[2]
                sink_str = params[1]
                item_str = params[0]
                faucet = [obj for obj in state['objects'] if obj['name'] == faucet_str]
                sink = [obj for obj in state['objects'] if obj['name'] == sink_str]

                if len(faucet) > 0 and len(sink)>0:
                    faucet = faucet[0]
                    sink = sink[0]

                    if faucet['isToggled'] and item_str in sink['receptacleObjectIds']:
                        self.clean_object({"objectId": item_str})
            param = params[0] if "character" not in params[0] else params[1]
            obj = [object for object in state["objects"] if object["objectId"] == param][0]
            success = not obj["isDirty"]
            print(f"check sat: {obj}")
        elif "INSIDE" in sub_goal or "IN" in sub_goal or ("ON" in sub_goal and len(params) == 2):
            success = self.check_obj_contained(params[0], params[1])
        elif "ACTIVE" in sub_goal:
            param = sub_goal.split()[1]
            obj= [object for object in state["objects"] if object["objectId"] == param][0]
            success  = obj['isToggled']
        # elif "FILL" in sub_goal:
        #     objs = [obj for obj in state["objects"] if obj["fillLiquid"] in params]
        #     success = len(objs) > 0
        elif "inside" == pred:
            container = params[1]
            obj = [object for object in state["objects"] if container.lower() in object["objectId"].lower() and object['receptacleObjectIds'] and any(params[0].lower() in id.lower() for id in object['receptacleObjectIds'])]
            if len(obj) == 0:
                return False, to_remove
            return True, to_remove
        else:
            # vhome_to_aithor = get_vhome_to_thor_dict()

            key = pred#vhome_to_aithor.get(pred, None)
            if key not in AI2THOR_PREDICATES:
                print(f"failed to find predicate: {pred} ")
                return False, to_remove
            param = params[0] if "character" not in params[0] else params[1]
            obj = [object for object in state["objects"] if self.check_same_obj(param, object['name'])]
            if len(obj) == 0:
                return False, to_remove
            obj = obj[0]
            if pred.lower() in "isopen" and not obj['openable']:
                return True, to_remove
            success = obj[key]
            print(obj[key])
            print(param)
            print(key)
            print(obj)
        if success:
            to_remove.append(sub_goal)
        return success, to_remove

    def goal_key_to_vhome_goal(self, goal_key):
        state = self.get_graph()
        goal = {}
        self.vhome_obj_ids
        if goal_key == "apple":
            fridge = [obj for obj in state['objects'] if 'fridge' in obj['name'].lower()][0]

            goal = {f'inside_Apple_{self.vhome_obj_ids[fridge["name"]]}': 1}
        elif goal_key == "coffee":
            mug = [obj for obj in state['objects'] if 'mug' in obj['name'].lower()][0]
            table = [obj for obj in state['objects'] if 'diningtable' in obj['name'].lower()][0]

            goal = {f'on_Mug_{self.vhome_obj_ids[table["name"]]}': 1, f'inside_coffee_{self.vhome_obj_ids[mug["name"]]}': 1}
        elif goal_key == "set_place":
            table = [obj for obj in state['objects'] if 'diningtable' in obj['name'].lower()][0]
            goal = {f'on_Plate_{self.vhome_obj_ids[table["name"]]}': 1, f'on_Fork_{self.vhome_obj_ids[table["name"]]}': 1,
                    f'on_Knife_{self.vhome_obj_ids[table["name"]]}': 1}
        elif goal_key == "clean_mug":
            sink = [obj for obj in state['objects'] if 'sinkbasin' in obj['name'].lower()][0]
            mug = [obj for obj in state['objects'] if 'mug' in obj['name'].lower()][0]
            faucet = [obj for obj in state['objects'] if 'faucet' in obj['name'].lower()][0]
            goal = {f'on_Mug_{self.vhome_obj_ids[sink["name"]]}': 1, f'turnon_{self.vhome_obj_ids[faucet["name"]]}': 1}
        return goal

    def get_vhome_obj_ids(self, state):
        obj_vals = list(self.vhome_obj_ids.values())
        if len(obj_vals) > 0:
            return self.vhome_obj_ids
        cur_id = 2 if len(obj_vals) == 0 else max(obj_vals) + 1
        for obj in state['objects']:
            if obj['name'] not in self.vhome_obj_ids:
                self.vhome_obj_ids[obj['name']] = cur_id
                cur_id += 1
        return self.vhome_obj_ids

    def check_goal(self, goal):
        state = self.get_graph()
        if goal == 'put_food':
            foods = [item['objectId'] for item in state['objects'] if item['salientMaterials'] is not None and'Food' in item['salientMaterials']]
            fridge = [item for item in state['objects'] if 'fridge' in item['name'].lower()][0]
            return all(food in fridge['receptacleObjectIds'] for food in foods)
        if goal == 'coffee':
            success = any([obj['fillLiquid'] == 'coffee' for obj in state["objects"] if 'mug' in obj['name'].lower()])
            if success:
                cup = [obj for obj in state['objects'] if 'mug' in obj['name'].lower() and obj['fillLiquid'] == 'coffee']
                success = success and len(cup) > 0 and any(any('table' in parent.lower() for parent in obj['parentReceptacles']) for obj in cup if obj['parentReceptacles'] is not None)
            return success
        if goal == 'water_cup':
            return any([obj['fillLiquid'] == 'water' for obj in state["objects"] if 'cup' in obj['name'].lower()])
        if goal == 'apple':
            fridge = [obj for obj in state['objects'] if 'fridge' in obj['name'].lower()][0]
            return any("apple" in obj.lower() for obj in fridge['receptacleObjectIds'] if
                       fridge['receptacleObjectIds'] is not None)
        if goal == 'set_place':
            objects = ['fork', 'knife', 'plate']
            tables = [obj for obj in state["objects"] if 'table' in obj['name'].lower()]
            success = True
            for obj in objects:
               success = success and any(obj in rec.lower() for table in tables for rec in table['receptacleObjectIds'] if table['receptacleObjectIds'] is not None)
            return success
        if goal == 'clean_mug':
            sinkbasin = [obj for obj in state["objects"] if 'sinkbasin' in obj['name'].lower()][0]
            print(sinkbasin)
            faucet = [obj for obj in state["objects"] if 'faucet' in obj['name'].lower()][0]
            success = sinkbasin['receptacleObjectIds'] is not None and any('mug' in obj.lower() for obj in sinkbasin['receptacleObjectIds'])
            success = success and faucet['isToggled']
            return success

        return False

    def validate_goal(self, goal):
        sub_goal = goal['conditions'][0]
        sub_goal = sub_goal.split(" ")
        value = int(sub_goal[-1])
        pred = sub_goal[0]
        params = sub_goal[1:-1]
        target = params[0]
        if pred not in AI2THOR_PREDICATES or target not in self.object_names:
            # raise "Invalid goal"
            return False
        return True

if __name__ == '__main__':
    sim = AI2ThorSimEnv()
    print(sim.check_goal('apple'))