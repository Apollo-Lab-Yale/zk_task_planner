import csv
from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
from cognitive_bt_framework.src.cbt_planner.cbtf import CognitiveBehaviorTreeFramework
import time
import numpy as np
from datetime import datetime
import json
import gzip
import threading
import base64


def encode_bytes(data):
    """Recursively encode bytes objects in a dictionary to base64 strings."""
    if isinstance(data, dict):
        return {k: encode_bytes(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [encode_bytes(item) for item in data]
    elif isinstance(data, bytes):
        return base64.b64encode(data).decode('utf-8')
    else:
        return data


def save_compressed_json(data, filename):
    try:
        data = encode_bytes(data)  # Encode bytes before saving
        with gzip.open(filename, 'ab') as f:  # Open in binary append mode
            json_data = json.dumps(data).encode('utf-8')
            f.write(json_data)
            f.write(b'\n')  # Add newline to separate JSON entries
    except Exception as e:
        print(e)


def load_compressed_json(filename):
    with gzip.open(filename, 'rb') as f:
        return [json.loads(line) for line in f]


goals = [ 'coffee',  'apple', 'clean_mug','set_place',  ]

goals_nl = {
    'coffee': 'bring a mug of coffee to the table',
    'set_place': 'set a place at the table',
    'clean_mug': 'soak the dirty mug',
    'apple': 'put the apple in the fridge'
}
ablate = True
data_dir = '/home/liam/dev/cognitive_bt_framework/cognitive_bt_framework/data'
ts = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
output_csv = f'{data_dir}/run_data_{"ablated_" if ablate else ""}{ts}.csv'
output_json_gz = f'{data_dir}/run_data_{"ablated_" if ablate else ""}{ts}.json.gz'


def parse_subplans(subplans):
    parsed_subplans = {}
    for subplan in subplans:
        parsed_subplans[subplan] = {'FailedPlans': subplans[subplan].failed_plans,
                                    'SuccessfulPlans': subplans[subplan].successful_plans,
                                    'SuccessCondition': subplans[subplan].condition,
                                    'ChildrenTasks': subplans[subplan].children}
    return parsed_subplans


def write_json_async(data, filename):
    thread = threading.Thread(target=save_compressed_json, args=(data, filename))
    thread.start()
    return thread


def main():
    sim = AI2ThorSimEnv(scene_index=28)
    cbtf = CognitiveBehaviorTreeFramework(sim, ablate)
    trials = 25
    scenes = [28, 27]

    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Goal', 'Scene', 'Trial', 'Success', 'ExecTime'])

    # Run simulations and log data
    for goal in goals:
        cbtf.set_goal(goal)
        for trial in range(trials):
            for scene in scenes:
                sim.reset(scene_index=scene)
                # if goal == 'wash_mug':
                #     mug =
                cbtf.set_goal(goal)
                start = time.time()
                success = cbtf.manage_task_ordered(goals_nl[goal])
                runtime = time.time() - start
                sub_plans = parse_subplans(cbtf.subgoals)
                final_state = sim.get_state()
                # Write data to CSV
                with open(output_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([goal, scene, trial, success, runtime])

                # Collect data for JSON
                data = {
                    'Goal': goal,
                    'Scene': scene,
                    'Trial': trial,
                    'Success': success,
                    'ExecTime': runtime,
                    'Subgoals': sub_plans,
                    'FinalState': final_state
                }

                # Append data to JSON in a separate thread
                write_json_async(data, output_json_gz)


if __name__ == "__main__":
    main()
