import sqlite3
import os
import numpy as np
from cachetools import LRUCache
# from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch
import json
import datetime
from sentence_transformers import SentenceTransformer, util


from cognitive_bt_framework.src.llm_interface.llm_interface_claude import LLMInterfaceClaude
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI
from cognitive_bt_framework.utils.db_utils import setup_database, add_behavior_tree, insert_subtasks,  \
    create_subtask_table, get_subtasks_and_embeddings
from cognitive_bt_framework.utils.bt_utils import parse_node, parse_bt_xml, BASE_EXAMPLE, FORMAT_EXAMPLE
from cognitive_bt_framework.utils.goal_gen_aithor import get_wash_mug_in_sink_goal, get_make_coffee, get_put_apple_in_fridge_goal
from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_ACTIONS, AI2THOR_PREDICATES, AI2THOR_ACTIONS_ANNOTATED
from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
from cognitive_bt_framework.src.cbt_planner.memory import Memory
from cognitive_bt_framework.utils.logic_utils import cosine_similarity, stop_words
from cognitive_bt_framework.src.cbt_planner.sub_task import SubTask

OPENAI_MODEL = 'gpt-4o'
CLAUDE_MODEL = 'claude-3-5-sonnet-20240620'

def preprocess_text(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

DEFAULT_DB_PATH = os.path.join(os.environ.get('PROJECT_ROOT', '.'), 'cognitive_bt_framework/src/')

class CognitiveBehaviorTreeFramework:
    def __init__(self, robot_interface, ablate=False, actions=AI2THOR_ACTIONS_ANNOTATED , conditions=AI2THOR_PREDICATES, db_path=DEFAULT_DB_PATH, model_name= OPENAI_MODEL, sim=True):
        self.robot_interface = robot_interface
        self.db_path = db_path
        self.ablate = ablate
        if 'claude' in model_name.lower():
            self.llm_interface = LLMInterfaceClaude(model_name)
        if 'gpt' in model_name.lower():
            self.llm_interface = LLMInterfaceOpenAI(model_name)
        self.bt_cache = LRUCache(maxsize=100)
        self.memory = Memory(db_path + f'behavior_tree.db')
        self.db_path += f'behavior_tree.db'
        self.actions = actions
        setup_database(self.db_path)
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # self.model = RobertaModel.from_pretrained('roberta-base')
        self.keyword_embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.object_names = set(robot_interface.get_object_names())
        print(self.object_names)
        self.cosine_similarity_threshold = 1  # Cosine similarity threshold
        self.example = BASE_EXAMPLE
        self.conditions = conditions
        self.known_objects = []
        self.goal = None
        self.max_actions = 3
        self.max_goal_retries = 5
        self.num_refinement_retires = 1


    def get_embedding(self, text):
        if '_' in text:
            text = text.replace('_', " ")
        print(f"Getting embedding for: {text}")
        # pp_text = preprocess_text(text)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tobytes()

    def get_keywords(self, text, top_n=10):
        keywords = self.keyword_model.extract_keywords(text, top_n=top_n)
        return [keyword for keyword, score in keywords]


    def get_keyword_similarity(self, keywords1, keywords2):
        embeddings1 = self.keyword_embedder.encode(keywords1, convert_to_tensor=True)
        embeddings2 = self.keyword_embedder.encode(keywords2, convert_to_tensor=True)
        # Compute cosine similarity between embeddings
        cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_similarities

    def find_most_similar_task_decomp(self, embedding, task_name):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TaskID, TaskName, IsSubtask, CompleteCondition FROM Tasks")
            best_match = None
            highest_similarity = self.cosine_similarity_threshold
            for row in cursor.fetchall():
                try:
                    existing_task_name = row[1]
                    existing_embedding = row[0]
                    complete_condition = row[3]
                    print(type(existing_embedding))
                    a_float32 = np.frombuffer(embedding, dtype=np.float32).tobytes()
                    b_float32 = np.frombuffer(existing_embedding, dtype=np.float32).tobytes()
                    similarity = cosine_similarity(a_float32, b_float32)
                    print(existing_task_name, similarity)
                    if task_name == existing_task_name or similarity > highest_similarity:
                        if row[2]:
                            best_match = (existing_task_name, existing_embedding, complete_condition)
                        else:
                            with self.connect_db() as conn:
                                best_match = get_subtasks_and_embeddings(conn, existing_task_name)

                        highest_similarity = similarity
                except Exception as e:
                    print(e)
                    pass
            return best_match

    def find_most_similar_task(self, embedding, task_name):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TaskID, TaskName, SubTasks FROM Tasks")
            best_match = None
            highest_similarity = -1
            for row in cursor.fetchall():
                try:
                    existing_task_name = row[1]
                    existing_embedding = np.frombuffer(row[0], dtype=np.float32).tobytes()
                    embedding = np.frombuffer(embedding, dtype=np.float32).tobytes()
                    similarity = cosine_similarity(embedding, existing_embedding)
                    if task_name == existing_task_name or similarity > highest_similarity:
                        best_match = row[2]
                        highest_similarity = similarity
                except Exception as e:
                    pass
            return best_match

    def set_goal(self, goal):
        self.goal = goal
        self.object_names = set(self.robot_interface.get_object_names())
        self.subgoals = {}
        if self.robot_interface.save_video:
            self.robot_interface.image_saver.goal = goal

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def load_or_generate_bt(self, big_task_name, task_id, task_name, completed_subtasks, context, complete_condition):
        print(f'Attempting to create new BT for: {task_name}')
        bt_xml = self.load_behavior_tree(task_id)
        if bt_xml is None:
            bt_xml = self.llm_interface.get_behavior_tree(big_task_name, task_name, self.actions, self.conditions,
                                                          self.example, self.robot_interface.object_names, completed_subtasks,
                                                          context, complete_condition)
            # isValid = validate_bt(bt_xml)
            # if isValid != 'Valid':
            #     raise isValid
            self.save_behavior_tree(task_name, bt_xml, task_id)
        return parse_bt_xml(bt_xml , self.actions, self.conditions), bt_xml

    def load_behavior_tree(self, task_id):
        if task_id in self.bt_cache:
            return self.bt_cache[task_id]
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT BehaviorTreeXML FROM BehaviorTrees WHERE TaskID = ?", (task_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def save_behavior_tree(self, task_name, bt_xml, task_id):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            embedding = self.get_embedding(task_name)
            task_id = embedding
            self.bt_cache[task_id] = bt_xml
            add_behavior_tree(conn, task_id, task_name, "Generated BT", bt_xml, 'system')

    def execute_behavior_tree(self, bt_root, complete_condition):
        # Placeholder for behavior tree execution logic
        print(f"Attempting to achieve: {complete_condition}")
        ret = (True, "", bt_root.to_xml())
        prev_failure = None
        for i in range(self.max_actions):
            self.known_objects = set(self.robot_interface.get_object_names())
            if (self.robot_interface.check_satisfied(complete_condition[0], memory=self.memory)[0] or
                    self.robot_interface.check_goal(self.goal)):
                print(f'{complete_condition} is satisfied')
                return True, "", bt_root.to_xml()
            else:
                print(f'{complete_condition} is NOT satisfied')
            # self.update_known_objects()
            ret = bt_root.execute(self.robot_interface.get_state(), interface=self.robot_interface, memory=self.memory)
            if ret[0] == False:
                print("exiting")
                return ret
        return ret

    def simulate_feedback(self, msg):
        # Placeholder for feedback simulation
        return "Feedback based on monitoring"

    def update_known_objects(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ObjectId from ObjectStates")
            rows = cursor.fetchall()
            if rows is not None:
                obs = []
                for row in rows:
                    splt = row[0].split('|')
                    if splt[-1][0].isalpha():
                        obs.append(splt[-1])
                    else:
                        obs.append(splt[0])
                self.known_objects = set(obs)
            else:
                self.known_objects = set([obj.split('|')[0] if not obj.split('|')[-1].isalpha() else obj.split('|')[-1] for obj in self.memory.object_cache])

    def update_bt(self, task_name, task_id, bt_xml):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            if task_id is None:
                embedding = self.get_embedding(task_name)
                task_id = embedding
            sql = "INSERT OR REPLACE INTO BehaviorTrees (TaskName, TaskID, BehaviorTreeXML) VALUES (?, ?, ?)"
            # Execute the SQL command
            cursor.execute(sql, (task_name, task_id, bt_xml))
            conn.commit()
            self.bt_cache[task_id] = bt_xml

    def refine_and_update_bt(self, big_task, task_name, task_id, bt_xml, feedback, context, complete_condition,
                             completed_subtasks, image_context=None):
        print(f'Attempting to refine BT for {task_name}')
        refined_bt_xml = self.llm_interface.refine_behavior_tree(big_task, task_name, self.actions, self.conditions, bt_xml,
                                                                 feedback, self.robot_interface.object_names, completed_subtasks,
                                                                 self.example, context, complete_condition, image_context)
        if refined_bt_xml:
            # isValid = validate_bt(refined_bt_xml)
            # if isValid != 'Valid':
            #     raise Exception(f"Behavior tree is not valid: {isValid}")
            self.update_bt(task_name, task_id, refined_bt_xml)
            return parse_bt_xml(refined_bt_xml, self.actions, self.conditions)
            print("Behavior Tree refined and updated.")
        else:
            print("Unable to refine behavior tree based on feedback.")

    def generate_decomposition(self, task_name, task_embedding, context):
        subtask_dict = self.llm_interface.get_task_decomposition(task_name, self.robot_interface.object_names, context)
        decomp = list(subtask_dict.keys())
        print(decomp)
        task_name, decomp = decomp[0], decomp[1:]
        print(decomp)
        print(f'\n\n {task_name}')

        decomp_embeddings = [self.get_embedding(subtask) for subtask in decomp]
        with self.connect_db() as conn:
            create_subtask_table(conn, task_name)
            print(list(subtask_dict.values()))
            complete_conditions = [val[0] for val in list(subtask_dict.values())[1:]]
            print(f"Complete conditions: {complete_conditions}")
            insert_subtasks(conn, task_name, decomp, decomp_embeddings, complete_conditions)
            cursor = conn.cursor()
            cursor.execute("INSERT or REPLACE INTO Tasks (TaskName, TaskID, IsSubtask, CompleteCondition) VALUES (?, ?, FALSE, ?)", (task_name, task_embedding, self.goal))
            for i in range(len(decomp)):
                cursor.execute("INSERT or REPLACE INTO Tasks (TaskName, TaskID, IsSubtask, CompleteCondition) VALUES (?, ?, TRUE, ?)",
                               (decomp[i], decomp_embeddings[i], complete_conditions[i]))
        return decomp, decomp_embeddings, complete_conditions

    def manage_task_ordered(self, task_name):
        episode_id = self.memory.start_new_episode(task_name)
        itter = 0
        print('starting task')
        while not self.robot_interface.check_goal(self.goal) and itter < self.max_goal_retries:
            itter += 1
            print('task loop')
            # try:
            context, states = self.robot_interface.get_context(1)
            decomposition, context = (
                self.llm_interface.get_task_decomposition_ordered_context(task_name,
                                                                            self.robot_interface.object_names,
                                                                            context))
            print(decomposition)

            for subtask_name, details in decomposition.items():
                sub_complete = False
                subtask_conditions = details['conditions']
                print(f"DETAILS {details}")
                if not self.robot_interface.validate_goal(details):
                    raise Exception("Object in goal condition is not in known objects")
            print(decomposition)
            # except Exception as e:
            #     print(f"Task Decomposition failed: {e}")
            #     continue

            def execute_subtasks(subtasks, completed_subtasks):
                print(subtasks)
                for subtask_name, details in subtasks.items():
                    subtask_conditions = details['conditions']
                    subtask_subtasks = details['subtasks']
                    if subtask_name not in self.subgoals:
                        self.subgoals[subtask_name] = SubTask(name=subtask_name, condition=subtask_conditions)
                        self.subgoals[subtask_name].children = subtask_subtasks
                    # Skip the subtask if its condition is already satisfied
                    if self.robot_interface.check_satisfied(subtask_conditions[0], memory=self.memory)[0]:
                        print(f"Subtask {subtask_name} already completed.")
                        completed_subtasks.append(subtask_name)
                        continue

                    if not self.robot_interface.validate_goal(details):
                        return False
                    sub_itter = 0
                    while (not self.robot_interface.check_satisfied(subtask_conditions[0], memory=self.memory)[0] and not
                           self.robot_interface.check_goal(self.goal)):
                        sub_itter += 1
                        if self.ablate and sub_itter > self.max_goal_retries:
                            break
                        if self.robot_interface.check_goal(self.goal):
                            print('Success!')
                            return True
                        plan = None
                        try:
                            bt_root, bt_xml = self.load_or_generate_bt(
                                task_name, subtask_name, subtask_name, completed_subtasks, context, subtask_conditions
                            )
                            success, msg, subtree_xml = self.execute_behavior_tree(bt_root, subtask_conditions)
                            plan = bt_xml
                            if self.robot_interface.check_satisfied(subtask_conditions[0], memory=self.memory)[0]:
                                completed_subtasks.append(subtask_name)
                                if plan not in self.subgoals[subtask_name].successful_plans:
                                    self.subgoals[subtask_name].successful_plans.append(plan)
                                break
                            else:
                                if plan not in self.subgoals[subtask_name].failed_plans:
                                    self.subgoals[subtask_name].failed_plans.append(plan)
                        except Exception as e:
                            print(f"Failed to execute or parse behavior tree due to {e}.")
                            subtree_xml = "NO SUBTREE DUE TO FAILURE"
                            success = False
                            msg = str(e)
                        new_root = None
                        new_xml = None
                        if not self.ablate:
                            for _ in range(self.max_goal_retries):
                                try:
                                    context_img = None  # self.robot_interface.get_context(1)
                                    new_root = self.refine_and_update_bt(
                                        task_name, subtask_name, subtask_name, subtree_xml, msg, context, subtask_conditions,
                                        completed_subtasks, context_img
                                    )
                                    success, msg, subtree_xml = self.execute_behavior_tree(new_root, subtask_conditions)
                                    plan = new_root.to_xml()
                                    if self.robot_interface.check_satisfied(subtask_conditions[0], memory=self.memory)[0]:
                                        completed_subtasks.append(subtask_name)
                                        if plan not in self.subgoals[subtask_name].successful_plans:
                                            self.subgoals[subtask_name].successful_plans.append(plan)
                                        break
                                    else:
                                        if plan not in self.subgoals[subtask_name].failed_plans:
                                            self.subgoals[subtask_name].failed_plans.append(plan)
                                except Exception as e:
                                    print(f"Failed to execute behavior tree due to {e}.")

                                    success = False
                                    # msg = str(e)
                            break

                        if success and not \
                        self.robot_interface.check_satisfied(subtask_conditions[0], memory=self.memory)[0]:
                            msg = f"Execution of behavior tree ended in success but task {subtask_name} is NOT COMPLETED."
                        print(f"Failed to execute behavior tree due to {msg}.")
                        if new_root is not None:
                            plan = new_root.to_xml()
                            if success:
                                if plan not in self.subgoals[subtask_name].successful_plans:
                                    self.subgoals[subtask_name].successful_plans.append(plan)
                            else:
                                if plan not in self.subgoals[subtask_name].failed_plans:
                                    self.subgoals[subtask_name].failed_plans.append(plan)

                    # Recursively execute subtasks
                    if subtask_subtasks:
                        subtask_completed = False
                        for i in range(1):
                            sub_dict = {subtask: decomposition[subtask] for subtask in subtask_subtasks}
                            subtask_completed = execute_subtasks(
                                sub_dict, completed_subtasks)
                            if subtask_completed:
                                break
                        if not subtask_completed:
                            print(f"Subtask {subtask_name} failed. Moving to the next top-level subtask.")
                            break

                return completed_subtasks

            # ------------------------------------------------------------------------------------------------------------
            completed_subtasks = []
            try:
                completed_subtasks = execute_subtasks(decomposition, completed_subtasks)
                if not completed_subtasks:
                    print('Failed to complete any subtasks, moving to the next top-level task.')
                    continue
            except TimeoutError as e:
                print(f"Failed during subtask execution: {e}, moving to the next top-level task.")
                continue
            success = self.robot_interface.check_goal(self.goal)
            if success:
                print("Successsssssss")
            else:
                print("exceded maximum retries, task failed")
            self.robot_interface.end_sim()
            return success


if __name__ == "__main__":
    # 28, 24, 9
    # 28, 27,
    # no walk 19, 23,
    sim = RobosuiteSimEnv()
    sim.start()
    print(sim.get_state())
    # goal, _ = get_make_coffee(sim)
    cbtf = CognitiveBehaviorTreeFramework(sim)
    cbtf.set_goal('poor')
    # get_wash_mug_in_sink_goal(sim)
    # sim.image_saver.goal = "Set a place at the table."
    print(cbtf.manage_task_ordered("poor the bottle into the sink."))

