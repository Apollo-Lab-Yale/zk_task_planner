from cognitive_bt_framework.src.cbt_planner.cbtf import CognitiveBehaviorTreeFramework
from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
from cognitive_bt_framework.utils.logic_utils import cosine_similarity

import numpy as np

def test_embedding_retrieval(cbtf: CognitiveBehaviorTreeFramework):
    task_name = "fill a cup with water from the faucet"
    embedding = cbtf.get_embedding(task_name)

def test_embeddings(cbtf: CognitiveBehaviorTreeFramework):
    task_name = "fill a cup with water from the faucet"
    prev_embedding = cbtf.get_embedding(task_name)
    for i in range(50):
        new_embedding = cbtf.get_embedding(task_name)
        if new_embedding != prev_embedding:
            print(f'fail! {i}')
            return False
        prev_embedding = new_embedding
    print('success!')

def test_embedding_similarity(cbtf: CognitiveBehaviorTreeFramework):
    task_name1 = "get a glass of water"
    task_name2 = "get a glass of milk"
    embedding = np.frombuffer(cbtf.get_embedding(task_name1), dtype=np.float32)
    existing_embedding = np.frombuffer(cbtf.get_embedding(task_name2), dtype=np.float32)
    similarity = cosine_similarity(embedding, existing_embedding)
    print(similarity)

def test_keywords(cbtf: CognitiveBehaviorTreeFramework):
    task_name1 = "get a cup of water"
    task_name2 = "bring me some water"
    keywords1 = cbtf.get_keywords(task_name1)
    keywords2 = cbtf.get_keywords(task_name2)
    k = min(len(keywords1), len(keywords2))
    similarities = cbtf.get_keyword_similarity(keywords1, keywords2)
    print(similarities)
    print(keywords1, keywords2)
    embedding = np.frombuffer(cbtf.get_embedding(task_name1), dtype=np.float32)
    existing_embedding = np.frombuffer(cbtf.get_embedding(task_name2), dtype=np.float32)
    similarity = cosine_similarity(embedding, existing_embedding)
    print(similarity)

if __name__ == '__main__':
    sim = AI2ThorSimEnv()
    cbtf = CognitiveBehaviorTreeFramework(robot_interface=sim)
    test_keywords(cbtf)