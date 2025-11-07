import json
import datetime
import sqlite3
from cachetools import LRUCache

class Memory:
    def __init__(self, db_path):
        self.db_path = db_path
        self.object_cache = LRUCache(maxsize=100)  # Cache for object states
        # self.setup_database()
        self.current_episode = -1

    # def setup_database(self):
    #     conn = sqlite3.connect(self.db_path)
    #     cursor = conn.cursor()
    #     cursor.execute('''
    #         CREATE TABLE IF NOT EXISTS Episodes (
    #             EpisodeID TEXT PRIMARY KEY,
    #             TaskName TEXT
    #         );
    #     ''')
    #     cursor.execute('''
    #         CREATE TABLE IF NOT EXISTS ObjectStates (
    #             StateID INTEGER PRIMARY KEY AUTOINCREMENT,
    #             ObjectID TEXT,
    #             State TEXT,
    #             EpisodeID TEXT,
    #             Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    #             FOREIGN KEY (EpisodeID) REFERENCES Episodes(EpisodeID)
    #         );
    #     ''')
    #     conn.commit()
    #     conn.close()

    def start_new_episode(self, task_name):
        episode_id = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Episodes (EpisodeID, TaskName) VALUES (?, ?)", (episode_id, task_name))
        conn.commit()
        conn.close()
        self.current_episode = episode_id
        return episode_id

    def store_object_state(self, object_id, state, episode_id):
        timestamp = datetime.datetime.now().isoformat()
        # Store in cache
        self.object_cache[object_id] = (json.dumps(state), timestamp)
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ObjectStates (ObjectID, State, EpisodeID, Timestamp) VALUES (?, ?, ?, ?)",
            (object_id, json.dumps(state), episode_id, timestamp)
        )
        conn.commit()
        conn.close()

    def retrieve_object_state(self, object):
        # Check cache first
        for object_id in self.object_cache.keys():
            if object in object_id.lower():
                state, timestamp = self.object_cache[object_id]
                return json.loads(state), timestamp
        # Fall back to database if not in cache
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        object_id = object[0].upper() + object[1:]
        try:
            # Be explicit about the object_id by finding the closest match first
            cursor.execute("SELECT ObjectID FROM ObjectStates WHERE ObjectID LIKE ? ORDER BY Timestamp DESC LIMIT 1",
                           (f'%{object_id}%',))
            result = cursor.fetchone()
            if result:
                object_id = result[0]
                cursor.execute(
                    "SELECT State, Timestamp FROM ObjectStates WHERE ObjectID = ? ORDER BY Timestamp DESC LIMIT 1",
                    (object_id,)
                )
                result = cursor.fetchone()
                if result:
                    state, timestamp = result
                    # Update the cache with the new data
                    self.object_cache[object_id] = (state, timestamp)
                    return json.loads(state), timestamp
        except Exception as e:
            print(e)
            conn.close()

        return None, None

    def store_multiple_object_states(self, object_states, episode_id, env_id):
        # Convert state dictionaries to JSON strings within the tuple list
        object_states_data = []
        timestamp = datetime.datetime.now().isoformat()
        for obj_id, state in object_states:
            state_json = json.dumps(state)
            self.object_cache[obj_id] = (state_json, timestamp)
            object_states_data.append((obj_id, state_json, episode_id, timestamp, env_id))
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO ObjectStates (ObjectID, State, EpisodeID, Timestamp, EnvID) VALUES (?, ?, ?, ?, ?)",
            object_states_data
        )
        conn.commit()
        conn.close()
