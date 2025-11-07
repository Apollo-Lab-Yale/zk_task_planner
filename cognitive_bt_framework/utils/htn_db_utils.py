import sqlite3

def setup_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create Tasks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Tasks (
        TaskID INTEGER PRIMARY KEY,
        TaskName TEXT NOT NULL,
        InitialDescription TEXT
    )''')

    # Create Decompositions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Decompositions (
        TaskID INTEGER NOT NULL,
        DecompositionText TEXT,
        CreationDate TEXT,
        CreatedBy TEXT,
        FOREIGN KEY (TaskID) REFERENCES Tasks (TaskID)
    )''')

    # Create Feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Feedback (
        FeedbackID INTEGER PRIMARY KEY,
        DecompositionID INTEGER NOT NULL,
        UserID TEXT,
        FeedbackText TEXT,
        FeedbackDate TEXT,
        FOREIGN KEY (DecompositionID) REFERENCES Decompositions (DecompositionID)
    )''')

    # Create DecompositionAdjustments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS DecompositionAdjustments (
        AdjustmentID INTEGER PRIMARY KEY,
        DecompositionID INTEGER NOT NULL,
        PreviousDecompositionText TEXT,
        NewDecompositionText TEXT,
        AdjustmentReason TEXT,
        AdjustmentDate TEXT,
        FOREIGN KEY (DecompositionID) REFERENCES Decompositions (DecompositionID)
    )''')

    conn.commit()
    conn.close()

def add_task_and_decomposition(task_name, initial_description, decomposition_text, created_by):
    conn = sqlite3.connect('task_decomposition.db')
    cursor = conn.cursor()

    # Insert the task
    cursor.execute("INSERT INTO Tasks (TaskName, InitialDescription) VALUES (?, ?)", (task_name, initial_description))
    task_id = cursor.lastrowid

    # Insert the initial decomposition
    cursor.execute("INSERT INTO Decompositions (TaskID, DecompositionText, CreationDate, CreatedBy) VALUES (?, ?, datetime('now'), ?)", (task_id, decomposition_text, created_by))

    conn.commit()
    conn.close()

def store_feedback(decomposition_id, user_id, feedback_text):
    conn = sqlite3.connect('task_decomposition.db')
    cursor = conn.cursor()

    cursor.execute("INSERT INTO Feedback (DecompositionID, UserID, FeedbackText, FeedbackDate) VALUES (?, ?, ?, datetime('now'))", (decomposition_id, user_id, feedback_text))

    conn.commit()
    conn.close()

def get_decompositions_with_feedback(task_id):
    conn = sqlite3.connect('task_decomposition.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT d.DecompositionText, f.FeedbackText
    FROM Decompositions d
    LEFT JOIN Feedback f ON d.DecompositionID = f.DecompositionID
    WHERE d.TaskID = ?
    ''', (task_id,))

    for row in cursor.fetchall():
        print("Decomposition:", row[0])
        print("Feedback:", row[1] if row[1] else "No feedback yet", "\n")

    conn.close()
