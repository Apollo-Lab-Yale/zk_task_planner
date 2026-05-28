import openai
import re

def set_api_key(api_key):
    """
        Initializes the OpenAI API with the provided API key.
        :param api_key: The API key for OpenAI.
    """
    openai.api_key = api_key

def get_openai_key():
    file_path = "/usr/config/llm_task_planning.txt"
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return first_line.strip()
    except:
        raise "failed to find api key."

def get_claude_key():
    file_path = "/usr/config/claud_api_key.txt"
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return first_line.strip()
    except:
        raise "failed to find api key."

def get_hf_key():
    file_path = "/usr/config/hf_keys.txt"
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return first_line.strip()
    except:
        raise "failed to find api key."

def setup_openai():
    file_path = "/usr/config/llm_task_planning.txt"
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            set_api_key(first_line.strip())
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
    except IOError as e:
        print(f"An error occurred while reading the file: {str(e)}")

def parse_llm_response(response):
    """
    Parse the LLM response to extract subtasks and their completion conditions.
    :param response: The response from the LLM as a string.
    :return: A dictionary with subtasks as keys and lists of completion conditions as values.
    """
    lines = response.split('\n')
    tasks = {}
    current_task = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match the subtask line with underscores between words
        subtask_match = re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', line)
        # Match the condition line (starts with a dash)
        condition_match = re.match(r'^-\s*(.+)$', line)

        if subtask_match:
            current_task = line
            tasks[current_task] = []
        elif condition_match and current_task:
            condition = condition_match.group(1).strip()
            tasks[current_task].append(condition)

    return tasks

# def parse_llm_response(response):
#     """
#     Parse the LLM response to extract subtasks and their completion conditions.
#     :param response: The response from the LLM as a string.
#     :return: A dictionary with subtasks as keys and lists of completion conditions as values.
#     """
#     lines = response.split('\n')
#     tasks = {}
#     current_task = None
#
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         # Match the subtask line
#         subtask_match = re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', line)
#         # Match the condition line (starts with a dash)
#         condition_match = re.match(r'^-\s*(.+)$', line)
#
#         if subtask_match:
#             current_task = line
#             tasks[current_task] = []
#         elif condition_match and current_task:
#             condition = condition_match.group(1).strip()
#             tasks[current_task].append(condition)
#
#     return tasks


def parse_llm_response_ordered(response):
    """
    Parse the LLM response to extract subtasks and their completion conditions.
    :param response: The response from the LLM as a string.
    :return: A dictionary with subtasks as keys and lists of completion conditions as values.
    """
    lines = response.split('\n')
    tasks = {}
    current_task = None
    task_stack = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match the subtask line with underscores between words
        subtask_match = re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', line)
        # Match the condition line (starts with a dash)
        condition_match = re.match(r'^-\s*(.+)$', line)

        if subtask_match:
            current_task = line
            tasks[current_task] = {'conditions': [], 'subtasks': []}
            if task_stack:
                tasks[task_stack[-1]]['subtasks'].append(current_task)
            task_stack.append(current_task)
        elif condition_match and current_task:
            condition = condition_match.group(1).strip()
            tasks[current_task]['conditions'].append(condition)

        # Check for end of current task
        if not subtask_match and not condition_match and task_stack:
            task_stack.pop()

    return tasks


import re


def parse_task_decomposition_and_context_with_subtask_context(output):
    """
    Parse the task decomposition, subtask-specific context, and overall context from the GPT model's output.
    :param output: The raw output string from the GPT model.
    :return: A dictionary with parsed task decomposition, subtask-specific context, and overall context.
    """

    # Split the output into lines
    lines = output.strip().split("\n")

    # Regular expressions to match subtasks, conditions, and subtask-specific context
    subtask_pattern = re.compile(r'^-\s*([a-zA-Z0-9_]+)$')  # Subtasks start with '- ' followed by subtask name
    condition_pattern = re.compile(r'^-\s*(?!Context:)(.+)$')  # Conditions start with '- ' and not 'Context:'
    subtask_context_pattern = re.compile(r'^-\s*Context:\s*(.+)$')  # Subtask-specific context starts with '- Context:'

    task_decomposition = {}
    current_subtask = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Check for subtask
        subtask_match = subtask_pattern.match(line)
        if subtask_match:
            current_subtask = subtask_match.group(1)
            task_decomposition[current_subtask] = {"conditions": [], "subtask_context": ""}
            continue

        # Check for subtask-specific context
        subtask_context_match = subtask_context_pattern.match(line)
        if subtask_context_match and current_subtask:
            subtask_context = subtask_context_match.group(1).strip()
            task_decomposition[current_subtask]["subtask_context"] = subtask_context
            continue

        # Check for condition
        condition_match = condition_pattern.match(line)
        if condition_match and current_subtask:
            condition = condition_match.group(1).strip()
            task_decomposition[current_subtask]["conditions"].append(condition)
            continue

    parsed_output = {
        "task_decomposition": task_decomposition,
        "context": ""  # Overall context is not part of the provided output, so it's left empty
    }

    return parsed_output