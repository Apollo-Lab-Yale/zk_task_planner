import openai
from openai import OpenAI
from cognitive_bt_framework.utils import setup_openai, get_openai_key, parse_llm_response, parse_llm_response_ordered, parse_task_decomposition_and_context_with_subtask_context
from ratelimit import limits, sleep_and_retry
from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_PREDICATES_ANNOTATED
from typing import List, Dict, Optional, Tuple, Union

import json
import pprint
import cv2
import base64
import numpy as np
import time
import re

from cognitive_bt_framework.utils import BOOL_PREDS, RELATIONAL_PREDS

class LLMInterfaceOpenAI:
    def __init__(self, model_name="gpt-4o-turbo"):
        """
        Initialize the interface with your OpenAI API key and model choice.
        :param api_key: Your OpenAI API key.
        :param model_name: The model identifier for GPT-3 (e.g., "text-davinci-003").
        """
        self.client = OpenAI(api_key=get_openai_key())
        self.model_name = model_name
        self.conversation_history = []
        self.token_limit = 4000

    def add_to_history(self, role, content):
        """
        Add a new message to the conversation history, managing token count.
        """
        new_message = {"role": role, "content": content}
        self.conversation_history.append(new_message)
        # Keep the last 4096 tokens of the conversation history to stay within token limits
        tokens = sum(len(m['content']) for m in self.conversation_history)
        while tokens > self.token_limit:
            tokens -= len(self.conversation_history[0]['content'])
            self.conversation_history.pop(0)

    def generate_prompt_htn(self, task, known_objects, context):
        """
        Generate a prompt for the task decomposition.
        :param task: The task description.
        :return: A string prompt for the GPT model.
        """
        instruction = {"role": 'system', 'content': 'You are assisting in decomposing a high-level goal for a robot. '
                                                    'Each subgoal should be its own line with NO ADDITIONAL CHARACTERS. '
                                                    'The format should be short and underscore separated similar to a pythonic class name '
                                                    'with no spaces. Please provide the most complete decomposition '
                                                    'possible. If the task is sufficiently small,'
                                                    ' then you do not need to generate subtasks. Sufficiently reduced '
                                                    'subtasks include tasks like empty_trash, clear_counters, '
                                                    'empty_dishwasher, etc. do not reduce subtasks to single actions.'
                                                    'Additionally, for each subtask, provide a SINGLE object state condition '
                                                    'that defines when the subtask is considered complete. Format these conditions '
                                                    'as a bulleted list directly under the subtask, each condition on a new line '
                                                    'with a dash "-" at the beginning. All Conditions should come from'
                                                    f'the following list without exception: {AI2THOR_PREDICATES_ANNOTATED} followed'
                                                    f'by a space and the object that the condition applies to and a 1 '
                                                    f'or 0 indicating if the predicate should be true of false. All conditions'
                                                    f'must target a specific object and cannot contain placeholders.'
                                                    f' All objects in your decomposition should come from this list without exception: {known_objects}. '
                                                    f'Here is a brief description of the environment: {context}'}
        message = {"role": "user", "content": f"Decompose the following task into detailed subtask steps: {task}"}
        return [instruction, message]

    def generate_prompt_task_id(self, task, context, states):
        """
        Generate a prompt for the task decomposition.
        :param task: The task description.
        :return: A string prompt for the GPT model.
        """
        instruction = {"role": 'system', 'content': 'You are assisting in generating a task id for a user requested task'
                                                    ' The format should be short and underscore separated similar to a '
                                                    'pythonic class name  with no spaces. Additionally, using the images provided'
                                                    ' as task context, generate all relevant environmental'
                                                    ' information that may assist an llm in generating subtasks '
                                                    'and action sequences for the task'
                                                    f' {task} and behavior trees to complete those sub tasks be as detailed as '
                                                    'possible while ensuring that it is easy to comprehend for an LLM attempting to plan'#including detailed spacial relations and object descriptions'
                                                    '. The task ID and context should be separated by a new line '
                                                    'character. Additionally give hints on potential actions or insight into what '
                                                    'might be needed to complete the task. The description should be '
                                                    'long and easily readable. Include no other text!'}
        context_data = [{
                        "type": "text",
                        "text": "The following images and states are the context for the task to be completed"
                    }]
        for i in range(len(context)):
            image = context[i]
            # state = states[i]
            img_msg = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}"
                }
            }
            context_data.append(img_msg)
        context_msg = {
            "role": "user",
            "content": context_data
        }
        message = {"role": "user", "content": f"Generate an id and context, in the format described for this task: {task}"}

        return [instruction, context_msg, message]

    def generate_behavior_tree_prompt(self, big_task, task, actions, conditions, example, relevant_objects,
                                      completed_subtasks, context, complete_condition):
        """
        Generate a prompt specifically for creating a behavior tree in XML format.
        """
        completed_goals_str = f'''
        **Completed Sub Goals**
        The following subgoals have already been completed:
        {completed_subtasks}
        ''' if len(completed_subtasks) > 0 else ""
        '''
         **Domain**
                    the domain is defined in pddl as follows:
                    {DOMAIN_DEF}
        '''
        system_message = f'''
                    You are going to be tasked with creating a behavior tree in XML format for a robot to execute a specific task. Follow these rules carefully:
                    **Actions**: Use only the actions from this list:
                     {actions}. 
                     Other actions are not allowed.
                     each action tag should contain name and target members where name is the action being performed and target is the target of the action
                    **Conditions**: Use only the conditions from this list:
                    {conditions}.
                    No other conditions are allowed.
                    each condition tag should contain name, target, and value members. where name is the condition being checked and target is the target of the condition
                    and value indicates whether the condition should be true or false and should only be a value in [0, 1]
                    {completed_goals_str}
                   ---
                    **Behavior Tree Structure**:
                    - Use <Sequence> tags to execute all child actions or conditions in order until one fails or returns False.
                    - Use <Selector> tags to execute each child action or condition in order until one succeeds or returns True.
                    
                   **Object Classes**: You can only interact with objects from this list:
                   {relevant_objects}.
                   All action and condition targets must come from the detectable object list.
                   No other objects are allowed. Exception: Slicing a food object results in '<item>sliced', e.g., 'apple' becomes 'applesliced'.
                    The one exception to the above requirement is: all food objects can be acted on by slice and become <item>sliced. For example, slicing "apple" results in "applesliced"
                    ** Environmental Description**
                    {context}
                    **Task**:
                    Now, please create a behavior tree in XML format for the robot to execute the task: {task}'.
                    This task is a sub task of {big_task}
                    The following subtasks have already been completed: {completed_subtasks}
                    Make sure that you consider the cases where objects are contained in other objects or the task is already complete in your answer!
                    The completion condition for this task is: {complete_condition}
                    Ensure your response contains only the XML behavior tree and no additional text. 
                    **Object Classes**: You can only interact with objects from this list:
                   {relevant_objects}.
                   f'Remember The robot can only hold ONE object at a time. '
                   That means all targets and goals should come from the provided list.
                    "{task}"
                '''
        instruction = {"role": 'system', 'content': system_message}
        message = {"role": "user", "content": f"Behavior Tree for {task}:"}

        return [instruction, message]

    def generate_prompt_htn_ordered(self, task, known_objects, context):
        """
        Generate a prompt for the task decomposition.
        :param task: The task description.
        :return: A string prompt for the GPT model.
        """
        instruction = {
            "role": 'system',
            'content': 'You are assisting in decomposing a high-level goal for a robot. '
                       'The response format must be strictly followed for ease of parsing: '
                       'Each subgoal should be its own line with NO ADDITIONAL CHARACTERS. '
                       'The format should be short and underscore separated, similar to a pythonic class name, '
                       'with no spaces. Provide the most complete decomposition possible. '
                       'If the task is sufficiently small, do not generate subtasks. Sufficiently reduced '
                       'subtasks include tasks like empty_trash, clear_counters, empty_dishwasher, etc. '
                       'Additionally, for each subtask, provide a SINGLE object state condition '
                       'that defines when the subtask is considered complete any objects included in these condition MUST'
                       f'come from this list: {known_objects}. Format these conditions '
                       'as a bulleted list directly under the subtask, each condition on a new line '
                       'with a dash "-" at the beginning. All conditions should come from '
                       f'the following list without exception: {AI2THOR_PREDICATES_ANNOTATED} followed '
                       'by a space, the object that the condition applies to, and a 1 or 0 indicating if the predicate should be true or false. '
                       'All conditions must target a specific object and cannot contain placeholders. '
                       f'Use the following context of my surroundings to guide task decomposition: {context}'
                       f'All objects in subtask completion criteria MUST come from this list without exception: {known_objects}.'
                       f'The robot can only hold ONE object at a time. '
        }
        message = {"role": "user", "content": f"Decompose the following task into detailed subtask steps: {task}"}
        return [instruction, message]

    def generate_combined_task_prompt(self, task, known_objects, context):
        """
        Generate a prompt for both task decomposition, relevant subtask contexts, and overall context.
        :param task: The task description.
        :param known_objects: List of known objects in the environment.
        :param context: The visual and environmental context (images).
        :param states: Additional state information corresponding to the images.
        :return: A prompt for the GPT model.
        """
        instruction = {
            "role": 'system',
            'content': 'You are assisting in decomposing a high-level goal for a robot and generating relevant environmental context. '
                       'The response format must be strictly followed for ease of parsing: '
                       'Each subgoal should be its own line with NO ADDITIONAL CHARACTERS. '
                       'The format should be short and underscore separated, similar to a pythonic class name, '
                       'with no spaces. Provide the most complete decomposition possible. '
                       'If the task is sufficiently small, do not generate subtasks. Sufficiently reduced '
                       'subtasks include tasks like empty_trash, clear_counters, empty_dishwasher, etc. '
                       'Additionally, for each subtask, provide a SINGLE object state condition '
                       'that defines when the subtask is considered complete any objects included in these condition MUST'
                       f'come from this list: {known_objects}. Format these conditions '
                       'as a bulleted list directly under the subtask, each condition on a new line '
                       'with a dash "-" at the beginning. All conditions should come from '
                       f'the following list without exception: {AI2THOR_PREDICATES_ANNOTATED} followed '
                       'by a space, the object that the condition applies to, and a 1 or 0 indicating if the predicate should be true or false. '
                       'The robot can only hold ONE object at a time. Use the following context to guide task decomposition: '
                       'In addition, for each subtask, provide a short context specific to that subtask that will help '
                       'guide behavior tree construction for completing it including help and tips about locating objects.'
                        ' This context may also contain hints on potential actions or insight into what '
                        'might be needed to complete the task. The description should be '
                        'long and easily readable. Include no other text!'
                       'All context should be provided after the task decomposition and should be proceded with &&&&'
        }

        context_data = [
            {"type": "text", "text": "The following images and states are the context for the task to be completed:"}]
        for i in range(len(context)):
            image = context[i]
            img_msg = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}"
                }
            }
            context_data.append(img_msg)

        context_msg = {
            "role": "user",
            "content": context_data
        }

        message = {
            "role": "user",
            "content": f"Decompose the following task into detailed subtask steps, and provide relevant subtask-specific context: {task}."
        }

        return [instruction, context_msg, message]

    def generate_behavior_tree_refinement_prompt(self, big_task, task, actions, conditions, original_bt_xml, feedback,
                                                 known_objects, completed_subtasks, example, context, complete_condition,
                                                 image_context, error_category=None, state=None):
        """
        Generate a prompt to refine an existing behavior tree based on user feedback, in XML format.
        """
        img_msg = []
        if image_context is not None:
            img_msg = [
                {
                    'type': 'text',
                    'text': "This is the current view from the robots perspective."
                },
                {
                    'type': 'image_url',
                    "image_url": {
                        "url": f"data:image/png;base64,{image_context}",
                    }
                }
            ]
        error_info = f"Error Category: {error_category}" if error_category else ""
        '''
        **Domain**
            the domain is defined in pddl as follows:
            {DOMAIN_DEF}
        '''
        system_message = f'''
            Task attempted: "{task}"
            This is a subtask of: {big_task}
            The following subtasks have already been completed: {completed_subtasks}
            The behavior tree provided for this task resulted in an error. 
            Sub-tree where the error occurred: {original_bt_xml}
            Associated feedback and error information: {feedback}. {error_info}
            ----
            Please modify the behavior tree to address this failure, adhering to the following requirements:
            1. Valid actions for an <Action> tag: {actions}. No other actions are allowed, and each action should contain name and target members.
            where name is the action being performed and target is the target of the action.
            2. Applicable <condition>s for the actions: {conditions}. No other conditions are allowed, and each condition should contain value, name and target members.
            where value is the boolean value of the condition either 0, or 1 name is the condition being checked and 
            target is the target of the condition. an additional 'recipient' 
            member can also be provided for <condition>s for example <Condition name='isOnTop' target='plate' recipient='diningtable'>.
            3. Detectable object classes: {known_objects}. Only these objects may be used.
               Exception: All food objects can be acted on by the slice action, becoming <item>sliced. For example, "apple" becomes "applesliced".
               All action and condition targets must come from the detectable object list.
            4. The only valid tags are <Action>, <Condition>, <Sequence>, <Selector>, <root>, <?xml version="1.0"?>.
            - Use <Sequence> tags to execute all child actions or conditions in order until one fails or returns False.
            - Use <Selector> tags to execute each child action or condition in order until one succeeds or returns True.
            Here is a brief description of the environment: {context}
            Your response should contain only the corrected behavior tree in XML format and no additional text.
            the completion criteria for this task is: {complete_condition}
            Remember ALL action and condition targets and goal objects should come from this list: {known_objects}
            Make sure that you consider the cases where objects are contained in other objects or the task is already complete in your answer!
            Again, fix the following broken tree:
            Sub-tree where the error occurred: {original_bt_xml}
            ENSURE YOU ADDRESS THIS ERROR:
            Associated feedback and error information: {feedback}. {error_info}
            f'Remember The robot can only hold ONE object at a time. '
        '''
        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Corrected Behavior Tree addressing error information: {feedback}. {error_info} in XML:"}
        ]
        if image_context is not None:
            img_msg = [
                {
                    'type': 'text',
                    'text': "This is my current view of my environment"
                },
                {
                    'type': 'image_url',
                    "image_url": {
                        "url": f"data:image/png;base64,{image_context}",
                    }
                }
            ]
            prompt.insert(1, {'role': 'user', "content": img_msg})
        return prompt

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _crop_object(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract object crop using mask"""
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        crop = image[y:y+h, x:x+w].copy()
        mask_crop = mask[y:y+h, x:x+w]
        crop[~mask_crop] = 0
        return crop

    def generate_state_query(self, image: np.ndarray, masks: np.ndarray, metadata: Dict) -> str:
        """Generate a prompt for GPT-4V to analyze object states"""
        # Create individual object crops
        object_crops = []
        for mask_id in np.unique(masks)[1:]:  # Skip 0 (background)
            if mask_id in metadata:
                mask = masks == mask_id
                crop = self._crop_object(image, mask)
                object_crops.append({
                    'id': mask_id,
                    'crop': crop,
                    'area': metadata[mask_id]['area']
                })

        # Sort objects by area (largest first)
        object_crops.sort(key=lambda x: x['area'], reverse=True)
        
        # Create prompt
        prompt = f"""Analyze this scene image to identify objects and their states. For each segmented region:

    1. First identify what the object is, considering its visual appearance and context.
    2. Then evaluate the following predicates for each identified object:

    Boolean predicates (1 for true, 0 for false):
    {', '.join(BOOL_PREDS)}

    Relational predicates (requiring additional object or room information):
    {', '.join(RELATIONAL_PREDS)}

    Please provide results in this JSON format:
    [
        {{
            "name": "object_name",
            "predicates": {{
                "visible": 1,
                "receptacle": 0,
                "inRoom": {{"value": 1, "room": "kitchen"}},
                "isOnTop": {{"value": 1, "object": "counter"}},
                "isInside": {{"value": 0, "object": null}}
            }},
            "region_id": "region_1",
            "caption": "a short text description of the object and its state that would be good context for an LLM"
        }},
        ...
    ]

    For relational predicates:
    - 'inRoom' should specify the room name
    - 'isOnTop' should specify the object being rested upon
    - 'isInside' should specify the containing object

    Focus on clearly visible properties and spatial relationships. Consider:
    - Physical properties (broken, sliced, cooked, etc.)
    - Containment relationships (what objects can contain others)
    - Spatial relationships (on top, inside, proximity)
    - States (open/closed, switched on/off, filled)

    Remember all of the following predicates should be evaluated for all detected objects and be present in the output state:
    {', '.join(BOOL_PREDS + RELATIONAL_PREDS)}

    For boolean predicates, use 1 for true and 0 for false.
    For relational predicates, include both the value (1/0) and the related object/room information."""
        return prompt

    def get_object_states(self, image: np.ndarray, mask_generator) -> Tuple[Dict, np.ndarray, Dict]:
        """
        Main method to get object states from an image
        
        Returns:
            Tuple[Dict, np.ndarray, Dict]: 
                - Object states dictionary
                - Labeled masks array
                - Mask metadata dictionary
        """
        # Get segmentation masks using FastSAM/OWL-ViT
        masks, metadata = mask_generator.generate_masks(image)
        
        if not np.any(masks):
            return {}, masks, metadata
            
        # Generate query for GPT-4V
        prompt = self.generate_state_query(image, masks, metadata)
        
        # Get state analysis from GPT-4V
        try:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._encode_image(image)}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            response_text = self.query_llm(messages)
            
            import json

            # Extract JSON from response
            print(response_text)
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                try:
                    object_states = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Response text: {response_text}")
                    object_states = {}
            else:
                object_states = {}
                
        except Exception as e:
            print(f"Error querying GPT-4V: {e}")
            return {}, masks, metadata
        
        # Add mask data to object states
        for state_info in object_states:
            print(state_info.keys())
            region_id = state_info['region_id'].split('_')[1]
            if region_id in metadata:
                mask_id = region_id
                state_info['mask'] = masks == mask_id
                state_info['bbox'] = metadata[mask_id]['bbox']
                state_info['image'] = image
        
        return object_states, masks, metadata


    @sleep_and_retry
    @limits(calls=100, period=60)  # Example: Max 10 calls per minute
    def query_llm(self, prompt):
        """
        Query the GPT model with the generated prompt.
        :param prompt: The prompt for task decomposition.
        :return: The model's response as a task decomposition.
        """
        # print(f"LLM PROMPT: {prompt}")
        try:
            # pprint.pp(prompt)
            self.conversation_history.append(prompt)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=1000,
                temperature=0.5
            )
            self.conversation_history.append([{
                'role': 'llm',
                'content': response.choices[0].message.content
            }])
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None  # Or handle appropriately
        return response.choices[0].message.content

    def query_llm_for_refinement(self, big_task, task, original_decomposition, user_feedback):
        """
        Query the LLM with the original decomposition and user feedback to generate a refined decomposition.
        """
        prompt = self.generate_refined_prompt(big_task, task, original_decomposition, user_feedback)
        try:
            self.add_to_history()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=1000,
                temperature=0.5
            )
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None  # Or handle appropriately
        return response.choices[0].message.content

    def get_task_decomposition(self, task, known_objects, context):
        """
        Get the task decomposition from the GPT model.
        :param task: The task description.
        :return: A list or string of decomposed tasks.
        """
        prompt = self.generate_prompt_htn(task, known_objects, context)
        decomposition = self.query_llm(prompt)
        return parse_llm_response(decomposition)

    def get_task_decomposition_ordered(self, task, known_objects, context):
        """
        Get the task decomposition from the GPT model.
        :param task: The task description.
        :return: A list or string of decomposed tasks.
        """
        prompt = self.generate_prompt_htn_ordered(task, known_objects, context)
        decomposition = self.query_llm(prompt)
        return parse_llm_response_ordered(decomposition)

    def get_task_decomposition_ordered_context(self, task, known_objects, context):
        """
        Get the task decomposition from the GPT model.
        :param task: The task description.
        :return: A list or string of decomposed tasks.
        """
        prompt = self.generate_combined_task_prompt(task, known_objects, context)
        decomposition = self.query_llm(prompt)
        print(decomposition)
        context = decomposition.split('&&&&')[1]
        return parse_llm_response_ordered(decomposition.split('&&&&')[0]), context

    def get_task_id(self, task, context, states):
        prompt = self.generate_prompt_task_id(task, context, states)
        ret = self.query_llm(prompt)
        print(ret)
        context_object = ret.split('\n')[1]
        task_id = ret.split('\n')[0]
        return task_id, context_object

    def get_behavior_tree(self, big_task, task, actions, conditions, example, known_objects, completed_subtasks,
                          context, complete_condition):
        """
        Get the behavior tree from the GPT model for a given task.
        """
        prompt = self.generate_behavior_tree_prompt(big_task, task, actions, conditions, example, known_objects,
                                                    completed_subtasks, context, complete_condition)

        behavior_tree_xml = self.query_llm(prompt)
        for i in range(len(behavior_tree_xml)):
            if behavior_tree_xml[i] == '<':
                behavior_tree_xml = behavior_tree_xml[i:]
                break
        for i in range(len(behavior_tree_xml)):
            if behavior_tree_xml[len(behavior_tree_xml) - i - 1] == '>':
                behavior_tree_xml = behavior_tree_xml[:len(behavior_tree_xml) - i ]
                break
        behavior_tree_xml = behavior_tree_xml.replace('```', '')
        return behavior_tree_xml

    def refine_behavior_tree(self, big_task, task, actions, conditions, original_bt_xml, user_feedback, known_objects,
                             completed_subtasks, example, context, complete_condition, image_context):
        """
        Refine a behavior tree based on user feedback.
        """
        prompt = self.generate_behavior_tree_refinement_prompt(big_task, task, actions, conditions, original_bt_xml,
                                                               user_feedback, known_objects, completed_subtasks,
                                                               example, context, complete_condition, image_context)
        print(prompt)
        refined_behavior_tree_xml = self.query_llm(prompt)
        print(f"Refined Behavior Tree: {refined_behavior_tree_xml}")
        for i in range(len(refined_behavior_tree_xml)):
            if refined_behavior_tree_xml[i] == '<':
                refined_behavior_tree_xml = refined_behavior_tree_xml[i:]
                break
        refined_behavior_tree_xml = refined_behavior_tree_xml.replace('```', '')
        return refined_behavior_tree_xml

if __name__ == "__main__":
    from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_ACTIONS
    # Example usage
    llm_interface = LLMInterface()
    task = "get a glass of water"
    bt_xml = llm_interface.get_behavior_tree(task, AI2THOR_ACTIONS)
    print(bt_xml)
    # Assuming some feedback was received
    feedback = "The robot fails to find the sink."
    refined_bt_xml = llm_interface.refine_behavior_tree(task, bt_xml, feedback)
    print(refined_bt_xml)
