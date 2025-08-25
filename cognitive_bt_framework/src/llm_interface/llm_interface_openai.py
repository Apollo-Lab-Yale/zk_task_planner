from openai import OpenAI, AsyncOpenAI
from cognitive_bt_framework.utils import setup_openai, get_openai_key, parse_llm_response, parse_llm_response_ordered
from ratelimit import limits, sleep_and_retry
from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_PREDICATES_ANNOTATED
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image
from io import BytesIO
import json
import cv2
import base64
import numpy as np
import asyncio
import uuid
from cognitive_bt_framework.utils import BOOL_PREDS, RELATIONAL_PREDS


class LLMInterfaceOpenAI:
    def __init__(self, model_name="o4-mini"):
        self.client = OpenAI(api_key=get_openai_key())
        self.async_client = AsyncOpenAI(api_key=get_openai_key())
        self.model_name = model_name
        self.is_realtime = "realtime" in model_name
        self.conversation_history = []
        self.token_limit = 40000

    def add_to_history(self, role, content):
        new_message = {"role": role, "content": content}
        self.conversation_history.append(new_message)
        tokens = sum(len(m['content']) for m in self.conversation_history)
        while tokens > self.token_limit:
            tokens -= len(self.conversation_history[0]['content'])
            self.conversation_history.pop(0)

    def generate_prompt_htn(self, task, known_objects, context):
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
        instruction = {"role": 'system', 'content': 'You are assisting in generating a task id for a user requested task'
                                                   ' The format should be short and underscore separated similar to a '
                                                   'pythonic class name  with no spaces. Additionally, using the images provided'
                                                   ' as task context, generate all relevant environmental'
                                                   ' information that may assist an llm in generating subtasks '
                                                   'and action sequences for the task'
                                                   f' {task} and behavior trees to complete those sub tasks be as detailed as '
                                                   'possible while ensuring that it is easy to comprehend for an LLM attempting to plan'
                                                   '. The task ID and context should be separated by a new line '
                                                   'character. Additionally give hints on potential actions or insight into what '
                                                   'might be needed to complete the task. The description should be '
                                                   'long and easily readable. Include no other text!'}
        context_data = [{"type": "text", "text": "The following images and states are the context for the task to be completed"}]
        for image in context:
            context_data.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"}
            })
        return [
            instruction,
            {"role": "user", "content": context_data},
            {"role": "user", "content": f"Generate an id and context, in the format described for this task: {task}"}
        ]

    def generate_behavior_tree_prompt(self, big_task, task, actions, conditions, example, relevant_objects,
                                    completed_subtasks, context, complete_condition):
        completed_goals_str = f'''
        **Completed Sub Goals**
        The following subgoals have already been completed:
        {completed_subtasks}
        ''' if completed_subtasks else ""
        
        system_message = f'''
            You are going to be tasked with creating a behavior tree in XML format for a robot to execute a specific task.
            Follow these rules carefully:
            **Actions**: Use only the actions from this list: {actions}
            **Conditions**: Use only the conditions from this list: {conditions}
            {completed_goals_str}
            **Behavior Tree Structure**:
            - Use <Sequence> tags to execute all child actions or conditions in order until one fails
            - Use <Selector> tags to execute each child action or condition in order until one succeeds
            **Object Classes**: You can only interact with objects from this list: {relevant_objects}
            All action and condition targets must come from the detectable object list.
            **Environmental Description**: {context}
            **Task**: Create a behavior tree in XML format for the robot to execute: {task}
            This is a subtask of {big_task}
            Completion condition: {complete_condition}
            Ensure your response contains only the XML behavior tree.
            Remember the robot can only hold ONE object at a time.
        '''
        return [
            {"role": 'system', 'content': system_message},
            {"role": "user", "content": f"Behavior Tree for {task}:"}
        ]

    def generate_behavior_tree_refinement_prompt(self, big_task, task, actions, conditions, original_bt_xml, feedback,
                                               known_objects, completed_subtasks, example, context, complete_condition, 
                                               image_context, error_category=None):
        img_content = []
        if image_context:
            img_content = [
                {"role": "user", "content": [
                    {"type": "text", "text": "Current robot view:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_context}"}}
                ]}
            ]

        error_info = f"Error Category: {error_category}" if error_category else ""
        system_message = f'''
            Task: "{task}" (subtask of: {big_task})
            Completed subtasks: {completed_subtasks}
            Error in behavior tree: {original_bt_xml}
            Error feedback: {feedback} {error_info}
            
            Requirements:
            1. Valid actions for <Action>: {actions}
            2. Valid conditions for <Condition>: {conditions}
            3. Valid objects: {known_objects}
            4. Valid tags: <Action>, <Condition>, <Sequence>, <Selector>, <root>, <?xml>
            
            Environment context: {context}
            Completion criteria: {complete_condition}
            Remember: Robot can hold ONE object at a time.
        '''
        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Provide corrected behavior tree:"}
        ]
        return img_content + prompt if img_content else prompt
    
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

            Remember all of the following predicates should be evaluated for all detected objects 0 value predicates can be ommitted:
            {', '.join(BOOL_PREDS + RELATIONAL_PREDS)}

            For boolean predicates, use 1 for true and 0 for false.
            For relational predicates, include both the value (1/0) and the related object/room information.
        """
        return prompt

    def _encode_image(self, image: np.ndarray) -> str:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    async def query_llm_realtime(self, messages):
        content = []
        for message in messages:
            if isinstance(message["content"], list):
                content.extend(message["content"])
            else:
                content.append({"type": "input_text", "text": message["content"]})

        async with self.async_client.beta.realtime.connect(model=self.model_name) as connection:
            await connection.session.update(session={'modalities': ['text', 'vision']})
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": content,
                }
            )
            await connection.response.create()
            response_text = ""
            print("ASYNC MODEL")
            async for event in connection:
                print("%%%%%%%%%%%%%%%%%%%")
                if event.type == 'response.text.delta':
                    response_text += event.delta
                    print(event.delta, flush=True, end="")
                elif event.type == "response.text.done":
                    print(response_text)
                elif event.type == "response.done":
                    break
                elif event.type == "error":
                    print(event.error.type)
                    print(event.error.message)
                    print(event.error.code)
                print(event.type)

            return response_text

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def query_llm(self, prompt):
        if self.is_realtime:
            return await self.query_llm_realtime(prompt)
        
        try:
            self.conversation_history.append(prompt)
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_completion_tokens=4096,
                temperature=0.5
            )
            self.conversation_history.append([{
                'role': 'llm',
                'content': response.choices[0].message.content
            }])
            print(response)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None
        
    @sleep_and_retry
    @limits(calls=100, period=60)
    def query_llm_sync(self, prompt):
        try:
            self.conversation_history.append(prompt)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_completion_tokens=4096,
                # temperature=0.5
            )
            self.conversation_history.append([{
                'role': 'llm',
                'content': response.choices[0].message.content
            }])
            print(response)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None

    async def get_task_decomposition(self, task, known_objects, context):
        prompt = self.generate_prompt_htn(task, known_objects, context)
        decomposition = await self.query_llm(prompt)
        return parse_llm_response(decomposition)

    async def get_task_decomposition_ordered(self, task, known_objects, context):
        prompt = self.generate_prompt_htn_ordered(task, known_objects, context)
        decomposition = await self.query_llm(prompt)
        return parse_llm_response_ordered(decomposition)

    async def get_task_id(self, task, context, states):
        prompt = self.generate_prompt_task_id(task, context, states)
        ret = await self.query_llm(prompt)
        context_object = ret.split('\n')[1]
        task_id = ret.split('\n')[0]
        return task_id, context_object

    async def get_behavior_tree(self, big_task, task, actions, conditions, example, known_objects, completed_subtasks,
                              context, complete_condition):
        prompt = self.generate_behavior_tree_prompt(big_task, task, actions, conditions, example, known_objects,
                                                  completed_subtasks, context, complete_condition)
        behavior_tree_xml = await self.query_llm(prompt)
        return self._clean_behavior_tree(behavior_tree_xml)

    async def refine_behavior_tree(self, big_task, task, actions, conditions, original_bt_xml, user_feedback, known_objects,
                                 completed_subtasks, example, context, complete_condition, image_context):
        prompt = self.generate_behavior_tree_refinement_prompt(big_task, task, actions, conditions, original_bt_xml,
                                                             user_feedback, known_objects, completed_subtasks,
                                                             example, context, complete_condition, image_context)
        refined_behavior_tree_xml = await self.query_llm(prompt)
        return self._clean_behavior_tree(refined_behavior_tree_xml)

    async def get_object_states(self, image: np.ndarray, mask_generator) -> Tuple[Dict, np.ndarray, Dict]:
        masks, metadata = mask_generator.generate_masks(image)
        labeled_image = mask_generator.visualize_masks(image, masks, metadata)
        cv2.imwrite('labeled_image.png', labeled_image)
        if not np.any(masks):
            return {}, masks, metadata
            
        prompt = self.generate_state_query(image, masks, metadata)
        text_type = 'input_text' if self.is_realtime else 'text'
        image_type = "image_url"
        messages = [{
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._encode_image(labeled_image)}"}},
                {"type": text_type, "text": prompt}
            ]
        }]
        
        response_text = await self.query_llm(messages)
        print(response_text)
        return self._process_object_states(response_text, masks, metadata, image)

    def _clean_behavior_tree(self, behavior_tree_xml):
        behavior_tree_xml = behavior_tree_xml.replace('```', '')
        start_idx = behavior_tree_xml.find('<')
        end_idx = behavior_tree_xml.rfind('>') + 1
        if start_idx >= 0 and end_idx > 0:
            return behavior_tree_xml[start_idx:end_idx]
        return behavior_tree_xml

    def _process_object_states(self, response_text, masks, metadata, image):
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        object_states = {}
        image_id = uuid.uuid4()
        object_states['images'] = {image_id: image}
        if start_idx >= 0 and end_idx > start_idx:
            try:
                object_states = json.loads(response_text[start_idx:end_idx])
                for state_info in object_states:
                    region_id = state_info['region_id']
                    region_id = int(region_id.split('r')[-1])
                    print(region_id)
                    if region_id in metadata:
                        mask_id = region_id
                        state_info['mask'] = masks == mask_id
                        state_info['bbox'] = metadata[mask_id]['bbox']
                        state_info['image_id'] = image_id
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
        
        return object_states, masks, metadata
    
    def get_response_with_image(self, prompt: str, image_b64: str) -> str:
        """
        Get response from LLM with image input for task planning
        
        Args:
            prompt: Text prompt for the LLM
            image_b64: Base64 encoded image
            
        Returns:
            LLM response text
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=4096
                # Remove temperature parameter - use default (1.0) for o4-mini model
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in get_response_with_image: {e}")
            return None