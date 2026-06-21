import json

from langchain_core.prompts import PromptTemplate
from strictjson import *

from agents import load_prompt2
from env_api.actions_executors import Move, Gather, Fight, Craft, CharacterInfo, Equip
from memory.entity_description_generator import entity_description_generator
from utils.environment_state import EnvironmentState
from utils.llm_connect import LLMConnect


class ActionAgent:
    """
    Agent responsible for task decomposition and action execution.
    
    This agent handles the conversion of high-level tasks into executable actions
    using LLM-based decomposition. It also manages the execution of actions
    in the environment.
    """

    def __init__(self, character_name, model = 'gpt-4.1-mini'):
        """
        Initialize the ActionAgent with character and model configuration.
        
        Args:
            character_name: Name of the character this agent controls
            model: LLM model to use for decomposition (default: 'gpt-4.1-mini')
        """
        self.agent_model = LLMConnect(model=model)
        self.character_info = CharacterInfo(character_name)
        self.action_move = Move(character_name)
        self.action_gather = Gather(character_name)
        self.action_fight = Fight(character_name)
        self.action_craft = Craft(character_name)
        self.action_equip = Equip(character_name)
        self.missing_items_text = ''
        self.response_generation_attempts = 5
        self.environment_state = EnvironmentState(character_name)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Call the LLM with retry logic for response generation.
        
        This method attempts to generate a response from the LLM with
        retry logic in case of failures.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            
        Returns:
            Dictionary containing the LLM response with 'subtasks' key
        """
        decomposition_result = {}
        response_generation_successful = False
        attempts = 0
        while not response_generation_successful and attempts < self.response_generation_attempts:
            try:
                decomposition_result = strict_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_format={'subtasks': 'JSON of subtasks'},
                llm=self.agent_model.send_message
                )
                if type(decomposition_result['subtasks']) is str:
                    decomposition_result['subtasks'] = json.loads(decomposition_result['subtasks'])
                response_generation_successful = True
            except Exception as _:
                attempts += 1
        if not response_generation_successful:
            decomposition_result = {'subtasks': [{'error': 'Decomposition agent failed to create subtasks.'}]}
        else:
            return decomposition_result

    def universal_decomposition(self, task_to_decompose: dict, system_prompt_name: str, user_prompt_name: str) -> dict:
        """
        Perform universal task decomposition using LLM prompts.
        
        This method takes a task and decomposes it into subtasks using
        specified system and user prompts. It gathers environment state
        and entity information to provide context for the decomposition.
        
        Args:
            task_to_decompose: Dictionary containing task information with keys:
                               - type: Task type
                               - target: Task target
                               - quantity: Task quantity
                               - previous_plan: Previous decomposition attempt
                               - critique: Feedback from previous attempt
            system_prompt_name: Name of the system prompt to use
            user_prompt_name: Name of the user prompt to use
            
        Returns:
            Dictionary containing decomposed subtasks
        """
        task_type = task_to_decompose['type']
        task_target = task_to_decompose['target']
        task_quantity = task_to_decompose['quantity']
        environment_state = self.environment_state.get_environment_state(self.missing_items_text)
        entity_info = entity_description_generator(task_target)
        environment_state['objective'] = f'Task type: {task_type} \nTask target:{task_target} \nQuantity: {task_quantity}'
        environment_state['entity_info'] = entity_info
        environment_state['previous_plan'] = task_to_decompose['previous_plan']
        environment_state['critique'] = task_to_decompose['critique']
        system_prompt = load_prompt2(system_prompt_name, prompts_version=2)
        template_main = load_prompt2(user_prompt_name, prompts_version=2)
        prompt_main = PromptTemplate.from_template(template_main)
        prompt_main = prompt_main.invoke(environment_state)
        return self._call_llm(system_prompt, prompt_main.text)

    def execute_action(self, action: dict[str,str,str] , force_execution: bool = False) -> dict[str,str]:
        """
        Execute a single action in the environment.
        
        This method maps action types to their corresponding executor methods
        and handles the execution logic for different action types. It supports
        force execution mode.
        
        Args:
            action: Dictionary containing action information with keys:
                    - type: Action type ('Move', 'Gather', 'Craft', 'Equip', 'Fight')
                    - target: Target coordinates, item, or resource
                    - quantity: Number of times to perform the action
            force_execution: Whether to force execution even if actions fail

                           
        Returns:
            Dictionary containing execution result with keys:
                - status: 'Success' or 'Fail'
                - message: Description of the result
        """
        action_result = {'status': '', 'message': ''}
        if action['type'] == 'Move':
            action_result = self.action_move.action(action['target'][0],action['target'][1])
        elif action['type'] == 'Gather':
            action_result = self.action_gather.action(resource_count= action['quantity'])
        elif action['type'] == 'Craft':
            action_result = self.action_craft.action(action['target'],action['quantity'])
        elif action['type'] == 'Equip':
            if action['quantity'] == 1:
                action_result = self.action_equip.action(action['target'])
            else:
                for i in range(action['quantity']):
                    action_result = self.action_equip.action(action['target'])
                    if action_result['status'] == 'Fail' and force_execution is False:
                        action_result['message'] = 'Failed to equip multiple copies of item'
                        break
                    elif action_result['status'] == 'Fail' and force_execution is True:
                        return {'status': 'Success', 'message': 'FORCE EXECUTION'}
                    else:
                        continue

        elif action['type'] == 'Fight':
            for i in range(action['quantity']):
                action_result = self.action_fight.action()
                if action_result['status'] == 'Fail' and force_execution is False:
                    break
                elif action_result['status'] == 'Fail' and force_execution is True:
                    return {'status': 'Success', 'message': 'FORCE EXECUTION'}
                else:
                    continue
        else:
            action_result = {'status':'Fail', 'message': 'Unknown action type'}

        return action_result

