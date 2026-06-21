from langchain_core.prompts import PromptTemplate
from strictjson import *

from agents import load_prompt2
from env_api.actions_executors import CharacterInfo
from memory.entity_description_generator import entity_description_generator
from utils.environment_state import EnvironmentState
from utils.llm_connect import LLMConnect


class CriticAgent:
    """
    Agent responsible for evaluating and critiquing plans.
    
    This agent analyzes proposed plans and provides feedback on their
    feasibility and correctness. It uses LLM-based evaluation to determine
    whether plans are suitable for execution.
    """

    def __init__(self, character_name: str, model: str = 'gpt-4.1-mini') -> None:
        """
        Initialize the CriticAgent with character and model configuration.
        
        Args:
            character_name: Name of the character this agent evaluates plans for
            model: LLM model to use for evaluation (default: 'gpt-4.1-mini')
        """
        self.agent_model = LLMConnect(model=model)
        self.character_info = CharacterInfo(character_name)
        self.missing_items_text = ''
        self.response_generation_attempts = 5
        self.environment_state = EnvironmentState(character_name)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> tuple[dict, bool]:
        """
        Call the LLM with retry logic for plan evaluation.
        
        This method attempts to generate a critique and evaluation from the LLM
        with retry logic in case of failures.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            
        Returns:
            Tuple containing:
                - critique: String containing the critique of the plan
                - plan_ok: Boolean indicating whether the plan is acceptable
        """
        plan_ok = False
        response_generation_successful = False
        attempts = 0
        critique = ''
        while not response_generation_successful and attempts < self.response_generation_attempts:
            try:
                critique_data = strict_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_format={
                'critique': 'Explain each problem clearly. Refer to subtask number or contents if needed.',
                'plan_ok': 'If plan pass checks, return true, else return false, type: bool'},
                llm = self.agent_model.send_message
                )
                critique = critique_data['critique']
                plan_ok = critique_data['plan_ok']
                response_generation_successful = True
            except Exception as _:
                attempts += 1
        if not response_generation_successful:
            critique = "Critic agent wasn't able to generate a critic. Try submitting current plan again."
            plan_ok = False

        return critique, plan_ok

    def evaluate_plan(self, action_plan: dict, target_task: dict, system_prompt_name: str, user_prompt_name: str) -> tuple[dict, bool]:
        """
        Evaluate an action plan against a target task.
        
        This method analyzes whether a proposed action plan is suitable for
        completing the given task. It considers the current environment state,
        entity information, and plan details.
        
        Args:
            action_plan: Dictionary containing the proposed action plan with keys:
                         - subtasks: List of proposed actions
            target_task: Dictionary containing the target task information with keys:
                         - type: Task type
                         - target: Task target
                         - quantity: Task quantity
            system_prompt_name: Name of the system prompt to use for evaluation
            user_prompt_name: Name of the user prompt to use for evaluation
            
        Returns:
            Tuple containing:
                - critique: String containing detailed feedback on the plan
                - plan_ok: Boolean indicating whether the plan is acceptable
        """
        environment_state =  self.environment_state.get_environment_state(self.missing_items_text)
        environment_state['objective'] = f'{target_task['type']} {target_task['target']} {target_task['quantity']}'
        environment_state['current_subtasks'] = action_plan['subtasks']
        entity_info = entity_description_generator(target_task['target'])
        environment_state['entity_info'] = entity_info
        system_prompt = load_prompt2(system_prompt_name,prompts_version=2)
        template_main = load_prompt2(user_prompt_name,prompts_version=2)
        prompt_main = PromptTemplate.from_template(template_main)
        prompt_main = prompt_main.invoke(environment_state)
        return self._call_llm(system_prompt, prompt_main.text)
