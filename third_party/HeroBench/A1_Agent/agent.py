import json
from enum import Enum

import dotenv

from agents.action_agent import ActionAgent
from agents.critic_agent import CriticAgent
from env_api.actions_executors import CharacterInfo
from env_api.api import evaluate_craft, evaluate_slay
from utils.agent_logger import AgentLogger

dotenv.load_dotenv()

class AgentState(Enum):
    """Enumeration of possible agent states during task execution."""
    SUBTASK_DECOMPOSITION = "SUBTASK_DECOMPOSITION"
    ACTION_DECOMPOSITION = "ACTION_DECOMPOSITION"
    FINISHED = "FINISHED"
    FAILED = 'FAILED'

class A1Agent:
    """
    Main agent class that orchestrates task decomposition and execution.
    
    This agent uses a hierarchical approach with two sub-agents:
    - ActionAgent: Handles task decomposition and action execution
    - CriticAgent: Evaluates and critiques generated plans
    
    The agent operates in two main phases:
    1. Task to subtasks decomposition
    2. Subtasks to actions decomposition
    """

    def __init__(self, character_name: str, model: str = 'gpt-4.1-mini', load_from_json: bool = False) -> None:
        """
        Initialize the A1Agent with character and model configuration.
        
        Args:
            character_name: Name of the character this agent controls
            model: LLM model to use for agent operations (default: 'gpt-4.1-mini')
            load_from_json: Whether to load agent settings from agent_settings.json
        """
        self.character_name = character_name
        self.action_agent = ActionAgent(character_name, model)
        self.critic_agent = CriticAgent(character_name, model)
        self.character_info = CharacterInfo(character_name)
        self.subtasks_decomposition_attempts_limit = 10
        self.actions_decomposition_attempts_limit = 10
        self.current_subtask_decomposition_attempts = 0
        self.current_action_decomposition_attempts = 0
        self.agent_state = AgentState.SUBTASK_DECOMPOSITION
        self.execute_actions_chains = False
        self.logger = AgentLogger(console_verbose=True)
        self.logger.start_logger()
        if load_from_json:
            self._load_params()

    def _load_params(self) -> None:
        try:
            with open('agent_settings.json', 'r') as f:
                json_data = json.load(f)
                target_keys = ['subtasks_decomposition_attempts_limit','actions_decomposition_attempts_limit','execute_actions_chains','console_verbose']
                all_required_keys_present = all (key in target_keys for key in json_data.keys())
                if not all_required_keys_present:
                        self.logger.log('Failed to load settings from json file')
                        return
                else:
                    self.logger.log('Overriding agent settings')
                    self.subtasks_decomposition_attempts_limit = json_data['subtasks_decomposition_attempts_limit']
                    self.actions_decomposition_attempts_limit = json_data['actions_decomposition_attempts_limit']
                    self.execute_actions_chains = json_data['execute_actions_chains']
                    self.logger.console_verbose = json_data['console_verbose']
                    self.logger.log('Overriding agent settings complete')
                    return
        except:
            self.logger.log('Failed to load settings from json file')
            return

    @staticmethod
    def _ensure_missing_fields(actions: dict) -> dict:
        if 'previous_subtasks' not in actions:
            actions['previous_plan'] = 'Previous plan not available'
        if 'critique' not in actions:
            actions['critique'] = 'Critique for previous plan not available'
        return actions

    def _log_subtask_decomposition_failure(self, failed_subtask: dict) -> str:
        self.logger.log("Subtasks to actions decomposition interrupted, returning to task to subtasks decomposition")
        error_text = f'Plan step "{failed_subtask['type']} {failed_subtask['target']} {failed_subtask['quantity']}" has failed'
        self.logger.log(error_text)
        return error_text

    def _log_plan(self, plan_data: dict) -> None:
        self.logger.log_plan(plan_data, 'info')
        self.logger.log('CHARACTER INTERMEDIATE STATE')
        self.logger.log(str(self.character_info.action()))

    def set_missing_items(self, items_names_list: list) -> None:
        """
        Set missing items names for initial decomposition.
        
        This method configures both action and critic agents with information
        about items that are missing from the character's inventory, which
        helps guide the decomposition process.
        
        Args:
            items_names_list: List of missing item names to be considered during planning
        """
        items_text = ', '.join(items_names_list)
        self.action_agent.missing_items_text = items_text
        self.critic_agent.missing_items_text = items_text
        self.logger.log(
            f'Injected items: {items_text}'
        )

    def reset_decomposition_attempts(self, subtask: bool = False) -> None:
        """
        Reset decomposition attempts counters.
        
        Can reset either low-level (actions) or high-level (subtasks) decomposition limits.
        
        Args:
            subtask: If True, reset high-level decomposition limits (subtasks).
                    If False, reset low-level decomposition limits (actions).
        """
        if subtask:
            self.current_subtask_decomposition_attempts = 0
        else:
            self.current_action_decomposition_attempts = 0

    def task_to_subtasks(self, task: dict) -> dict:
        """
        Decompose main task into subtasks using action and critic agents.
        
        This method iteratively attempts to break down a high-level task into
        subtasks, using the critic agent to validate each attempt. If the
        maximum number of attempts is reached, the agent state is set to FAILED.
        
        Args:
            task: Dictionary containing task information with keys:
                  - type: Task type (e.g., 'Craft', 'Fight')
                  - target: Target of the task
                  - quantity: Quantity to craft/fight
                  - previous_plan: Previous decomposition attempt
                  - critique: Feedback from previous attempt
                  
        Returns:
            Dictionary containing:
                - subtasks: List of decomposed subtasks
                - previous_plan: Previous plan attempt
                - critique: Feedback from critic agent
                - is_ok: Boolean indicating if decomposition was successful
        """
        decomposition_successful = False
        subtasks = {'subtasks': [], 'previous_plan': '', 'critique': "Wasn't able to create subtasks"}
        self.logger.decomposition_step_mark('TASK -> SUBTASKS')
        while not decomposition_successful:
            if self.current_subtask_decomposition_attempts == self.subtasks_decomposition_attempts_limit:
                self.agent_state = AgentState.FAILED
                subtasks['is_ok'] = False
                self.logger.decomposition_step_mark('TASK -> SUBTASKS', finished=True)
                return subtasks
            subtasks = self.action_agent.universal_decomposition(task,
                                                'create_subtasks_system',
                                                'create_subtasks')
            critique, decomposition_successful = self.critic_agent.evaluate_plan(subtasks, task,
                                                             'subtasks_critic_system',
                                                             'subtasks_critic')
            task['critique'] = critique
            task['previous_plan'] = subtasks
            task['is_ok'] = decomposition_successful
            self.logger.log('SUBTASKS')
            self.logger.log_plan(task,'info')
            if type(decomposition_successful) == str:
                decomposition_successful = eval(decomposition_successful.capitalize())
            if decomposition_successful:
                self.agent_state = AgentState.ACTION_DECOMPOSITION
                self.current_subtask_decomposition_attempts += 1
                break
            else:
                self.current_subtask_decomposition_attempts += 1
        subtasks['is_ok'] = decomposition_successful
        self.logger.decomposition_step_mark('TASK -> SUBTASKS', finished=True)
        return subtasks

    def _run_action(self, action: dict[str,str,str]) -> dict:
        action_result = self.action_agent.execute_action(action)
        action_text = f'Action {action} result: {action_result}'
        self.logger.log(action_text)
        return action_result

    def subtasks_to_actions(self, subtask):
        """
        Decompose subtask into executable actions using action and critic agents.
        
        This method converts a subtask into a sequence of executable actions.
        If execute_actions_chains is True, it will also attempt to execute
        the actions and validate their success.
        
        Args:
            subtask: Dictionary containing subtask information with keys:
                     - type: Subtask type
                     - target: Target of the subtask
                     - quantity: Quantity to process
                     - previous_plan: Previous action plan
                     - critique: Feedback from previous attempt
                     
        Returns:
            Dictionary containing:
                - subtasks: List of executable actions
                - previous_plan: Previous action plan
                - critique: Feedback from critic agent
                - is_ok: Boolean indicating if decomposition was successful
        """
        decomposition_successful, subtask_execution_successful  = False,  False
        self.reset_decomposition_attempts()
        self.logger.decomposition_step_mark('SUBTASKS -> ACTIONS')
        while not decomposition_successful and not subtask_execution_successful:
            if self.current_action_decomposition_attempts == self.actions_decomposition_attempts_limit:
                self.agent_state = AgentState.SUBTASK_DECOMPOSITION
                subtask['is_ok'] = False
                self.logger.decomposition_step_mark('SUBTASKS -> ACTIONS',finished=True)
                return subtask
            actions = self.action_agent.universal_decomposition(subtask,
                                                                      'create_actions_system',
                                                                      'create_actions')
            critique, decomposition_successful = self.critic_agent.evaluate_plan(actions,
                                                                                 subtask,
                                'actions_critic_system',
                                'actions_critic', )
            subtask['critique'], subtask['previous_plan'], subtask['is_ok'] = critique, actions, decomposition_successful
            self.logger.log('ACTIONS')
            self.logger.log_plan(subtask,'info')
            if type(decomposition_successful) == str:
                decomposition_successful = eval(decomposition_successful.capitalize())
            if not decomposition_successful:
                self.current_action_decomposition_attempts += 1
            if not self.execute_actions_chains and decomposition_successful:
                actions['is_ok'] = True
                self.logger.decomposition_step_mark('SUBTASKS -> ACTIONS',finished=True)
                return actions
            if decomposition_successful and not subtask_execution_successful and self.execute_actions_chains:
                subtask_execution_successful = True
                for action in actions['subtasks']:
                    action_result = self._run_action(action)
                    if action_result['status'] == 'Fail':
                        self.current_action_decomposition_attempts += 1
                        decomposition_successful, subtask_execution_successful  = False, False
                        subtask['critique'] = f'Action {action['type']} {action['target']} {action['quantity']}" has failed due to error: "{action_result['message']}"'
                        break

            if decomposition_successful == True and subtask_execution_successful == True:
                actions['is_ok'] = True
                self.logger.decomposition_step_mark('SUBTASKS -> ACTIONS',finished=True)
                return actions

    def evaluate_task_completion(self, task_for_validation: dict) -> bool:
        """
        Evaluate whether a task has been completed successfully in the environment.
        
        This method validates task completion by checking the environment
        to see if the required items have been crafted or monsters have been slain.
        
        Args:
            task_for_validation: Dictionary containing task information with keys:
                                 - type: Task type ('Craft' or 'Fight')
                                 - target: Target item or monster name
                                 
        Returns:
            Boolean indicating whether the task was completed successfully
        """
        task_completed_successfully = False
        if task_for_validation['type'] == 'Craft':
            task_completed_successfully = evaluate_craft(self.character_name, task_for_validation['target'], n=900)
            self.logger.log(f'Validation on environment result: {task_completed_successfully}')
        if task_for_validation['type'] == 'Fight':
            task_completed_successfully = evaluate_slay(self.character_name, task_for_validation['target'], n=900)
            self.logger.log(f'Validation on environment result: {task_completed_successfully}')
        return task_completed_successfully

    def _agent_prepare(self) -> None:
        self.agent_state = AgentState.SUBTASK_DECOMPOSITION
        self.reset_decomposition_attempts()
        self.reset_decomposition_attempts(subtask = True)

    def task_to_subtask_decomposition(self, task: dict) -> dict:
        """
        Execute task to subtask decomposition with logging.
        
        This is a wrapper method that handles the task to subtask decomposition
        process with proper logging and state management.
        
        Args:
            task: Dictionary containing task information
            
        Returns:
            Dictionary containing decomposed subtasks and metadata
        """
        self.logger.log(f'Task to subtasks decomposition')
        subtasks = self.task_to_subtasks(task)
        self.logger.log( f'Remaining task to subtasks decomposition attempts: {self.subtasks_decomposition_attempts_limit - self.current_subtask_decomposition_attempts}')
        self._log_plan(subtasks)
        return subtasks

    def subtask_to_actions_decomposition(self, subtask: dict) -> dict:
        """
        Execute subtask to actions decomposition with logging.
        
        This is a wrapper method that handles the subtask to actions decomposition
        process with proper logging and state management.
        
        Args:
            subtask: Dictionary containing subtask information
            
        Returns:
            Dictionary containing decomposed actions and metadata
        """
        subtask = self._ensure_missing_fields(subtask)
        action_plan = self.subtasks_to_actions(subtask)
        self._log_plan(action_plan)
        return action_plan

    def _agent_run_results_finalize(self, final_plan: list, evaluation_success_environment: bool) -> tuple[list, tuple[bool, bool]]:
        self.logger.log('CHARACTER FINAL STATE')
        self.logger.log(str(self.character_info.action()))
        self.logger.log(f'Actions plan: {final_plan}')
        self.logger.stop_logger()
        evaluation_success_agent = True
        if self.agent_state == AgentState.FAILED:
            evaluation_success_agent = False
            return final_plan, (evaluation_success_agent, evaluation_success_environment)
        elif self.agent_state == AgentState.FINISHED:
            return final_plan, (evaluation_success_agent, evaluation_success_environment)
        else:
            evaluation_success_agent = False
            return [], (evaluation_success_agent, evaluation_success_environment)

    def run(self, task: dict, eval_on_env: bool = False) -> tuple[list, tuple[bool, bool]]:
        """
        Run the complete agent workflow to solve a provided task.
        
        This is the main entry point for task execution. The agent will:
        1. Decompose the task into subtasks
        2. Decompose each subtask into actions
        3. Optionally execute actions and validate results
        4. Return the final plan and success indicators
        
        Args:
            task: Dictionary containing task information with keys:
                  - type: Task type ('Craft' or 'Fight')
                  - target: Target item or monster name
                  - quantity: Quantity to craft/fight
            eval_on_env: Whether to evaluate task completion in the environment
                        (requires execute_actions_chains to be True)
                        
        Returns:
            Tuple containing:
                - List of action plans for each subtask
                - Tuple of (agent_success, environment_success) booleans
                
        Raises:
            ValueError: If eval_on_env is True but execute_actions_chains is False
        """
        if eval_on_env and not self.execute_actions_chains:
            raise ValueError("Can't evaluate actions on environment with 'execute_actions_chains' set to False'")
        self._agent_prepare()
        subtasks = None
        final_plan = []
        while self.agent_state not in [AgentState.FINISHED, AgentState.FAILED]:
            if self.agent_state == AgentState.SUBTASK_DECOMPOSITION:
                subtasks = self.task_to_subtask_decomposition(task)
                if self.agent_state == AgentState.FAILED:
                    break
                if len(subtasks['subtasks']) == 0 and self.agent_state != 'FA':
                    self.agent_state = AgentState.SUBTASK_DECOMPOSITION
            if self.agent_state == AgentState.ACTION_DECOMPOSITION:
                self.logger.log('Subtasks to actions decomposition')
                subtasks_num = len(subtasks['subtasks'])
                executed_ok = True
                for i in range(subtasks_num):
                    subtask = subtasks['subtasks'][i]
                    action_plan = self.subtask_to_actions_decomposition(subtask)
                    if self.agent_state == AgentState.SUBTASK_DECOMPOSITION:
                        executed_ok = False
                        error_text = self._log_subtask_decomposition_failure(subtask)
                        task['previous_subtasks_critique'] = error_text
                        final_plan.clear()
                        break
                    final_plan.append(action_plan)
                if executed_ok:
                    self.agent_state = AgentState.FINISHED

        env_success_eval = True
        if eval_on_env:
            env_success_eval = self.evaluate_task_completion(task)
        return self._agent_run_results_finalize(final_plan, env_success_eval)










