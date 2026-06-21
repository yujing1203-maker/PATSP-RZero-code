import os
import json
from time import time
from dotenv import load_dotenv
from typing import List, Dict, Union, Optional, Tuple

from taskgen import strict_json_async, Agent, Function
from agent.task_knowledge import TaskKnowledge
from agent.agents_description.curriculum import get_curriculum_desc, get_global_context_curriculum
from agent.agents_description.fight_analytic import get_fight_analytic_desc, get_global_context_fight
from agent.agents_description.decomposer import decomposer_desc, get_global_context_decomposer
from agent.agents_description.mapper import mapper_desc, get_global_context_mapper
from agent.agents_description.crafter import crafter_desc, get_global_context_crafter
from agent.agents_description.critic import critic_desc
from agent.agents_description.action import get_action_prompt

from utils.api import ArtifactsApi
from utils.reward import compute_episode_reward

load_dotenv()
class A2Agent:
    def __init__(self, api: ArtifactsApi, model_name='gpt-4.1', use_openrouter=False, verbose=False):
        """
        Initialises the A2Agent.
        
        Parameters:
        api (ArtifactsApi): The ArtifactsAPI object to interact with the game.
        model_name (str): The model name to use for LLM calls. Defaults to 'gpt-4.1'.
        use_openrouter (bool): Whether to use the OpenRouter API for LLM calls. Defaults to False.
        verbose (bool): Whether to print debug messages. Defaults to False.
        """

        self.model_name = model_name
        self.orouter = use_openrouter
        self.tokens = 0
        self.api = api
        self.verbose = verbose
    
    def _llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a response from a language model given system and user prompts.

        Parameters:
        system_prompt (str): The prompt intended for the system to provide context or instructions.
        user_prompt (str): The prompt provided by the user to guide the response generation.

        Returns:
        str: The generated response content from the language model.
        """

        from openai import OpenAI
        
        if self.orouter:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key = os.getenv('OPENROUTER_API_KEY')
            )
        else:
            client = OpenAI(
                api_key = os.getenv('OPENAI_API_KEY')
            )
            
        response = client.chat.completions.create(
            model=self.model_name,
            temperature = 0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout = 60
        )
        self.tokens += response.usage.total_tokens
        return response.choices[0].message.content
    
    async def _llm_async(self, system_prompt: str, user_prompt: str) -> str:
        """
        Asynchronously generates a response from a language model using system and user prompts.

        Parameters:
        system_prompt (str): The prompt intended for the system to provide context or instructions.
        user_prompt (str): The prompt provided by the user to guide the response generation.

        Returns:
        str: The generated response content from the language model.
        """

        from openai import AsyncOpenAI
        
        if self.orouter:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key = os.getenv('OPENROUTER_API_KEY')
            )
        else:
            client = AsyncOpenAI(
                api_key = os.getenv('OPENAI_API_KEY')
            )
            
        response = await client.chat.completions.create(
            model=self.model_name,
            temperature = 0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout = 60
        )
        
        self.tokens += response.usage.total_tokens
        return response.choices[0].message.content
            
    def _safe_exec(self, code: str) -> str:
        '''Safe run of the code written by an agent'''
        import sys
        import io

        # Capture the output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Safe environment to execute the user code
            allowed_globals = {
                '__builtins__': {
                    'print': print,
                    'range': range,
                    'len': len,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'any': any,
                    'all': all,
                    'sorted': sorted,
                    'type': type,
                    'isinstance': isinstance,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'move': self.api.move,
                    'slay': self.api.slay,
                    'equip': self.api.equip_item,
                    'unequip': self.api.unequip_item,
                    'gather': self.api.gather,
                    'craft': self.api.craft,
                    'buy': self.api.buy
                }
            }

            safe_locals = {}

            exec(code, allowed_globals, safe_locals)
            output = sys.stdout.getvalue()
        except Exception as e:
            output = f"Error: {e}"
        finally:
            # Restore the original stdout
            sys.stdout = old_stdout

        return output
    
    @staticmethod
    def parse_global_task(task_data) -> Tuple[str, str, Optional[str]]:
        """
        Parses the global task information from the provided task data.

        Parameters:
        task_data (dict): A dictionary containing task details. It must have either 
                          'monster_name' or 'item' as a key.

        Returns:
        tuple: A tuple containing:
            - global_task (str): A description of the main task, formatted for execution.
            - task_prompt (str): Instructions on how the task should be approached, 
            including requirements for crafting and equipping items.
            - duplicate_prompt (Optional[str]): Additional constraints or guidelines regarding 
            item duplication in equipment slots, particularly for rings.
        """

        if 'monster_name' in task_data:
            final_objective = task_data['monster_name']
            global_task = f'Slay 1 {final_objective}.'
            task_prompt = f'''\nIf you need to slay a monster, make sure character can defeat monster. The character may require to craft and equip items to do it.
            During your work do not make assumptions about the outcome of the fight. You must call fight analytic agent, he can calculate the fight success.
            
            End your turn only after you confirmed the winning of the fight, there is always a solution, double check the fight analytic output, it can be wrong.
            There is always a way to complete the global task, if fight analytic states that it is impossible to use any available craftable items, run it again.\n'''
            
            missing_items = task_data['missing_items']
            if 'ring1_slot' in missing_items and 'ring2_slot' in missing_items and missing_items['ring1_slot'] != missing_items['ring2_slot']:
                duplicate_prompt = 'The character can wear duplicate items in multiple slots of the same type, except the ring slots, they should be unique.'
            else:
                duplicate_prompt = 'The character can wear duplicate items in multiple slots of the same type' 
        elif 'item' in task_data:
            final_objective = task_data['item']
            global_task = f'Craft 1 {final_objective}.'
            task_prompt = ''
            duplicate_prompt = None
        else:
            raise ValueError('Unknown task type')
        
        return global_task, task_prompt, duplicate_prompt

    def init_agents(self,
                     char_name: str,
                     global_task: str,
                     task_prompt: str,
                     duplicate_prompt: Optional[str],
                     env_knowledge: TaskKnowledge
                    ) -> None:
        '''
        Initializes all agents and assigns their functions for the current benchmark task.

        Parameters:
        char_name (str): Name of the character to be used in the task.
        global_task (str): Description of the main task, formatted for execution.
        task_prompt (str): Instructions on how the task should be approached, including requirements for crafting and equipping items.
        duplicate_prompt (Optional[str]): Additional constraints or guidelines regarding item duplication in equipment slots, particularly for rings.
        env_knowledge (TaskKnowledge): Object containing environment knowledge for the current task, such as available items, monsters, and starting equipment.
        '''
        char_status = None
        known_recipes = []
        known_locations = []
        print(env_knowledge.knowledge)
        if duplicate_prompt is not None:
            analytic_desc = get_fight_analytic_desc('\n' + duplicate_prompt + '\n')
            task_prompt += '\n' + duplicate_prompt
        else:
            analytic_desc = get_fight_analytic_desc()
        
        # Main curriculum agent
        curriculum_desc = get_curriculum_desc(char_name, task_prompt)
        self.curriculum = Agent(
            agent_name='Curriculum agent',
            agent_description=curriculum_desc,
            max_subtasks=10,
            summarise_subtasks_count=8,
            shared_variables = {'char_status': char_status,
                                'global_task': global_task,
                                'monsters': env_knowledge.get_monsters(),
                                'craft_equip': env_knowledge.get_equipment(),
                                'start_equip': env_knowledge.get_starting_equipment(),
                                },
            get_global_context = get_global_context_curriculum,
            llm=self._llm,
            verbose=self.verbose)
        # Wrapping bound methods and assigning functions to the curriculum agent
        def get_items() -> List[Dict]:
            '''
            Return all available items for the current benchmark task in a more readable format.
            '''
            return env_knowledge.get_items()
        
        def get_monsters() -> List[Dict]:
            '''
            Returns a list of all available monsters for the current benchmark task in a readable format.
            '''
            return env_knowledge.get_monsters()
        curriculum_fn = [
            Function(external_fn = get_items),
            Function(external_fn = get_monsters),
        ]
        self.curriculum.assign_functions(curriculum_fn)
        
        # Task giver helper agent to calculate fight outcome
        fight_analytic = Agent(
            agent_name='Fight analytic',
            agent_description=analytic_desc,
            get_global_context=get_global_context_fight,
            max_subtasks=10,
            summarise_subtasks_count=5,
            llm=self._llm,
            verbose=self.verbose)
        self.curriculum.assign_agent([fight_analytic])
        # Agent for task decomposition
        self.decomposer =  Agent(
            agent_name='Task decomposer',
            agent_description=decomposer_desc,
            max_subtasks=10,
            shared_variables={'recipes': known_recipes, 'map': known_locations},
            get_global_context=get_global_context_decomposer,
            summarise_subtasks_count=8,
            llm=self._llm,
            verbose=self.verbose)
        # Two helping agents for the decomposer
        mapper = Agent(
            agent_name='Map expert',
            agent_description=mapper_desc,
            shared_variables={'map': known_locations},
            get_global_context=get_global_context_mapper,
            max_subtasks=10,
            summarise_subtasks_count=8,
            llm=self._llm,
            verbose=self.verbose)
        
        # Wrapping bound methods and assigning functions to the map expert
        def get_map_entities(tile_type: str) -> Dict[str, str]:
            '''
            Returns a list of all entities names according to the tile_type.
            Possible tile_types: 'monster', 'resource', 'workshop'
            '''
            return env_knowledge.get_map_entities(tile_type)
        def get_coordinates(shared_variables: Dict, entity_list: List[str]) -> Union[List[Dict], str]:
            '''
            Takes in entity_list and returns a coordinates of entities with that name.
            Saves the coordinates in shared_variables['map'].
            '''
            return env_knowledge.get_coordinates(shared_variables, entity_list)
        
        mapper_fn = [
            Function(external_fn = get_map_entities),
            Function(external_fn = get_coordinates)
        ]
        mapper.assign_functions(mapper_fn)
        crafter = Agent(
            agent_name='Recipe expert',
            agent_description=crafter_desc,
            shared_variables={'recipes': known_recipes},
            get_global_context=get_global_context_crafter,
            max_subtasks=10,
            summarise_subtasks_count=8,
            llm=self._llm,
            verbose=self.verbose)
        
        # Wrapping bond method and assigning function to the recipe expert
        def get_recipe(shared_variables: Dict, item_name: str) -> str:
            '''
            Returns a recipe of an item_name item with all required ingredients.
            Saves the recipe in shared_variables['recipes'].
            '''
            return env_knowledge.get_recipe(shared_variables, item_name)
        
        crafter_fn = [
            Function(external_fn = get_recipe)
        ]      
        crafter.assign_functions(crafter_fn)
        self.decomposer.assign_agent([mapper, crafter])
        
        self.critic = self.critic_agent
        self.action = self.action_agent

        self.curriculum.reset()
        self.decomposer.reset()
        mapper.reset()
        crafter.reset()
    
    
    async def critic_agent(self, tasks_given_decomp: Union[str, dict], correction_attempts: int=5) -> Dict:
        '''
        This function is a wrapper for the critic agent. It takes in the tasks 
        given to the decomposer and asks the critic agent to correct any errors.

        If the critic agent finds an error, it will ask the decomposer to correct
        the error and then re-run the critic agent. This is done until no more
        errors are found or the maximum number of correction attempts is reached.

        Parameters:
        tasks_given_decomp (Union[str, dict]): The tasks given by the
            decomposer.
        correction_attempts (int): The maximum number of correction attempts. Defaults to 5.

        Returns:
        Dict: The corrected tasks given by the decomposer.
        '''
        error_count = 0
        while True:
            corrections = await strict_json_async(
                system_prompt=critic_desc + f'\nKnown recipes: {self.decomposer.shared_variables["recipes"]}\nKnown locations: {self.decomposer.shared_variables["map"]}',
                user_prompt = str(tasks_given_decomp),
                output_format = {
                    'comment': 'Concise comment about where the errors was found, type: str',
                    'user_prompt': 'Corrected plan, type: dict',
                    'need_more_info': 'Should be True, if you can\'t correct the plan with the given information, type: bool'
                },
                llm = self._llm_async
            )
            # Workaround if taskgen fails to parse the output
            need_info = corrections.get('###need_more_info###', corrections['need_more_info'])
            if need_info:
                if self.verbose:
                    print('[CRITIC] Error found: ', corrections['comment'])
                corrections_made = self.decomposer.run(f'Error found: {corrections["comment"]}. Analyze the corrections and return corrected plan with related comments.')
                tasks_given_decomp = self.decomposer.reply_user()
                error_count += 1
                if error_count == correction_attempts:
                    raise Exception(f"Can't correct the plan in {correction_attempts} attempts")
            else:
                if self.verbose:
                    print(f"[CRITIC] No errors found. Corrected plan:\n{corrections['user_prompt']}")
                break
            
        tasks_given_decomp = corrections['user_prompt']
        return tasks_given_decomp

    async def action_agent(self, prompt) -> Tuple[str, str]:
        '''
        This function is a wrapper for the action agent. It takes in a prompt,
        runs it through the LLM to generate a sequence of code, and then runs
        that code in a safe environment.

        Parameters:
        prompt (str): The prompt to generate code for.

        Returns:
        Tuple[str, str]: A tuple of the generated code and any error message
            if there was an error.
        '''
        code = await self._llm_async('You are a helpful assistant.', prompt)
        code_start_idx = code.rfind('Final answer:') + len('Final answer: ')
        code = code[code_start_idx:]
        code = code.replace('```python\n','').replace('```','')
        if self.verbose:
            print(f'Generated code:\n{code}')
        
        code_error = None
        output = self._safe_exec(code)
        
        if 'Error' in output:
            if self.verbose:
                print(f'[ACTION] Error in code: {output}')
            code_error = output
        elif self.verbose:
            print(f'[ACTION] Successfully executed code.')
            if len(output):
                print(f'[ACTION] Output: {output}')
            
        return code, code_error
    
    async def run(self, correction_attempts: int=5, difficulties: Optional[List[int]]=None, tasks_file='combined_tasks8_small.json') -> None:
        '''
        Runs the agent system through all tasks in the given task file.

        Parameters:
        correction_attempts (int, optional): Number of critic agent correction attempts. Defaults to 5.
        difficulties (Optional[List[int]], optional): List of difficulties to run. Defaults to None to run all difficulties.
        tasks_file (str, optional): Path to the json file with tasks. Defaults to 'combined_tasks8_small.json'.
        '''
       
        results_file = f'{self.model_name}_results_benchmark.json'
        # Loading tasks
        tasks_path = os.path.normpath(f'./tasks/{tasks_file}')
        with open(tasks_path) as f:
            json_data = json.load(f)
        # Loading old logs if exists
        logs_path = os.path.normpath(f'./results/{results_file}')
        if os.path.exists(logs_path):
            with open(logs_path) as f:
                results = json.load(f)
        else:
            results = {}
            
        for difficult_level, data in json_data.items():
            if difficulties is not None:
                if int(difficult_level) not in difficulties:
                    continue
            if difficult_level not in results.keys():
                results[difficult_level] = {}
                
            for task_num, task_data in enumerate(data):
                # If task is already completed, skip it
                if f'task_{task_num}' in results[difficult_level].keys():
                    status = results[difficult_level][f'task_{task_num}']['completion_status']
                    if status is not None:
                        if self.verbose:
                            print(f'Task {task_num} at difficulty level {difficult_level} already completed with status {status}, skipping...')
                        continue
                    elif self.verbose:
                        print(f'Task {task_num} at difficulty level {difficult_level} was interrupted, restarting...')
                
                # Initial log
                results[difficult_level][f'task_{task_num}'] = {
                    'task': task_data,
                    'logs': {},
                    'completion_status': None,
                    'start_time': time(),
                    'start_tokens': self.tokens
                }
                # Parsing character data and creating custom character
                char_stat = task_data['character']
                char_name = char_stat.pop('name', 'Hero')
                char_skin = char_stat.pop('skin', 'men1')

                self.api.delete_char(char_name)
                self.api.create_custom_char(char_name, char_stat, char_skin)
                
                # Getting current instance knowledge
                knowledge = TaskKnowledge(task_data['task_info'], self.api)
                
                # Parsing global task
                global_task, task_prompt, duplicate_prompt = self.parse_global_task(task_data)
                self.init_agents(char_name, global_task, task_prompt, duplicate_prompt, knowledge)
                self.curriculum.shared_variables['char_status'] = self.api.check_status(char_name)
                if self.verbose:
                    print(f'Starting difficulty level: {difficult_level}, task number {task_num}\nGlobal goal: {global_task}')
                    print('Character status: ')
                    print(self.curriculum.shared_variables['char_status'])
                
                curriculum_output = None
                tasks_given = None
                decomposer_output = None
                tasks_given_decomp = None
                code = None
                code_error = None 
                # Curriculum agent offers next goal
                curriculum_output = self.curriculum.run(f'Provide next task.')
                tasks_given = str(self.curriculum.reply_user())
                
                if global_task[:-1].lower() not in tasks_given.lower():
                    if self.verbose:
                        print(f'[CURRICULUM] Curriculum tasks missing global task:\n{tasks_given}')
                else:
                    # Decomposition of global tasks by decomposer
                    decomposer_output = self.decomposer.run(f'Decompose: {tasks_given}')
                    tasks_given_decomp = self.decomposer.reply_user()
                    # Error correction
                    tasks_given_decomp = await self.critic(tasks_given_decomp, correction_attempts=correction_attempts)
                    # Parsing of tasks for action agent
                    tasks_list = []
                    subtasks_list = []
                    for char, tasks in tasks_given_decomp.items():
                        for task in tasks:
                            tasks_list.append(task['task'])
                            subtasks_list.extend(task['subtasks'].values())
                    
                    if self.verbose:
                        print(f'[ACTION] Final input subtasks list:\n{subtasks_list}')        
                    action_prompt = get_action_prompt(char_name, subtasks_list, global_task)
                    code, code_error = await self.action(action_prompt)
                    
                # Writing logs
                results[difficult_level][f'task_{task_num}']['logs'] = {
                    'curriculum_output': curriculum_output,
                    'tasks_given': tasks_given,
                    'decomposer_output': decomposer_output,
                    'tasks_given_decomp': tasks_given_decomp,
                    'code': code,
                    'code_error': code_error,
                    'char_status': self.api.check_status(char_name),
                    'turn_end_tokens': self.tokens,
                }
                
                # Checking global task completion based on environment logs
                api_logs = self.api.get_logs(char_name, 1000)
                for i, log in enumerate(api_logs):
                    if log['action_type'] == 'create_custom_character':
                        api_logs = api_logs[:i+1]
                        break
                reward, info_reward, global_status = compute_episode_reward(task_data, api_logs)
                if self.verbose:
                    print('Checking global task completion...')
                    print(global_status, end='\n\n')
                    
                # Computing reward and saving logs
                results[difficult_level][f'task_{task_num}']['fastapi_logs'] = api_logs
                results[difficult_level][f'task_{task_num}']['completion_status'] = global_status
                results[difficult_level][f'task_{task_num}']['reward'] = reward
                results[difficult_level][f'task_{task_num}']['info_reward'] = info_reward
                results[difficult_level][f'task_{task_num}']['end_time'] = time()
                
                with open(logs_path, 'w') as f:
                    json.dump(results, f, indent=4)
                
                
                