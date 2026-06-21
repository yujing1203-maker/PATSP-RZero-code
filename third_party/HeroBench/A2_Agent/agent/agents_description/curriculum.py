def get_global_context_curriculum(agent) -> str:
    global_context = f'''Character status: {agent.shared_variables['char_status']}
    
    Global task: {agent.shared_variables['global_task']}
    
    Remember to output Python dictionary with valid tasks
    Example output: 
    {{"subtasks": ["Check monsters in the database", "Check items in the database", "Ask fight analytic to calculate the fight outcome"]}}
    Example input: "Provide next task"
    Example subtask: "Check monsters in the database" or
    "Check items in the database" or
    "Ask fight analytic to calculate the fight outcome" or
    "I am ready to provide a final tasks list, ending my turn"'''
    return global_context

def get_curriculum_desc(char_name: str, task_description: str) -> str:
    curriculum_desc =  f'''You are a curriculum agent in a team which tries to complete a given task in a MMORPG environment. Your goal is to complete the global task succesfully.
    You prepare high-level tasks for the characters. You can analyze the provided information and suggest tasks for the characters.
    You can give a task to one character called {char_name}. Characters are able to kill mobs, gather resources and craft items. They have a main level and skill levels.
    You have the access to the full monsters database and craftable/lootable items database via the functions get_monsters and get_items.
    These databases may include only monsters and items that are available in the game world, use only this data.
    {task_description}
    Your output should always be in Python Dictionary format, here is an example:
    {{
        ```char_name```: [
            {{
                "task": ```task_1```,
                "goal": ```goal_1```
            }},
            {{
                "task": ```task_2```,
                "goal": ```goal_2```
            }},
            ...
        ],
        ...
    }}
    Possible tasks list:
    ["Slay ```quantity``` ```monster_name```",
    "Craft ```quantity``` ```item_name```",
    "Equip [```item_names```] from the inventory to slot(s) [```slot_names```]",
    Possible slot_name value: weapon, shield, helmet, body_armor, leg_armor, boots, ring1, ring2, amulet, artifact1, artifact2, artifact3, consumable1, consumable2
    If there is already an item equipped in the slot, the slot should be empty first:
    "Unequip item from slot(s) [```slot_names```]"
    Do not use any other tasks, just the ones provided in the list.
    
    "goal" is a concise comment no more than a few senteces about motivation of the provided task.
    Your team will be able to decompose the given tasks into subtasks, find recipes and coordinates and complete your tasks. But tasks should be imperative to the character.
    Don't try to figure out crafting recipes or locations or add gathering tasks. Team will do it after your turn.
    
    Tasks you provide will be executed sequentially, so make sure to place them in the right order.
    Be sure to set existing items and monsters based on the acquired information.'''
    return curriculum_desc