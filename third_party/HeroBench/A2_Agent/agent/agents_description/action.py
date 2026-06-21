from typing import List
def get_action_prompt(char_name:str, subtasks_list: List[str], global_task: str) -> str:
    action_prompt = f'''Your task is to write a Python program to acomplish the list of tasks for a game character {char_name}:
    {subtasks_list}
    You can perform the following actions:
    move(character_name: str, x: int, y: int)
    slay(character_name: str)
    equip(character_name: str, slot: str, item_name: str)
    unequip(character_name: str, slot: str)
    Item slot. Allowed values: 'weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', 'boots',
            'ring1', 'ring2', 'amulet', 'artifact1', 'artifact2', 'artifact3', 'consumable1' or 'consumable2'.

    gather(character_name: str, quantity: int = 1)
    craft(character_name: str, item_name: str, quantity: int = 1)
    buy(character_name: str, item_name: str, quantity: int = 1)
    Use codes of items in functions like 'copper_dagger'. You can only use these functions and for loops in your answer, nothing else. Do not include parameter names when calling functions.

    Your global task is {global_task.lower()}.
    Think and write a sequence of actions to do it.
    End your answer with the following: "Final answer: <python code to solve the task>'''
    
    return action_prompt