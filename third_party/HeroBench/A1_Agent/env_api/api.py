import requests
import itertools
from typing import List, Dict, Optional, Tuple, Union
import time

def create_char(name:str, skin='men1') -> dict:
    url = 'http://127.0.0.1:8000/characters/create'
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}

    payload = {'name': name,
               'skin': 'men1'}
    response = requests.post(url, headers=headers, json=payload).json()
    return response

def create_custom_char(name:str, char_data:dict, skin:str='men1',) -> dict:
    url = 'http://127.0.0.1:8000/characters/create_custom'
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}

    payload = {'name': name,
               'skin': skin,
               'char_data': char_data
               }
    response = requests.post(url, headers=headers, json=payload).json()
    if 'error' in response:
        raise Exception(f"Error code: {response['error']['code']}, message: {response['error']['message']}")
    return response
    
def delete_char(name:str) -> dict:
    url = 'http://127.0.0.1:8000/characters/delete'
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}
    
    payload = name
    response = requests.post(url, headers=headers, json=payload).json()
    return response

def reset_char(name:str) -> dict:
    delete_char(name)
    return create_char(name)

# Full map dictionary with all parameters from the server
def get_map_dict(required_type=None):
    url = "http://127.0.0.1:8000/maps/"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    response_data = response.json()  # Fetch map data from the API
    resource_map = []
        # Process each tile in the response data
    for tile in response_data:
        if tile['content']:  # Check if there is any content on the tile
            name = tile['name']  # Name of the location
            x = tile['x']
            y = tile['y']
            content_type = tile['content']['code']
            object_type = tile['content']['type']
            if content_type == 'mining':
                content_type = 'smeltery'
            # if required_type == object_type or required_type == None:
            resource_map.append({
                'name': content_type.replace('_', ' '),
                'type': object_type.replace('_', ' '),
                'x': x,
                'y': y,
            })

    return resource_map

def get_item_dict():
    import requests, itertools
    items = []
    url = "http://127.0.0.1:8000/items/"
    headers = {"Accept": "application/json"}
    items = requests.get(url, headers=headers).json()
    return items

def get_monster_dict():
    import requests, itertools
    url = "http://127.0.0.1:8000/monsters/"
    headers = {"Accept": "application/json"}
    monsters = requests.get(url, headers=headers).json()
    return monsters

def get_recipes():
    items_list = get_item_dict()
    crafting_tree = []

    for item in items_list:
        item_to_add = {'name': item['name'].lower()}
        if item['craft']:
            item_to_add["crafting components"] = item['craft']['items']
            item_to_add['required skill'] = f"{item['craft']['skill']} level {item['craft']['level']}"
        else:
            if item['subtype']:
                if item['subtype'] == 'mob':
                    item_to_add["crafting components"] = f'Lootable from mob on level {item["level"]}'
                else:
                    item_to_add["crafting components"] = f'Gatherable, item subtype {item["subtype"]}, item level {item["level"]}'
            else:
                item_to_add["crafting components"] = f'Gatherable/Lootable, item level {item["level"]}'
        crafting_tree.append(item_to_add)
    return crafting_tree

def get_items():
    items_list = get_item_dict()
    result = []

    for item in items_list:
        item_to_add = dict(
            name = item['name'].lower(),
            type = item['type'],
            level = item['level'],
        )
        if len(item['subtype']):
            item_to_add['subtype'] = item['subtype']
        if len(item['effects']):
            item_to_add['effects'] = item['effects']
        if len(item['description']):
            item_to_add['description'] = item['description']
        result.append(item_to_add)
    return result

def get_monsters():
    monsters_list = get_monster_dict()
    result = []

    for monster in monsters_list:
        monster_to_add = dict(
            name = monster['name'].lower(),
            level = monster['level'],
            attack = dict(
                fire = monster['attack_fire'],
                earth = monster['attack_earth'],
                water = monster['attack_water'],
                air = monster['attack_air']
            ),
            defence = dict(
                fire = monster['res_fire'],
                earth = monster['res_earth'],
                water = monster['res_water'],
                air = monster['res_air']
            ),
        )
        result.append(monster_to_add)
    return result
    
def get_map_entities(tile_type: str) -> Dict[str, str]:
    '''
    Returns a list of all entities names according to the tile_type.
    Possible tile_types: 'monster', 'resource', 'workshop'
    '''
    map = get_map_dict()
    result = []
    for tile in map:
        if tile['type'] == tile_type:
            result.append((tile['name'], tile['type']))
    return result

def get_coordinates(shared_variables, entity_list:List[str]) -> Union[List[Dict], str]:
    '''
    Takes in entity_list and returns a coordinates of entities with that name.
    '''
    map = get_map_dict()
    result = {}
    for entity in entity_list:
        entity = entity.replace('_', ' ')
        coordinates = list(filter(lambda x: x['name'] == entity, map))
        if coordinates:
            coordinates = coordinates[0]
            if coordinates not in shared_variables['map']:
                shared_variables['map'].append(coordinates)
            result[coordinates['name']] = {'x': coordinates['x'], 'y': coordinates['y']}
        else:
            result[entity] = 'Entity not found on map, try another name'
    return result

def get_recipe(shared_variables, item_name: str) -> str:
    '''
    Returns a recipe of an item_name item with all required ingredients.
    '''
    recipes = get_recipes()
    item_name = item_name.replace('_', ' ').lower()
    try:
        result = list(filter(lambda x: x['name'] == item_name, recipes))[0]
    except:
        result = 'Item not found, try to rewrite the item name'
        return result
    if result not in shared_variables['recipes']:
        shared_variables['recipes'].append(result)
    return result

def get_item_by_level(min_level: int=1, max_level: int=40) -> List[Dict]:
    '''
    Return all available items sorted with level between min_level and max_level.
    '''
    items = get_items()
    result = [i for i in items if min_level <= i['level'] <= max_level]
    return result

def get_monster_by_level(min_level: int=1, max_level: int=45) -> List[Dict]:
    '''
    Return all available monsters sorted with level between min_level and max_level.
    '''
    monsters = get_monsters()
    result = [i for i in monsters if min_level <= i['level'] <= max_level]
    return result

# def sleep(response_json: str) -> None:
#     '''
#     Allows to wait for some amount of time before performing an action. Useful to wait until cooldown finish.
#     '''
#     cooldown = response_json['cooldown']['total_seconds']
#     cooldown = int(cooldown)
#     time.sleep(cooldown)

def move(char_name: str, x: int, y: int) -> bool:
    '''
    Moves character char_name to specified coordinates x and y.
    '''
    url = f"http://127.0.0.1:8000/my/{char_name}/action/move"
    payload = {
        'x': x,
        'y': y
    }
    headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers)
    response_j = response.json()
        
    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 404:
            raise Exception('No tile with such coodrinates')
        elif err_code == 490:
            return 'You already arrived at destination'
        else:
            raise Exception('Unexpected error, try again')
    else:
        time.sleep(0.1)
        return True

def gather(char_name: str, quantity: int=1) -> bool:
    '''
    Gather quantity resource on tile by char_name.
    '''
    url = f"http://127.0.0.1:8000/my/{char_name}/action/gathering"
    payload = str(quantity)
    headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
            }
    response = requests.post(url, json=payload, headers=headers)
    response_j = response.json()

    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 598:
            raise Exception('No resource on this tile')
        elif err_code == 493:
            raise Exception('You cant mine this resource, skill level too low')
        elif err_code == 497:
            raise Exception('Your inventory is full')
        else:
            raise Exception('Unexpected error, try again')
    else:
        time.sleep(0.1)
        return True


def craft(char_name: str, item_name: str, quantity: int=1) -> bool:
    '''
    Craft quantity item_name item by character char_name.
    '''
    if item_name.endswith('bar'):
        item_name = item_name[:-4]
    item_name = item_name.replace(" ", "_").lower()
    url = f"http://127.0.0.1:8000/my/{char_name}/action/crafting"
    payload = {
        "code": f"{item_name}",
        "quantity": quantity
            }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
                }
    response = requests.post(url, json=payload, headers=headers)
    response_j = response.json()

    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 404:
            raise Exception('No such craft recepie')
        elif err_code == 478:
            raise Exception('Missing item or insufficient quantity')
        elif err_code == 486:
            raise Exception('An action is already in progress')
        elif err_code == 493:
            raise Exception("You can't craft this item, improve your skill")
        elif err_code == 497:
            raise Exception('Your inventory is full')
        elif err_code == 498:
            raise Exception('Character not found')
        elif err_code == 499:
            raise Exception('Character in cooldown')
        elif err_code == 598:
            raise Exception('No workshop on the current position of character')
        else:
            raise Exception('Unexpected error, try again')
    else:
        time.sleep(0.1)
        return True

def unequip_item(char_name: str, slot_name: str) -> bool:
    '''
    Unequip item from slot slot_name of character char_name.
    Possible slots:
    weapon, shield, helmet, body_armor, leg_armor, boots, ring1, ring2, amulet, artifact1, artifact2, artifact3, consumable1, consumable2
    '''
    url = f"http://127.0.0.1:8000/my/{char_name}/action/unequip"
    payload = {
        "slot": f"{slot_name}",
        "quantity": 1
                }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
            }
    response = requests.post(url, json=payload, headers=headers)
    response_j = response.json()

    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 404:
            raise Exception('No such item in your equipment')
        elif err_code == 478:
            raise Exception('Missing item or insufficient quantity')
        elif err_code == 491:
            raise Exception('This slot already empty, update your State information')
        elif err_code == 497:
            raise Exception('Your inventory is full')
        else:
            raise Exception('Unexpected error, try again')
    else:
        time.sleep(0.1)
        return True 
    

def equip_item(char_name: str, slot_name: str, item_name:str) -> bool:
    '''
    Equip item item_name to slot slot_name of characer char_name.
    Possible slots:
    weapon, shield, helmet, body_armor, leg_armor, boots, ring1, ring2, amulet, artifact1, artifact2, artifact3, consumable1, consumable2
    '''
    url = f"http://127.0.0.1:8000/my/{char_name}/action/equip"
    slot_name = slot_name.replace(" ", "_").lower()
    item_name = item_name.replace(" ", "_").lower()
    payload = {
        "code": f"{item_name}",
        "slot": f"{slot_name}",
        "quantity": 1
        }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
                }

    response = requests.post(url, json=payload, headers=headers)
    response_j = response.json()        

    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 404 or err_code == 478:
            raise Exception('Cant equip item,no such item in your inventory')
        elif err_code == 485:
            raise Exception('This item already equiped')
        elif err_code == 472:
            raise Exception('This item is not valid for this slot')
        elif err_code == 491:
            raise Exception('This slot is not empty, empty it, then try equiping item again')
        elif err_code == 496:
            raise Exception('Character level to low')
        elif err_code == 422:
            raise Exception('Provided input not valid for this action')
        else:
            raise Exception('Unexpected error, try again')
    else:
        time.sleep(0.1)
        return True
    

def slay(char_name: str) -> bool:
    '''
    Try to kill a monster on tile with character char_name.
    '''
    url = f"http://127.0.0.1:8000/my/{char_name}/action/fight"
    headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
                }
    response = requests.post(url, headers=headers)
    response_j = response.json()        

    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 598:
            raise Exception('Mob not found on this tile')
        else:
            raise Exception('Unexpected error, try again')
    else:
        time.sleep(0.1)
        fight_result = response_j['fight']['result']
        if fight_result == 'lose':
            return False
        else:
            return True
    
def give(char_from: str, char_to: str, item_name: str, quantity: int=1) -> bool:
    '''
    Give quantity item_name item from char_from to character char_to.
    '''
    url = f"http://127.0.0.1:8000/my/{char_from}/action/give"
    payload = {
        "recepient": char_to,
        "code": item_name.replace(" ", "_").lower(),
        "quantity": quantity
                }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
                    }
    response = requests.post(url, json=payload, headers=headers)
    response_j = response.json()        

    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 478:
            raise Exception('Insufficient quantity of the item in character inventory.')
        elif err_code == 498:
            raise Exception('Giver character not found')
        elif err_code == 598:
            raise Exception('Recepient character not found')
        elif err_code == 422:
            raise Exception('Provided input not valid for this action')
        else:
            raise Exception('Unexpected error, try again')
    else:
        time.sleep(0.1)
        return True

def check_inventory(char_name: str) -> str:
    '''
    Returns information about character char_name inventory.
    '''
    url = f"http://127.0.0.1:8000/characters/{char_name}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    response_j = response.json()
    
    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 404:
            raise Exception('Character not found')
        elif err_code == 422:
            raise Exception('Provided input not valid for this action')
        else:
            raise Exception('Unexpected error, try again')
    else:
        return response_j['inventory']

def buy(char_name: str, item_name: str, quantity: int=1) -> bool:
    '''
    Buy quantity item_name from Grand Exchange with character char_name.
    '''
    url = f"http://127.0.0.1:8000/my/{char_name}/action/buy"
    payload = {
        "code": item_name.replace(" ", "_").lower(),
        "quantity": quantity
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers)
    response_j = response.json()        

    if 'error' in response_j:
        err_code = response_j['error']['code']
        err_code = int(err_code)

        if err_code == 404:
            raise Exception('Item not found in Grand Exchange')
        elif err_code == 422:
            raise Exception('Provided input not valid for this action')
        elif err_code == 478:
            raise Exception('Quantity must be at least 1')
        elif err_code == 498:
            raise Exception('Character not found')
        elif err_code == 598:
            raise Exception('No Grand Exchange on the current position of character')
        else:
            raise Exception('Unexpected error, try again')
    else:
        time.sleep(0.1)
        return True
    
def check_item(char_name: str, item_name: str) -> int:
    '''
    Returns the amount of the item_name in inventory of char_name character.
    '''
    inventory = check_inventory(char_name)
    item_name = item_name.replace(' ', '_').lower()
    for item in inventory:
        if item['code'] == item_name:
            return int(item['quantity'])
    return 0

def check_position(char_name: str) -> Tuple[int, int]:
    '''
    Returns tuple with the char_name current position coordinates x and y.
    '''
    url = f"http://127.0.0.1:8000/characters/{char_name}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers).json()
    return (response['x'], response['y'])

def check_status(char_name: str) -> str:
    '''
    Returns all available information about character.
    '''
    url = f"http://127.0.0.1:8000/characters/{char_name}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers).json()

    if 'error' in response:
        raise Exception('Character name not found')
    response['character_hp'] = f"{response.pop('hp')}"
    response['character_level'] = response.pop('level')
    response['character_xp'] = f"{response.pop('xp')}/{response.pop('max_xp')}"

    response['skills'] = {
        # 'alchemy': {
        #     'level': response.pop('alchemy_level'),
        #     'xp': f"{response.pop('alchemy_xp')}/{response.pop('alchemy_max_xp')}"
        # },
        'cooking': {
            'level': response.pop('cooking_level'),
            'xp': f"{response.pop('cooking_xp')}/{response.pop('cooking_max_xp')}"
        },
        'fishing': {
            'level': response.pop('fishing_level'),
            'xp': f"{response.pop('fishing_xp')}/{response.pop('fishing_max_xp')}"
        },
        'gearcrafting': {
            'level': response.pop('gearcrafting_level'),
            'xp': f"{response.pop('gearcrafting_xp')}/{response.pop('gearcrafting_max_xp')}"
        },
        'jewelrycrafting': {
            'level': response.pop('jewelrycrafting_level'),
            'xp': f"{response.pop('jewelrycrafting_xp')}/{response.pop('jewelrycrafting_max_xp')}"
        },
        'mining': {
            'level': response.pop('mining_level'),
            'xp': f"{response.pop('mining_xp')}/{response.pop('mining_max_xp')}"
        },
        'weaponcrafting': {
            'level': response.pop('weaponcrafting_level'),
            'xp': f"{response.pop('weaponcrafting_xp')}/{response.pop('weaponcrafting_max_xp')}"
        },
        'woodcutting': {
            'level': response.pop('woodcutting_level'),
            'xp': f"{response.pop('woodcutting_xp')}/{response.pop('woodcutting_max_xp')}"
        }
    }
    to_remove = ['skin', 'x', 'y']
    for key in to_remove:
        response.pop(key)

    return response

def get_logs(char_name: str, n: int) -> list[str]:
    '''
    Returns last n logs of char_name character.
    '''
    url = f"http://127.0.0.1:8000/logs/{n}/{char_name}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def evaluate_slay(char_name: str, monster_name: str, n: int=20) -> bool:
    '''
    Returns True if char_name character won fight against monster_name.
    '''
    logs = get_logs(char_name, n)
    logs = [log['log'] for log in logs]
    if f"{char_name} win his fight against {monster_name}." in logs:
        return True
    else:
        return False
    
def evaluate_craft(char_name: str, item_name: str, n: int=20) -> bool:
    '''
    Returns True if char_name character crafted item_name.
    '''
    logs = get_logs(char_name, n)
    for log in logs:
        if f"{char_name} crafts {item_name.lower().replace(' ', '_')}" in log['log']:
            return True 
    return False