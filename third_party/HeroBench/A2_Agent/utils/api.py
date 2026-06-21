import requests
from typing import List, Dict, Optional, Tuple, Union
import time

class ArtifactsApi:
    def __init__(self, api_url: str='http://127.0.0.1:8000', request_delay: float=0.02):
        """
        Initialize the API client.

        Parameters
        ----------
        api_url : str
            The base URL of the API endpoint. Defaults to 'http://127.0.0.1:8000'.
        request_delay : float
            Time to wait between action requests in seconds. Defaults to 0.02 seconds.
        """
        
        self.url = api_url
        self.delay = request_delay
        
    def create_char(self, name: str, skin: str='men1') -> dict:
        '''
        Create a new character with the specified name and skin.

        Parameters
        ----------
        name : str
            The name of the character to be created.
        skin : str, optional
            The skin type of the character. Defaults to 'men1'.

        Returns
        -------
        dict
            A dictionary containing the details of the newly created character.
        '''

        url = self.url + '/characters/create'
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        payload = {'name': name, 'skin': skin}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def create_custom_char(self, name: str, char_data: dict, skin: str='men1',) -> dict:
        '''
        Create a new character with the specified name, skin and character data.

        Parameters
        ----------
        name : str
            The name of the character to be created.
        char_data : dict
            A dictionary containing the character data.
        skin : str, optional
            The skin type of the character. Defaults to 'men1'.

        Returns
        -------
        dict
            A dictionary containing the details of the newly created character.
        '''

        url = self.url + '/characters/create_custom'
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        payload = {'name': name, 'skin': skin, 'char_data': char_data}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
        
    def delete_char(self, name: str) -> dict:
        '''
        Delete a character with the specified name.

        Parameters
        ----------
        name : str
            The name of the character to be deleted.

        Returns
        -------
        dict
            A dictionary containing the response from the server after attempting to delete the character.
        '''

        url = self.url + '/characters/delete'
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        payload = name
        response = requests.post(url, headers=headers, json=payload)
        #response.raise_for_status()
        return response.json()

    def reset_char(self, name: str) -> dict:
        '''
        Reset a character with the specified name.

        Parameters
        ----------
        name : str
            The name of the character to be reset.

        Returns
        -------
        dict
            A dictionary containing the details of the newly created character.
        '''
        self.delete_char(name)
        return self.create_char(name)

    def get_map_dict(self) -> List[Dict[str, Union[str, int]]]:
        '''
        Returns all available locations on the map for the current benchmark task.

        Returns
        -------
        list of dict
            A list of dictionaries, where each dictionary contains the name, type, x and y coordinates of a location on the map.
        '''
        url = self.url + '/maps/'
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        response = response.json()  # Fetch map data from the API
        resource_map = []
        # Process each tile in the response data
        for tile in response:
            if tile['content']:  # Check if there is any content on the tile
                x = tile['x']
                y = tile['y']
                content_type = tile['content']['code']
                object_type = tile['content']['type']
                if content_type == 'mining':
                    content_type = 'smeltery'
                resource_map.append({
                    'name': content_type.replace('_', ' '),
                    'type': object_type.replace('_', ' '),
                    'x': x,
                    'y': y,
                })

        return resource_map

    def get_item_dict(self) -> Dict:
        '''
        Fetches and returns a dictionary of all available items from the API.

        Returns
        -------
        dict
            A dictionary containing the details of the items retrieved from the server.
        '''

        items = []
        url = self.url + '/items/'
        headers = {"Accept": "application/json"}
        items = requests.get(url, headers=headers)
        items.raise_for_status()
        return items.json()

    def get_resources_dict(self) -> Dict:
        '''
        Fetches and returns a dictionary of all available resources from the API.

        Returns
        -------
        dict
            A dictionary containing the details of the resources retrieved from the server.
        '''

        url = self.url + '/resources/'
        headers = {"Accept": "application/json"}
        resources = requests.get(url, headers=headers)
        resources.raise_for_status()
        return resources.json()

    def get_monster_dict(self) -> Dict:
        '''
        Fetches and returns a dictionary of all available monsters from the API.

        Returns
        -------
        dict
            A dictionary containing the details of the monsters retrieved from the server.
        '''

        url = self.url + '/monsters/'
        headers = {"Accept": "application/json"}
        monsters = requests.get(url, headers=headers)
        monsters.raise_for_status()
        return monsters.json()

    def get_monster_by_item(self, item_name: str) -> List[str]:
        '''
        Returns a list of all monster names that drop an item with the given name.

        Parameters
        ----------
        item_name : str
            The name of the item to search for.

        Returns
        -------
        list
            A list of monster names that drop the given item.
        '''
        item_name = item_name.lower().replace(' ', '_')
        monsters_list = self.get_monster_dict()
        result = []
        for monster in monsters_list:
            if item_name in map(lambda x: x['code'], monster['drops']):
                result.append(monster['name'])
        return result

    def get_resource_by_item(self, item_name: str) -> List[str]:
        '''
        Returns a list of all resource names that drop an item with the given name.

        Parameters
        ----------
        item_name : str
            The name of the item to search for.

        Returns
        -------
        list
            A list of resource names that drop the given item.
        '''
        item_name = item_name.lower().replace(' ', '_')
        resources_list = self.get_resources_dict()
        result = []
        for resource in resources_list:
            if item_name in map(lambda x: x['code'], resource['drops']):
                result.append(resource['name'])
        return result

    def get_recipes(self) -> List:
        """Return a string with information about resource gathering of an item with name item_name.
        If item is not gatherable, returns None.
        If item is gatherable, returns a string with information about the resource gathering, including the mob name if applicable.
        """
        items_list = self.get_item_dict()
        crafting_tree = []

        for item in items_list:
            item_to_add = {'name': item['name'].lower()}
            if item['craft']:
                item_to_add["crafting components"] = item['craft']['items']
                item_to_add['required skill'] = f"{item['craft']['skill']} level {item['craft']['level']}"
            else:
                if item['subtype']:
                    if item['subtype'] == 'mob':
                        item_to_add["crafting components"] = f'Lootable from a monster {self.get_monster_by_item(item["name"])}, item level {item["level"]}'
                    elif item['subtype'] == 'grand_exchange':
                        item_to_add["crafting components"] = f'Buyable from the grand exchange, item level {item["level"]}'
                    else:
                        item_to_add["crafting components"] = f'Gatherable from the locations {self.get_resource_by_item(item["name"])}, item subtype {item["subtype"]}, item level {item["level"]}'
                else:
                    res = self.get_monster_by_item(item["name"])
                    if len(res):
                        item_to_add["crafting components"] = f'Lootable from monster {res}, item level {item["level"]}'
                    else:
                        item_to_add["crafting components"] = f'Gatherable from location {self.get_resource_by_item(item["name"])}, item level {item["level"]}'
            crafting_tree.append(item_to_add)
        return crafting_tree

    def get_items(self) -> List:
        '''
        Return all available items for the current benchmark task in a more readable format.

        Returns
        -------
        list
            A list of dictionaries containing the details of the items retrieved from the server.
        '''
        items_list = self.get_item_dict()
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

    def get_monsters(self) -> List:
        '''
        Returns a list of all available monsters for the current benchmark task in a more readable format.

        Returns
        -------
        list
            A list of dictionaries containing the details of the monsters retrieved from the server.
        '''
        
        monsters_list = self.get_monster_dict()
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
        
    def get_map_entities(self, tile_type: str) -> Dict[str, str]:
        '''
        Returns a list of all entities names according to the tile_type.
        Possible tile_types: 'monster', 'resource', 'workshop'
        '''
        map = self.get_map_dict()
        result = []
        for tile in map:
            if tile['type'] == tile_type:
                result.append((tile['name'], tile['type']))
        return result

    def get_coordinates(self, entity_list:List[str], shared_variables=None) -> Union[List[Dict], str]:
        '''
        Takes in entity_list and returns a coordinates of entities with that name.
        '''
        map = self.get_map_dict()
        result = {}
        for entity in entity_list:
            entity = entity.replace('_', ' ')
            coordinates = list(filter(lambda x: x['name'] == entity, map))
            if coordinates:
                coordinates = coordinates[0]
                if shared_variables is not None:
                    if coordinates not in shared_variables['map']:
                        shared_variables['map'].append(coordinates)
                result[coordinates['name']] = {'x': coordinates['x'], 'y': coordinates['y']}
            else:
                result[entity] = 'Entity not found on map, try another name'
        return result

    def get_recipe(self, item_name: str, shared_variables=None) -> str:
        '''
        Returns a recipe of an item_name item with all required ingredients.
        '''
        recipes = self.get_recipes()
        item_name = item_name.replace('_', ' ').lower()
        try:
            result = list(filter(lambda x: x['name'] == item_name, recipes))[0]
        except:
            result = 'Item not found, try to rewrite the item name'
            return result
        if shared_variables is not None:
            if result not in shared_variables['recipes']:
                shared_variables['recipes'].append(result)
        return result

    def get_item_by_level(self, min_level: int=1, max_level: int=40) -> List[Dict]:
        '''
        Return all available items sorted with level between min_level and max_level.
        '''
        items = self.get_items()
        result = [i for i in items if min_level <= i['level'] <= max_level]
        return result

    def get_monster_by_level(self, min_level: int=1, max_level: int=45) -> List[Dict]:
        '''
        Return all available monsters sorted with level between min_level and max_level.
        '''
        monsters = self.get_monsters()
        result = [i for i in monsters if min_level <= i['level'] <= max_level]
        return result

    def move(self, char_name: str, x: int, y: int) -> bool:
        '''
        Moves character char_name to specified coordinates x and y.
        '''
        url = f"{self.url}/my/{char_name}/action/move"
        payload = {'x': x, 'y': y}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, json=payload, headers=headers).json()
        
        if 'error' in response:
            error_codes = {
                404: f'No tile with such coodrinates {x}, {y}',
                490: f'Character {char_name} already arrived at destination {x}, {y}'
            }
            self._error_handler(error_codes, response, 'MOVE')
            return False
        else:
            time.sleep(self.delay)
            return True

    def gather(self, char_name: str, quantity: int=1) -> bool:
        '''
        Gather quantity resource on tile by char_name.
        '''
        url = f"{self.url}/my/{char_name}/action/gathering"
        payload = str(quantity)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, json=payload, headers=headers).json()

        if 'error' in response:
            error_codes = {
                598: f'Character {char_name} can\'t gather, no resource on the current tile',
                493: f'Characters {char_name} skill level is too low',
                497: f'Characters {char_name} inventory is full',
            }
            self._error_handler(error_codes, response, 'GATHER')
            return False
        else:
            time.sleep(self.delay)
            return True


    def craft(self, char_name: str, item_name: str, quantity: int=1) -> bool:
        '''
        Craft quantity item_name item by character char_name.
        '''
        if item_name.endswith('bar'):
            item_name = item_name[:-4]
        item_name = item_name.replace(" ", "_").lower()
        url = f"{self.url}/my/{char_name}/action/crafting"
        payload = {"code": f"{item_name}", "quantity": quantity}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, json=payload, headers=headers).json()

        if 'error' in response:
            error_codes = {
                404: f'Impossible to craft {item_name}, no such recipe in the database',
                478: f'Missing component or insufficient quantity to craft item {item_name}',
                486: f'Can\'t craft item {item_name}, an action is already in progress',
                493: f'Not enough skill level to craft item {item_name}',
                497: f'Can\'t craft item {item_name}, your inventory is full',
                498: f'Can\'t craft item {item_name}, character not found',
                499: f'Can\'t craft item {item_name}, character in cooldown',
                500: f'Can\'t craft item {item_name}, wrong workshop',
                598: f'Can\'t craft item {item_name}, character is not on the workshop tile'
            }
            self._error_handler(error_codes, response, 'CRAFT')
            return False
        else:
            time.sleep(self.delay)
            return True

    def unequip_item(self, char_name: str, slot_name: str) -> bool:
        '''
        Unequip item from slot slot_name of character char_name.
        Possible slots:
        weapon, shield, helmet, body_armor, leg_armor, boots, ring1, ring2, amulet, artifact1, artifact2, artifact3, consumable1, consumable2
        '''
        url = f"{self.url}/my/{char_name}/action/unequip"
        payload = {"slot": slot_name, "quantity": 1}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        response_j = response.json()

        if 'error' in response_j:
            error_codes = {
                404: 'No such item in characters equipment',
                478: f'Can\'t unequip item from slot {slot_name}, no such item',
                491: f'Slot {slot_name} is already empty',
                497: f'Can\'t unequip item from slot {slot_name}, character inventory is full',
                422: f'Can\'t unequip item from slot {slot_name}, provided input not valid for this action'
            }
            self._error_handler(error_codes, response_j, 'UNEQUIP')
            return False
        else:
            time.sleep(self.delay)
            return True
        

    def equip_item(self, char_name: str, slot_name: str, item_name: str) -> bool:
        '''
        Equip item item_name to slot slot_name of characer char_name.
        Possible slot_name value:
        weapon, shield, helmet, body_armor, leg_armor, boots, ring1, ring2, amulet, artifact1, artifact2, artifact3, consumable1, consumable2
        '''
        url = f"{self.url}/my/{char_name}/action/equip"
        slot_name = slot_name.replace(" ", "_").lower()
        item_name = item_name.replace(" ", "_").lower()
        payload = {"code": item_name, "slot": slot_name, "quantity": 1}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, json=payload, headers=headers).json()   

        if 'error' in response:
            error_codes = {
                404: f'Can\'t equip item {item_name}, no such item in your inventory',
                478: f'Can\'t equip item {item_name}, no such item',
                485: f'Can\'t equip item {item_name}, item already equiped',
                472: f'Item {item_name} is not valid for the {slot_name} slot',
                491: f'Slot {slot_name} is already full, unequip it first',
                496: f'Can\'t equip item {item_name}, characters level to low',
                422: 'Provided input not valid for this action'
            }
            self._error_handler(error_codes, response, 'EQUIP')
            return False
        else:
            time.sleep(self.delay)
            return True
        

    def slay(self, char_name: str) -> bool:
        '''
        Try to kill a monster on tile with character char_name.
        '''
        url = f"{self.url}/my/{char_name}/action/fight"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, headers=headers).json()  
    
        if 'error' in response:
            error_codes = {
                598: 'There is no monster on this tile',
            }
            self._error_handler(error_codes, response, 'SLAY')
            return False
        else:
            time.sleep(self.delay)
            fight_result = response['fight']['result']
            if fight_result == 'lose':
                return False
            else:
                return True
        
    def give(self, char_from: str, char_to: str, item_name: str, quantity: int=1) -> bool:
        '''
        Give quantity item_name item from char_from to character char_to.
        '''
        url = f"{self.url}/my/{char_from}/action/give"
        payload = {"recepient": char_to, "code": item_name.replace(" ", "_").lower(), "quantity": quantity}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        response_j = response.json()        

        if 'error' in response_j:
            error_codes = {
                478: f'Insufficient quantity of the item {item_name} in {char_from} inventory.',
                498: f'Giver character {char_from} not found',
                598: f'Recepient character {char_to} not found',
                422: 'Provided input not valid for this action'
            }
            self._error_handler(error_codes, response_j, 'GIVE')
            return False
        else:
            time.sleep(self.delay)
            return True

    def check_inventory(self, char_name: str) -> str:
        '''
        Returns information about character char_name inventory.
        '''
        url = f"{self.url}/characters/{char_name}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)
        response_j = response.json()

        if 'error' in response_j:
            error_codes = {
                404: f'Character {char_name} not found',
                422: 'Provided input not valid for this action'
            }
            self._error_handler(error_codes, response_j, 'CHECK INVENTORY')
            return False
        else:
            return response_j['inventory']

    def buy(self, char_name: str, item_name: str, quantity: int=1) -> bool:
        '''
        Buy quantity item_name from Grand Exchange with character char_name.
        '''
        url = f"{self.url}/my/{char_name}/action/buy"
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
            error_codes = {
                404: f'Item {item_name} not found in Grand Exchange',
                422: 'Provided input not valid for this action',
                478: f'Quantity must be at least 1, {quantity} provided',
                498: f'Character {char_name} not found',
                598: f'No Grand Exchange on the current position of character {char_name}'
            }
            self._error_handler(error_codes, response_j, 'BUY')
            return False
        else:
            time.sleep(self.delay)
            return True
        
    def check_item(self, char_name: str, item_name: str) -> int:
        '''
        Returns the amount of the item_name in inventory of char_name character.
        '''
        try:
            inventory = self.check_inventory(char_name)
        except Exception as e:
            raise Exception(f'[CHECK ITEM ERROR] Failed to get character {char_name} inventory with error: {e}')
        item_name = item_name.replace(' ', '_').lower()
        for item in inventory:
            if item['code'] == item_name:
                return int(item['quantity'])
        return 0

    def check_position(self, char_name: str) -> Tuple[int, int]:
        '''
        Returns tuple with the char_name current position coordinates x and y.
        '''
        url = f"{self.url}/characters/{char_name}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers).json()
        
        if 'error' in response:
            error_codes = {
                404: f'Character {char_name} not found',
                422: 'Provided input not valid for this action'
            }
            self._error_handler(error_codes, response, 'CHECK POSITION')
            return False
        else:
            return (response['x'], response['y'])

    def check_status(self, char_name: str) -> str:
        '''
        Returns all available information about character.
        '''
        url = f"{self.url}/characters/{char_name}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        response = response.json()
        
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

    def get_logs(self, char_name: str, n: int) -> list[str]:
        '''
        Returns last n logs of char_name character.
        '''
        url = f"{self.url}/logs/{n}/{char_name}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def _error_handler(error_codes: dict, response: dict, function_name: str, raise_error: bool=False) -> None:
        '''
        Handles errors for all API functions. If error code is in error_codes, it will print/raise
        appropriate error message. If error code is not in error_codes, it will print/raise
        unexpected error message.

        Args:
            error_codes (dict): Dictionary with error codes as keys and error messages as values
            response (dict): Response from API
            function_name (str): Name of the function that caused the error
            raise_error (bool): If True, function will raise Exception, otherwise it will print error message
        '''
        err_code = response['error']['code']
        err_code = int(err_code)
        if err_code in error_codes:
            if raise_error:
                raise Exception(f'[{function_name.upper()} ERROR] {error_codes[err_code]}')
            else:
                print(f'[{function_name.upper()} ERROR] {error_codes[err_code]}')
        else:
            if raise_error:
                raise Exception(f'[{function_name.upper()} ERROR] Unexpected error, try again')
            else:
                print(f'[{function_name.upper()} ERROR] Unexpected error, try again')
