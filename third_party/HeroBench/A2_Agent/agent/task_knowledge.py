from typing import List, Dict, Optional, Tuple, Union
from utils.api import ArtifactsApi

class TaskKnowledge:
    def __init__(self, knowledge: Dict, api: ArtifactsApi):
        """
        Initialize the TaskKnowledge class with knowledge and api.

        The TaskKnowledge class provides methods to get information about the
        current benchmark task, such as the available items, monsters, resources,
        and map locations.

        Args:
            knowledge (Dict): The knowledge dictionary containing the task data.
            api (ArtifactsApi): The ArtifactsApi object used to interact with the game.
        """
        self.knowledge = knowledge
        self.api = api
    def get_map_dict(self) -> List[Dict]:
        ''' 
        Return all available locations on the map for the current benchmark task.
        '''
        resource_map = []
        # Process each tile in the response data
        for tile in self.knowledge['Locations']:
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

    def get_item_dict(self) -> List[Dict]:
        '''
        Return all available items for the current benchmark task.
        '''
        result = []
        keys_list = ['Craftable items', 'Resources']
        for key in keys_list:
            result.extend(self.knowledge[key])
        return result

    def get_monster_dict(self) -> List[Dict]:
        '''
        Return all available monsters for the current benchmark task.
        '''
        return self.knowledge.get('Monsters', [])

    def get_items(self) -> List[Dict]:
        '''
        Return all available items for the current benchmark task in a more readable format.
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

    def get_equipment(self) -> List[Dict]:
        '''
        Return all available equipment for the current benchmark task.
        '''
        items_list = self.get_items()
        result = []
        
        for item in items_list:
            if item['type'] != 'resource':
                result.append(item)
        return result

    def get_resource_by_item(self, item_name: str) -> List[str]:
        '''
        Returns a list of all gatherable resources by item item_name.
        '''
        item_name = item_name.lower().replace(' ', '_')
        resources_list = self.api.get_resources_dict()
        available_sites = [x[0] for x in self.get_map_entities('resource')]
        result = []
        for resource in resources_list:
            if item_name in map(lambda x: x['code'], resource['drops']) and resource['name'].lower() in available_sites:
                result.append(resource['name'].lower())
        return result

    def get_monster_by_item(self, item_name: str) -> List[str]:
        '''
        Returns a list of all monsters that drop item item_name.
        '''
        item_name = item_name.lower().replace(' ', '_')
        monsters_list = self.get_monster_dict()
        result = []
        for monster in monsters_list:
            if item_name in map(lambda x: x['code'], monster['drops']):
                result.append(monster['name'].lower())
        return result

    def get_recipes(self) -> List[Dict]:
        '''
        Returns a list of all available recipes for the current benchmark task.
        '''
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
                        item_to_add["crafting components"] = f'Gatherable from the location {self.get_resource_by_item(item["name"])}, item subtype {item["subtype"]}, item level {item["level"]}'
                else:
                    res = self.get_monster_by_item(item["name"])
                    if len(res):
                        item_to_add["crafting components"] = f'Lootable from monster {res}, item level {item["level"]}'
                    else:
                        item_to_add["crafting components"] = f'Gatherable from location {self.get_resource_by_item(item["name"])}, item level {item["level"]}'
            crafting_tree.append(item_to_add)
        return crafting_tree

    def get_monsters(self) -> List[Dict]:
        '''
        Returns a list of all available monsters for the current benchmark task in a readable format.
        '''
        monsters_list = self.get_monster_dict()
        result = []

        for monster in monsters_list:
            monster_to_add = dict(
                name = monster['name'].lower(),
                level = monster['level'],
                hp = monster['hp'],
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
                drops = monster['drops']
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

    def get_coordinates(self, shared_variables: Dict, entity_list:List[str]) -> Union[List[Dict], str]:
        '''
        Takes in entity_list and returns a coordinates of entities with that name.
        Saves the coordinates in shared_variables['map'].
        '''
        map = self.get_map_dict()
        result = {}
        for entity in entity_list:
            entity = entity.replace('_', ' ').lower()
            coordinates = list(filter(lambda x: x['name'] == entity, map))
            if coordinates:
                coordinates = coordinates[0]
                if coordinates not in shared_variables['map']:
                    shared_variables['map'].append(coordinates)
                result[coordinates['name']] = {'x': coordinates['x'], 'y': coordinates['y']}
            else:
                result[entity] = 'Entity not found on map, try another name'
        return result

    def get_recipe(self, shared_variables: Dict, item_name: str) -> str:
        '''
        Returns a recipe of an item_name item with all required ingredients.
        Saves the recipe in shared_variables['recipes'].
        '''
        recipes = self.get_recipes()
        item_name = item_name.replace('_', ' ').lower()
        try:
            result = list(filter(lambda x: x['name'] == item_name, recipes))[0]
        except:
            result = 'Item not found, try to rewrite the item name'
            return result
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
    
    def get_starting_equipment(self):
        '''
        Return all characters' starting items for the current benchmark task.
        '''
        
        return self.knowledge.get('Items stats', [])