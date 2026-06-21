from memory.entity_description_generator import entity_description_generator

def character_state_parse(data: dict) -> dict:
    """
    Parse raw character state data into formatted, human-readable information.
    
    This function takes raw character data from the game API and converts it
    into a structured format with detailed descriptions of character stats,
    equipment, inventory, and skills. It also generates descriptions for
    equipped items and inventory contents.
    
    Args:
        data: Dictionary containing raw character state data with keys:
              - x, y: Character position coordinates
              - level: Character level
              - hp: Current health points
              - attack_fire, attack_earth, attack_water, attack_air: Attack values
              - res_fire, res_earth, res_water, res_air: Resistance values
              - mining_level, woodcutting_level, fishing_level: Gathering skill levels
              - weaponcrafting_level, gearcrafting_level, jewelrycrafting_level, cooking_level: Crafting skill levels
              - weapon_slot, shield_slot, helmet_slot, body_armor_slot, leg_armor_slot, boots_slot: Equipment slots
              - ring1_slot, ring2_slot, amulet_slot: Accessory slots
              - artifact1_slot, artifact2_slot, artifact3_slot: Artifact slots
              - inventory: List of inventory items with 'code' and 'quantity' keys
              
    Returns:
        Dictionary containing parsed character state with keys:
            - character_stats: Formatted character statistics string
            - character_equipment: Formatted equipment information string
            - character_inventory: Formatted inventory contents string
            - character_skills: Formatted skill levels string
            - skills_dict: Dictionary mapping skill names to levels
    """

    def code_to_name(code: str) -> str:
        """
        Convert item code to human-readable name.
        
        Args:
            code: Item code with underscores
            
        Returns:
            Formatted item name with spaces and title case
        """
        code = code.replace('_', ' ')
        code = code.title()
        return code

    def item_description(item_code: str) -> str:
        """
        Generate description for an item based on its code.
        
        Args:
            item_code: Item code to get description for
            
        Returns:
            Item description or 'Empty' if no item is equipped
        """
        if item_code == '':
            return 'Empty'
        item_name_ = code_to_name(item_code)
        search_result = entity_description_generator(item_name_)
        return search_result

    ch_info = f'''
Current position: X = {data['x']}, Y = {data['y']}
Current level: {data['level']}
Current HP: {data['hp']}  points
Fire damage: {data['attack_fire']} points
Earth damage: {data['attack_earth']} points
Water damage: {data['attack_water']} points
Air damage: {data['attack_air']} points
Fire resistance: {data['res_fire']} percent
Earth resistance: {data['res_earth']} percent
Water resistance: {data['res_water']} percent
Air resistance: {data['res_air']} percent
'''
    skills = f'''
Mining: {data['mining_level']} level
Woodcutting: {data['woodcutting_level']} level
Fishing: {data['fishing_level']} level
Weaponcrafting: {data['weaponcrafting_level']} level
Gearcrafting: {data['gearcrafting_level']} level
Jewelrycrafting: {data['jewelrycrafting_level']} level
Cooking: {data['cooking_level']} level

'''
    skills_dict = {
'mining': data['mining_level'],
'woodcutting': data['woodcutting_level'],
'fishing': data['fishing_level'],
'weaponcrafting': data['weaponcrafting_level'],
'gearcrafting': data['gearcrafting_level'],
'jewelrycrafting': data['jewelrycrafting_level'],
'cooking': data['cooking_level'],
}
    equipment = f'''
Weapon: {item_description(data['weapon_slot'])}
    
Shield: {item_description(data['shield_slot'])}
    
Helmet: {item_description(data['helmet_slot'])}
    
Body armor: {item_description(data['body_armor_slot'])}
    
Leg armor: {item_description(data['leg_armor_slot'])}
    
Boots: {item_description(data['boots_slot'])}
    
First ring: {item_description(data['ring1_slot'])}
    
Second ring: {item_description(data['ring2_slot'])}
    
Amulet: {item_description(data['amulet_slot'])}

First artifact: {item_description(data['artifact1_slot'])}

Second artifact: {item_description(data['artifact2_slot'])}

Third artifact: {item_description(data['artifact3_slot'])}
'''
    inventory = ''
    for item in data['inventory']:
        if item['code'] == "":
            continue
        item_name = code_to_name(item['code'])
        item_quantity = str(item['quantity'])
        inventory += f'Item "{item_name}" (quantity: {item_quantity}) \n'
    if inventory == '':
        inventory = 'Inventory is empty'
    parsed_state = {
        'character_stats': ch_info,
        'character_equipment': equipment,
        'character_inventory': inventory,
        'character_skills': skills,
        'skills_dict': skills_dict
        }

    return parsed_state
