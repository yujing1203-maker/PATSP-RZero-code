def name_formater(name, value):
    """
    Format effect names and values into human-readable descriptions.
    
    This function takes effect names and their values and converts them into
    natural language descriptions. It handles various types of effects including
    attack bonuses, resistances, skill boosts, and utility effects.
    
    Args:
        name: The effect name/code (e.g., 'attack_fire', 'res_water', 'hp', etc.)
        value: The numerical value of the effect (can be positive or negative)
        
    Returns:
        Formatted string describing the effect in human-readable language

    """

    def val_text(val):
        """
        Format a value as text indicating increase or decrease.
        
        Args:
            val: The numerical value to format
            
        Returns:
            String describing the value change (increase or decrease)
        """
        if val < 0:
            val = -val
            return f' decreased by {val} '
        else:
            return f' increased by {val} '

    if 'attack' in name:
        name_splited = name.split('_')
        new_name = f'{name_splited[1]} element damage'
        new_name += val_text(value)
        new_name += 'points'
        return new_name
    if 'res' in name:
        name_splited = name.split('_')
        new_name = f'{name_splited[1]} element resistance'
        new_name += val_text(value)
        new_name += 'percent'
        return new_name

    if 'dmg' in name:
        name_splited = name.split('_')
        new_name = f'{name_splited[1]} element damage'
        new_name += val_text(value)
        new_name += 'percent'
        return new_name

    if 'boost' in name != 'boost_hp':
        name_splited = name.split('_')
        if name_splited[1] == 'res':
            new_name = f'{name_splited[2]} element resistance'
        else:
            new_name = f'{name_splited[2]} element damage'

        new_name += val_text(value)
        new_name += 'percent'
        return new_name

    if name == 'hp':
        new_name = f'Increase HP by {value} points'
        return new_name

    if name == 'heal':
        new_name = f'Heal by {value} points'
        return new_name

    if name == 'gold':
        new_name = f'Get {value} gold'
        return new_name

    if name == 'teleport_x':
        new_name = f'Teleport to position X = {value}'
        return new_name

    if name == 'teleport_y':
        new_name = f'Teleport to position Y = {value}'
        return new_name

    if name == 'restore':
        new_name = f'Restore {value} HP if the player has lost more than 50 percent of his health points'
        return new_name

    if name == 'mining':
        new_name = f'Reduced cooldown by {value} percent when a character mines a resource'
        return new_name

    if name == 'woodcutting':
        new_name = f'Reduced cooldown by {value} percent when a character logs a tree'
        return new_name

    if name == 'fishing':
        new_name = f"Reduced cooldown by {value} percent when a character is fishing."
        return new_name

    if name == 'alchemy':
        new_name = f"Reduced cooldown by {value} percent when a character harvest a plant"
        return new_name

    if name == 'inventory_space':
        new_name = f'Increased inventory space by {value}'
        return new_name

    if name == 'haste':
        new_name = f'Decreased action cooldown by {value} percent'
        return new_name

    if name == 'boost_hp':
        new_name = f'Temporary increased max HP by {value}'
        return new_name

