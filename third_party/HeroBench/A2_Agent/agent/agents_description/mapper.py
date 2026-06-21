def get_global_context_mapper(agent) -> str:
    global_context = f'''Known locations: {agent.shared_variables['map']}
    You should only determine which workshop, resource or mob is needed and find the coordinates.
    Possible workshop names: ["woodcutting", "smeltery", "cooking", "weaponcrafting", "jewelrycrafting", "gearcrafting"]
    All other entities are available through the function call ```get_map_entities(entity_type)```
    Once you found all entities name, you can find coordinates via ```get_coordinates(entities_names)```
    Never use LLM to determine the coordinates, always use the function call ```get_coordinates(entities_names)```
    If there is no such entity on the map, it does not exists.
    Your output should be in JSON format and very concise, do not describe or comment anything. Just a name and coordinates.'''
    return global_context
    
mapper_desc = '''You are a game map expert. You take in a list of locations to provide the coordinates.'''