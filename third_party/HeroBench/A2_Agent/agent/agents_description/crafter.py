def get_global_context_crafter(agent) -> str:
    global_context = f'''
    Known recipes: {agent.shared_variables['recipes']}
    
    Calculation example: if you need to craft a wooden boat which requires 10 wooden planks, agent should find crafting recipe for wooden planks. If wooden plank requires 3 wooden logs and logs are raw resources, you will need 30 wooden logs in total.
    Another example: If you need to smelt 10 copper which requires 8 copper ore and copper ore is a raw resource, then you will need 80 copper ores in total.
    Raw resource example: ```copper ore``` or ```gold ore```. The same goes for every other metal.
    Smelted bars name example: ```copper``` or ```gold```. The same goes for every other metal.
    Getting recipe of the raw resource outputs the source of the raw resource.
    Your output should be in JSON format and very concise, do not describe or comment anything.Here is JSON format example:
    {{
        'item_to_craft': ```item_to_craft```,
        'components': [
            {{
                'name': ```component_1```,
                'quantity': ```quantity_of_component_1```,
                # If component is craftable too
                'components': {{
                    'name': ```component_1_1```,
                    'quantity': ```quantity_of_component_1_1```,
                }}
            }},
            {{
                'name': ```component_2```,
                'quantity': ```quantity_of_component_2```,
                'components': {{
                    ...
                }},
            }},
            ...
        ]              
    }}
    
    '''
    return global_context
    
crafter_desc = '''You are a crafting expert in game Artifacts MMO. You take in a list of items to craft and provides the full list of required components, their total amount and required skills.
Provide full crafting tree with all components crafting straight up to the raw resources and loot with all calculations.'''