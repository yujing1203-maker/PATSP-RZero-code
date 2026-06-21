def get_global_context_fight(agent) -> str:
    global_context = f'''Character status: {agent.shared_variables['char_status']}
    Monsters: {agent.shared_variables['monsters']}
    Available equipment to craft: {agent.shared_variables['craft_equip']}
    
    Example subtask: "Check the outcome of the fight with the current equipment" or
    "Check the outcome of the fight considering a craftable equipment or
    "I have found a successful fight outcome, providing a final output"'''
    return global_context

def get_fight_analytic_desc(additional_prompt: str='') -> str:
    fight_analytic_desc = f'''You are a monster fight analytic for a game.
    You will be asked to find a way to defeat a monster in the fight between character and monster.
    Use LLM to calculate the outcome of the fight considering the statuses of the character and monster, current equipment of the character, and craftable items.

    Rules for items:
    Weapons have attack stats that will provide the damage type and damage value, for example {{'name': 'attack_air', 'value': 8}} means air damage type and 8 damage per turn. Armor and jewelry can have hp stat, that will add hp number to the players hp {{'name': 'hp', 'value': 20}}, damage amplifications stat in % {{'name': 'dmg_fire', 'value': 3}} and resistance stat in % {{'name': 'res_air', 'value': 3}}. The damage amplification will increase the attack of your current weapon by the stated value if you have matching attack type weapon.

    Rules for Fighting Monsters:
    Combat follows a turn-based system where character attack first, followed by the monster. There are different damage types: fire, earth, water, and air. If a character or monster has resistance matching the attacker's damage type, the damage is either reduced or amplified based on the resistance percentage.
    The calculations of damage follow these formulas:
    character_damage = base_attack + dmg_boost - res_reduction
    monster_damage = base_attack - res_reduction
    res_reduction = base_attack * (resist_percentage / 100)
    dmg_boost = base_attack * (dmg_percentage / 100)

    The damage is calculated from every non 0 attack type the monster or character has. After each turn, the health points of the character and monster decrease by the calculated damage. The fight is automatic and continues until the health points of one participant drops below 0.
    Characters health and monsters health is restored to full after each fight.

    Find a way to defeat a monster by calculating the outcomes. Consider a craftable equipment if fight is lost. You need to be sure that the fight will be successful.{additional_prompt}
    Be sure to calculate the remaining hp of the character and the monster at the end of the same turn.
    Your final output should be in Python dictionary format and very concise, do not describe or comment anything.Here is final output example:
    {{
        'character_win': True,
        'comment': 'Craft [```item_names```] to win the fight' # Optional
    }}
    '''

    return fight_analytic_desc