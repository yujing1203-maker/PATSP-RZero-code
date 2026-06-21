import json
from collections import defaultdict

# =============================================================================
# Data Loading
# =============================================================================
with open("./utils/data/monsters.json", "r") as file:
    file_monsters = json.load(file)
with open("./utils/data/items.json", "r") as file:
    file_items = json.load(file)
with open("./utils/data/resources.json", "r") as file:
    file_resources = json.load(file)
with open("./utils/data/maps.json", "r") as file:
    file_locations = json.load(file)

# Global datasets
MONSTERS_DATA = file_monsters  
ITEMS_DATA = file_items
RESOURCES_DATA = file_resources
LOCATIONS_DATA = file_locations

# -----------------------------------------------------------------------------
# Build Lookup Dictionaries
# -----------------------------------------------------------------------------
items_by_code = {item['code']: item for item in ITEMS_DATA}
items_by_name = {item['name'].lower(): item for item in ITEMS_DATA}

# -----------------------------------------------------------------------------
# Build Monster Drop Lookup
# -----------------------------------------------------------------------------
drops_by_item = defaultdict(list)
for monster in MONSTERS_DATA:
    for drop in monster.get('drops', []):
        code = drop['code']
        drops_by_item[code].append({
            'name': monster['name'],
            'code': monster['code'],
            'rate': drop['rate'],
            'min_quantity': drop['min_quantity'],
            'max_quantity': drop['max_quantity'],
            'level': monster.get('level', 0)
        })

# -----------------------------------------------------------------------------
# Build Resource Gathering Lookup
# -----------------------------------------------------------------------------
resources_by_item = defaultdict(list)
for resource in RESOURCES_DATA:
    for drop in resource.get("drops", []):
        code = drop["code"]
        resources_by_item[code].append({
            "resource_name": resource["name"],
            "resource_code": resource["code"],
            "skill": resource["skill"],
            "resource_level": resource["level"],
            "rate": drop["rate"],
            "min_quantity": drop["min_quantity"],
            "max_quantity": drop["max_quantity"]
        })

# -----------------------------------------------------------------------------
# Build Location Lookups (full info)
# -----------------------------------------------------------------------------
locations_by_monster = defaultdict(list)
locations_by_resource = defaultdict(list)

for location in LOCATIONS_DATA:
    content = location.get("content")
    if content and isinstance(content, dict):
        if content.get("type") == "monster":
            locations_by_monster[content["code"]].append(location)
        elif content.get("type") == "resource":
            locations_by_resource[content["code"]].append(location)

# =============================================================================
# Crafting Tree Functions
# =============================================================================

def compute_difficulty(tree):
    if 'craft' in tree:
        sub_sum = 1+ sum(compute_difficulty(ing)
                      for ing in tree['craft']['ingredients'])
        return 1 + sub_sum
    else:
        return 1

def build_crafting_tree(item):
    """
    Recursively builds the crafting tree for a given item.
    If the item is craftable, its ingredients are expanded.
    Otherwise, drop/resource gathering/location information is attached.
    Also attaches a computed 'difficulty' level based on the number of distinct actions needed.
    """
    tree = {
        'name': item['name'],
        'code': item['code']
    }
    if item.get('craft'):
        craft = item['craft']
        tree['craft'] = {
            'skill': craft['skill'],
            'level': craft['level'],
            'quantity': craft['quantity'],
            'ingredients': []
        }
        for ingredient in craft['items']:
            ing_code = ingredient['code']
            ing_quantity = ingredient['quantity']
            if ing_code in items_by_code:
                sub_item = items_by_code[ing_code]
                subtree = build_crafting_tree(sub_item)
                subtree['required_quantity'] = ing_quantity
                tree['craft']['ingredients'].append(subtree)
            else:
                tree['craft']['ingredients'].append({
                    'code': ing_code,
                    'required_quantity': ing_quantity,
                    'note': 'Item not found in database.'
                })
    else:
        # Basic resource.
        tree['basic'] = True
        # Add monster drop info if available.
        if drops_by_item.get(item['code']):
            tree['monsters'] = drops_by_item[item['code']]
            for monster in tree['monsters']:
                monster_code = monster['code']
                if monster_code in locations_by_monster:
                    monster['locations'] = locations_by_monster[monster_code]
        # Add resource gathering info if available.
        if resources_by_item.get(item['code']):
            tree['resources'] = resources_by_item[item['code']]
            for resource in tree['resources']:
                resource_code = resource['resource_code']
                if resource_code in locations_by_resource:
                    resource['locations'] = locations_by_resource[resource_code]
        # If no drop or resource info exists, mark as gatherable.
        if not drops_by_item.get(item['code']) and not resources_by_item.get(item['code']):
            tree['gatherable'] = True

    # Attach the computed difficulty level.
    tree['difficulty'] = compute_difficulty(tree)
    return tree