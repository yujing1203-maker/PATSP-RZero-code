import json
from collections import defaultdict

# =============================================================================
# Data Loading
# =============================================================================
with open("Virtual_Environment/Data/monsters.json", "r") as file:
    file_monsters = json.load(file)
with open("Virtual_Environment/Data/items.json", "r") as file:
    file_items = json.load(file)
with open("Virtual_Environment/Data/resources.json", "r") as file:
    file_resources = json.load(file)
with open("Virtual_Environment/Data/maps.json", "r") as file:
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
# def compute_difficulty(tree):
#     """
#     Recursively compute the difficulty of obtaining/crafting an item based on its tree.

#     For a craftable item:
#       difficulty = 1 (for the current craft action) 
#                    + maximum difficulty among its ingredient branches 
#                    + (number of ingredients - 1) to account for multiple distinct actions.

#     For a basic item (non-craftable):
#       difficulty = 1, regardless of whether it comes from a monster drop or resource gathering.
#     """
#     if 'craft' in tree:
#         ingredients = tree['craft'].get('ingredients', [])
#         if ingredients:
#             # Compute difficulty for each ingredient branch.
#             sub_difficulties = [compute_difficulty(ing) for ing in ingredients]
#             # The current craft action adds 1, plus the longest branch and a penalty for additional branches.
#             return 1 + max(sub_difficulties) + (len(ingredients) - 1)
#         else:
#             return 1  # Only the craft action itself.
#     else:
#         # Basic items (whether drop or gatherable) require one action.
#         return 1

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


def search_tree_for_unwanted_drop(tree, target_monster):
    """
    Recursively searches a crafting tree.
    Returns True if any drop in the tree is from:
      - the target monster, OR
      - a monster whose level is higher than the target monster's level.
    """
    if 'monsters' in tree:
        for drop in tree['monsters']:
            if drop['code'] == target_monster['code'] or drop.get('level', 0) > target_monster.get('level', 0):
                return True
    if 'craft' in tree:
        for ingredient in tree['craft']['ingredients']:
            if search_tree_for_unwanted_drop(ingredient, target_monster):
                return True
    return False

def item_uses_unwanted_drop(item_code, target_monster):
    """
    Returns True if the given item (by its code) or any of its crafting ingredients
    uses a drop from the target monster OR from a monster whose level is higher than the target.
    """
    if item_code not in items_by_code:
        return False
    item = items_by_code[item_code]
    tree = build_crafting_tree(item)
    return search_tree_for_unwanted_drop(tree, target_monster)

def filter_items_for_monster(items, target_monster):
    """
    Returns a list of items that do NOT have a crafting tree (or basic drop)
    which involves:
      - the target monster, OR
      - any monster with a level higher than the target monster's level.
    """
    filtered = []
    for item in items:
        if not item_uses_unwanted_drop(item['code'], target_monster):
            filtered.append(item)
    return filtered

def get_monster_by_name(monster_name, monsters_list):
    for m in monsters_list:
        if m["name"].lower() == monster_name.lower():
            return m
    return None

# -----------------------------------------------------------------------------
# (Optional) Allow running this module standalone for testing tree generation.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    search_name = "gold sword"  # Replace with the desired item name
    test_item = items_by_name.get(search_name.lower())

    if not test_item:
        print(f"Item '{search_name}' not found.")
    else:
        tree = build_crafting_tree(test_item)
        print("Crafting tree for", test_item['name'])
        import pprint
        pprint.pprint(tree)