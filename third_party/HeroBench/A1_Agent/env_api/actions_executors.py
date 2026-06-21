import requests

class APIBasicAction:
    """
    Base class for API-based actions.
    
    This class provides common functionality for making HTTP requests
    to the environment API, including request formatting, response parsing,
    and error handling.
    """

    def __init__(self) -> None:
        """
        Initialize the API action with standard headers.
        """
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
                        }

    def send_request(self, url: str, payload: dict, parse_response:bool=True) -> dict:
        """
        Send a POST request to the game API with error handling.
        
        This method handles the HTTP communication with the game API,
        including request formatting, response parsing, and error handling.
        
        Args:
            url: API endpoint URL to send the request to
            payload: Data to send in the request body
            parse_response: Whether to parse the response into standard format
            
        Returns:
            Dictionary containing the API response or error information
        """
        response = requests.post(url, json=payload, headers=self.headers)
        try:
            response_json = response.json()
        except:
            response_json = {'status' : 'Fail', 'error': {'message': 'Game internal error'}}
            print(response)
        if response.status_code != 200:
            try:
                error_message = response_json['error']['message']
            except:
                error_message = 'Game internal error'
            if parse_response:
                try:
                    parsed_response = self.parse_response(response_json)
                    return parsed_response
                except:
                    pass
            else:
                return {'status': 'Fail', 'error': error_message}
        if parse_response:
            return self.parse_response(response_json)
        else:
            return response_json

    @staticmethod
    def parse_response(response: dict) -> dict:
        """
        Parse API response into standard format.
        
        Args:
            response: Raw API response dictionary
            
        Returns:
            Dictionary with standardized 'status' and 'message' keys
        """
        status = {'status': None, 'message': None}
        if 'error' in response:
            status['status'] = 'Fail'
            status['message'] = response['error']['message']
            return status
        else:
            status['status'] = 'Success'
            status['message'] = ''
            return status

class CharacterInfo:
    """
    Class for retrieving character information from the environment API.
    
    This class provides methods to get current character state including
    stats, equipment, inventory, and position.
    """

    def __init__(self, character_name: str) -> None:
        """
        Initialize character info with character name.
        
        Args:
            character_name: Name of the character to get info for
        """
        super().__init__()
        self.character_name = character_name
        self.url = f"http://127.0.0.1:8000/characters/{character_name}"

    def action(self) -> dict:
        """
        Get current character state from the environment API.
        
        Returns:
            Dictionary containing character state information
        """
        headers = {"Accept": "application/json"}
        response = requests.get(self.url, headers=headers)
        return response.json()

class Move(APIBasicAction):
    """
    Class for executing movement actions in the environment.
    
    This class handles character movement to specific coordinates
    in the environment.
    """

    def __init__(self, character_name: str) -> None:
        """
        Initialize movement action with character name.
        
        Args:
            character_name: Name of the character to move
        """
        super().__init__()
        self.character_name = character_name
        self.url = f"http://127.0.0.1:8000/my/{self.character_name}/action/move"
        
    def action(self, x: int,y: int) -> dict:
        """
        Move character to specified coordinates.
        
        Args:
            x: X coordinate to move to
            y: Y coordinate to move to
            
        Returns:
            Dictionary containing movement result with status and message
        """
        payload = {
            "x": x,
            "y": y,
        }
        action_result = self.send_request(url=self.url, payload=payload)
        if action_result['status'] == 'Fail' and action_result['message'] == 'Character already at destination.':
            action_result['status'] = 'Success'
            action_result['message'] = ''
        return action_result

class Gather(APIBasicAction):
    """
    Class for executing resource gathering actions.
    
    This class handles gathering resources from the current location.
    """

    def __init__(self, character_name: str) -> None:
        """
        Initialize gathering action with character name.
        
        Args:
            character_name: Name of the character performing the gathering
        """
        super().__init__()
        self.character_name = character_name
        self.url = f"http://127.0.0.1:8000/my/{self.character_name}/action/gathering"

    def action(self, resource_count: int = 1) -> dict:
        """
        Gather resources from the current location.
        
        Args:
            resource_count: Number of resources to gather (default: 1)
            
        Returns:
            Dictionary containing gathering result with status and message
        """
        action_result = self.send_request(url=self.url, payload=str(resource_count))
        return action_result

class Fight(APIBasicAction):
    """
    Class for executing combat actions.
    
    This class handles combat encounters with monsters at the current location.
    """

    def __init__(self, character_name: str) -> None:
        """
        Initialize fight action with character name.
        
        Args:
            character_name: Name of the character engaging in combat
        """
        super().__init__()
        self.character_name = character_name
        self.url = f"http://127.0.0.1:8000/my/{self.character_name}/action/fight"

    @staticmethod
    def parse_fight(fight_data: dict) -> dict:
        """
        Parse fight result data into standard format.
        
        This method handles various fight outcomes including wins, losses,
        and error conditions.
        
        Args:
            fight_data: Raw fight result data from the API
            
        Returns:
            Dictionary containing standardized fight result with status and message
        """
        status = {'status' : None, 'message' : None}

        if 'detail' in fight_data:
            status['status'] = 'Fail'
            status['message'] = 'Character too weak'
            return status

        if 'error' in fight_data:
            status['status'] = 'Fail'
            try:
                status['message'] = fight_data['error']['message']
            except:
                status['message'] = fight_data['error']
            status['message'] = status['message'].replace('map.','tile.')
            return status

        fight_result = fight_data['fight']['result']
        if fight_result == 'lose':
            status['status'] = 'Fail'
            status['message'] = 'Character too weak'
            return status

        if fight_result == 'win':
            status['status'] = 'Success'
            status['message'] = 'Character wins'
            return status

    def action(self) -> dict:
        """
        Engage in combat with monster at current location.
        
        Returns:
            Dictionary containing combat result with status and message
        """
        action_result = self.send_request(url=self.url, payload={},parse_response=False)
        action_result = self.parse_fight(action_result)
        return action_result

class Equip(APIBasicAction):
    """
    Class for executing equipment actions.
    
    This class handles equipping and unequipping items in various equipment slots.
    """

    def __init__(self, character_name: str) -> None:
        """
        Initialize equip action with character name.
        
        Args:
            character_name: Name of the character to equip items for
        """
        super().__init__()
        self.character_name = character_name
        self.url = f"http://127.0.0.1:8000/my/{self.character_name}/action/equip"
        self.unequip_url = f"http://127.0.0.1:8000/my/{self.character_name}/action/unequip"
        self.slots = ['weapon','shield','helmet','body_armor','leg_armor','boots','ring1','ring2','amulet','artifact1','artifact2','artifact3']
        self.character_info = CharacterInfo(self.character_name)

    def _try_equip(self, item_code: str, slot: str) -> dict:
        """
        Attempt to equip an item in a specific slot.
        
        This method first tries to unequip any existing item in the slot,
        then attempts to equip the new item.
        
        Args:
            item_code: Code of the item to equip
            slot: Equipment slot to equip the item in
            
        Returns:
            Dictionary containing equip result with status and message
        """
        unequip_res = self._equip_action(item_code, slot, unequip=True)
        if not 'detail' in unequip_res and not 'error' in unequip_res:
            unequiped_item_code = unequip_res['item']['code']
            equip_res = self._equip_action(item_code, slot)
            if equip_res['status'] == 'Success':
                return {'status': 'Success', 'message': ''}
            else:
                _ = self._equip_action(unequiped_item_code, slot)

        else:
            equip_res = self._equip_action(item_code, slot)
            if equip_res['status'] == 'Success':
                return {'status': 'Success', 'message': ''}

        return {'status': 'Fail', 'message': 'Cant equip this item'}

    def action(self,item_name: str) -> dict:
        """
        Equip an item by name, automatically finding an appropriate slot.
        
        This method attempts to equip an item by trying available slots
        in order of preference.
        
        Args:
            item_name: Name of the item to equip
            
        Returns:
            Dictionary containing equip result with status and message
        """
        item_code = item_name.lower().replace(" ", "_")
        character_info = self.character_info.action()
        available_slots = {'weapon': True,'shield': True,'helmet': True,'body_armor': True,'leg_armor': True,'boots': True,'ring1': True,'ring2': True,'amulet': True,'artifact1': True,'artifact2': True,'artifact3': True}
        item_equiped = False
        for slot in available_slots:
            slot_name_inventory = slot+'_slot'
            if character_info[slot_name_inventory] != '':
                available_slots[slot] = False

        for slot in available_slots:
            if available_slots[slot] is False: continue
            equip_res = self._try_equip(item_code, slot)
            if equip_res['status'] == 'Success':
                return equip_res

        if not item_equiped:
            for slot in self.slots:
                equip_res = self._try_equip(item_code, slot)
                if equip_res['status'] == 'Success':
                    return equip_res

        return {'status': 'Fail', 'message': 'Cant equip this item'}

    def _equip_action(self, item_code: str, slot: str, unequip=False) -> dict:
        """
        Perform equip or unequip action for a specific item and slot.
        
        Args:
            item_code: Code of the item to equip/unequip
            slot: Equipment slot to use
            unequip: Whether to unequip (True) or equip (False) the item
            
        Returns:
            Dictionary containing the equip/unequip result
        """
        payload = {
            'code': item_code,
            'slot': slot,
            'quantity': 1
        }
        if unequip:
            action_result = self.send_request(url=self.unequip_url, payload=payload, parse_response=False)
        else:
            action_result = self.send_request(url = self.url,payload=payload)

        return action_result

class Craft(APIBasicAction):
    """
    Class for executing crafting actions.
    
    This class handles crafting items at workshops, with
    automatic equipping of crafted items.
    """

    def __init__(self, character_name: str, auto_equip: bool = True) -> None:
        """
        Initialize craft action with character name and auto-equip setting.
        
        Args:
            character_name: Name of the character performing the crafting
            auto_equip: Whether to automatically equip crafted items (default: True)
        """
        super().__init__()
        self.character_name = character_name
        self.url = f"http://127.0.0.1:8000/my/{self.character_name}/action/crafting"
        self.equip_action = Equip(character_name)
        self.auto_equip = auto_equip

    def action(self, item_name: str,quantity: int) -> dict:
        """
        Craft an item with specified quantity.
        
        Args:
            item_name: Name of the item to craft
            quantity: Number of items to craft
            
        Returns:
            Dictionary containing crafting result with status and message
        """
        item_code = item_name.lower().replace(" ", "_")
        payload = {
            "code": item_code,
            "quantity": quantity,
        }
        action_result = self.send_request(url=self.url, payload=payload)
        if action_result['status'] == 'Success' and self.auto_equip:
            equip_action_result = self.equip_action.action(item_code)
            if equip_action_result['status'] == 'Success':
                return {'status' : 'Success', 'message': 'Item crafted and equipped'}
            else:
                return {'status' : 'Success', 'message': 'Item crafted and not equipped'}
        else:
            return action_result




