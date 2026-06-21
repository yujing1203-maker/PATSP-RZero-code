import json
import unittest

from Virtual_Environment.api_calls import delete_character, create_character, gather, give_item, move, craft, equip, unequip, fight, \
    get_logs, get_character_logs, create_custom_character, delete_item


class TestGiveItem(unittest.TestCase):
    def setUp(self):
        self.name = "test_give"
        self.recipient = "recipient"

    def test_a_create_character(self):
        delete_character(self.name)
        status_code, info = create_character(name=self.name, skin="men1")
        assert status_code == 200
        assert info["name"] == self.name

    def test_b_create_recipient_character(self):
        delete_character(self.recipient)
        status_code, info = create_character(name=self.recipient, skin="men2")
        assert status_code == 200
        assert info["name"] == self.recipient

    def test_c_gather(self):
        move(self.name, 2, 0)
        status_code, info = gather(self.name) # Gather copper_ore
        print(info)
        assert status_code == 200


    def test_d_give_item(self):
        status_code, info = give_item(name=self.name, recipient=self.recipient, code="copper_ore", quantity=1)
        assert status_code == 200


class TestCopperDagger(unittest.TestCase):

    def setUp(self):
        self.name = "copper"

    def test_a_create_character(self):
        delete_character(self.name)
        status_code, info = create_character(name=self.name, skin="men1")
        assert status_code == 200
        assert info["name"] == self.name

    def test_b_mine_copper_ore(self):
        move(self.name, 2, 0)  # Move to copper rocks
        gather(self.name, 48)

    def test_c_craft_copper(self):
        move(self.name, 1, 5)  # Move to mining workshop
        status_code, info = craft(self.name, "copper", 6)
        assert status_code == 200

    def test_d_copper_dagger_craft(self):
        move(self.name, 2, 1)  # Move to weaponcrafting workshop
        status_code, info = craft(self.name, "copper_dagger", 1)  # Craft copper dagger
        assert status_code == 200
        assert info["details"]["items"][0]["code"] == "copper_dagger"
        assert info["details"]["items"][0]["quantity"] == 1

    def test_e_unequip(self):
        status_code, info = unequip(self.name, "weapon")
        print(info)
        assert status_code == 200

    def test_f_equip(self):
        status_code, info = equip(name=self.name, slot="weapon", code="copper_dagger")
        assert status_code == 200
        assert info["item"]["code"] == "copper_dagger"
        assert info["character"]["attack_air"] == 8

    def test_h_get_logs(self):
        status_code, info = get_logs(100)
        assert status_code == 200
        print(info)

    def test_i_get_character_logs(self):
        status_code, info = get_character_logs(self.name, 10)
        assert status_code == 200
        print(info)


class TestConsumable(unittest.TestCase):
    def setUp(self):
        self.name = "consume"

    def test_a_create_character(self):
        delete_character(self.name)
        status_code, info = create_character(name=self.name, skin="men1")
        assert status_code == 200
        assert info["name"] == self.name

    def test_b_fish_gudgeon(self):
        move(self.name, 4, 2)  # Move to gudgeon_fishing_spot
        status_code, info = gather(self.name, 44)
        assert status_code == 200

    def test_c_cook_gudgeon(self):
        move(self.name, 1, 1)  # Move to cooking workshop
        status_code, info = craft(self.name, "cooked_gudgeon", 44)
        assert status_code == 200

    def test_d_equip_gudgeon(self):
        status_code, info = equip(name=self.name, slot="consumable1", code="cooked_gudgeon", quantity=22)
        assert status_code == 200
        status_code, info = equip(name=self.name, slot="consumable2", code="cooked_gudgeon", quantity=22)
        assert status_code == 200

    def test_e_unequip_gudgeon(self):
        status_code, info = unequip(name=self.name, slot="consumable1", quantity=1)
        assert status_code == 200

    def test_f_fight(self):
        move(name=self.name, x=1, y=-2) # Move to yellow slime
        status_code, info = fight(self.name)
        assert status_code == 200
        assert info["fight"]["result"] == "win"

    def test_g_delete_character(self):
        status_code, info = delete_character(self.name)
        assert status_code == 200
        print(info)

class TestCustomCharacter(unittest.TestCase):
    def setUp(self):
        self.name = "custom"

    def test_a_create_character(self):
        delete_character(self.name)
        with open("../Data/example_custom_character.json") as json_file:
            char_data = json.load(json_file)
        status_code, info = create_custom_character(name=self.name, skin="men1", char_data=char_data)
        assert status_code == 200
        assert info["name"] == self.name


class TestFight(unittest.TestCase):
    def setUp(self):
        self.name = "fight"

    def test_a_create_character(self):
        delete_character(self.name)
        status_code, info = create_character(name=self.name, skin="men1")
        assert status_code == 200
        assert info["name"] == self.name

    def test_b_fight(self):
        move(name=self.name, x=0, y=1) # Move to Chicken
        status_code, info = fight(self.name)
        assert status_code == 200
        assert info["fight"]["result"] == "win"
        print(info)

class TestHardFight(unittest.TestCase):
    def setUp(self):
        self.name = "fight"

    def test_a_create_character(self):
        delete_character(self.name)
        status_code, info = create_character(name=self.name, skin="men1")
        assert status_code == 200
        assert info["name"] == self.name

    def test_b_mine_copper_ore(self):
        move(self.name, 2, 0)  # Move to copper rocks
        gather(self.name, 1600)

    def test_c_craft_copper(self):
        move(self.name, 1, 5)  # Move to workshop mining
        status_code, info = craft(self.name, "copper", 200)
        assert status_code == 200

    def test_d_craft_copper_boots(self):
        move(self.name, 3, 1)  # Move to workshop gear
        status_code, info = craft(self.name, "copper_boots", 25)
        print(info)
        assert status_code == 200

    def test_e_fight_chicken(self):
        move(name=self.name, x=0, y=1)  # Move to chicken
        for i in range(6):
            fight(self.name)

    def test_f_craft_feather_coat(self):
        move(name=self.name, x=3, y=1)  # Move to workshop gear
        status_code, info = craft(self.name, "feather_coat", 1)
        print(info)
        assert status_code == 200

    def test_g_equip_feather_coat(self):
        status_code, info = equip(self.name, "body_armor", "feather_coat")
        assert status_code == 200

    def test_h_equip_copper_boots(self):
        status_code, info = equip(self.name, "boots", "copper_boots")
        assert status_code == 200

class TestWrongCraft(unittest.TestCase):
    def setUp(self):
        self.name = "test"

    def test_a_create_character(self):
        delete_character(self.name)
        status_code, info = create_character(name=self.name, skin="men1")
        assert status_code == 200
        assert info["name"] == self.name

    def test_b_mine_copper_ore(self):
        move(self.name, 2, 0)  # Move to copper rocks
        gather(self.name, 80)

    def test_c_craft_copper(self):
        move(self.name, 1, 5)  # Move to workshop mining
        status_code, info = craft(self.name, "copper", 8)
        assert status_code == 200

    def test_d_craft_wrong_weapon(self):
        move(self.name, 3, 1)  # Move to workshop gear
        status_code, info = craft(self.name, "copper_weapon", 1)
        assert status_code == 404

    def test_e_delete_items(self):
        status_code, info = delete_item(self.name, code="copper", quantity=8)
        assert status_code == 200