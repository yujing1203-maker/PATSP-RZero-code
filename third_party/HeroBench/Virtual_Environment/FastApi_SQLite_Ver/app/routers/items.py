from app.routers import error_response
import json
from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Path
from sqlmodel import Field, Relationship, Session, SQLModel, select, or_

from app.db import SessionDep
from app.routers import error_response


class CraftSkill(str, Enum):
    weaponcrafting = "weaponcrafting"
    gearcrafting = "gearcrafting"
    jewelrycrafting = "jewelrycrafting"
    cooking = "cooking"
    woodcutting = "woodcutting"
    mining = "mining"


class ItemType(str, Enum):
    weapon = "weapon"
    body_armor = "body_armor"
    resource = "resource"
    leg_armor = "leg_armor"
    helmet = "helmet"
    boots = "boots"
    shield = "shield"
    amulet = "amulet"
    ring = "ring"
    artifact = "artifact"
    consumable = "consumable"
    currency = "currency"


router = APIRouter()


class CraftItemLink(SQLModel, table=True):
    craft_id: int | None = Field(default=None, foreign_key="craft.id", primary_key=True)
    craft_item_id: int | None = Field(default=None, foreign_key="craftitem.id", primary_key=True)


class ItemCraftLink(SQLModel, table=True):
    item_id: int | None = Field(default=None, foreign_key="item.id", primary_key=True)
    craft_id: int | None = Field(default=None, foreign_key="craft.id", primary_key=True)


class CraftItem(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    code: str = Field(description="Item code.", index=True, regex=r'^[a-zA-Z0-9_-]+$')
    quantity: int = Field(description="Item quantity.", default=1, ge=1)

    crafts: list["Craft"] = Relationship(back_populates="items", link_model=CraftItemLink)


class Craft(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    skill: CraftSkill = Field(description="Skill required to craft the item.", index=True)
    level: int = Field(description="The skill level required to craft the item.", index=True, ge=1, default=1)
    items: list[CraftItem] = Relationship(back_populates="crafts", link_model=CraftItemLink)
    quantity: int = Field(description="Quantity of items crafted.", default=1)

    items_craft: list["Item"] = Relationship(back_populates="craft", link_model=ItemCraftLink)


class ItemEffectLink(SQLModel, table=True):
    item_id: int | None = Field(default=None, foreign_key="item.id", primary_key=True)
    effect_id: int | None = Field(default=None, foreign_key="effect.id", primary_key=True)


class Effect(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(description="Effect name.", index=True)
    value: int = Field(description="Effect value.")

    items_effect: list["Item"] = Relationship(back_populates="effects", link_model=ItemEffectLink)


class Item(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(description="Item name.", index=True)
    code: str = Field(description="Item code. This is the item's unique identifier (ID).", index=True, unique=True, regex=r'^[a-zA-Z0-9_-]+$')
    level: int = Field(description="Item level.", default=1, index=True)
    type: ItemType = Field(description="Item type.", index=True)
    subtype: str = Field(description="Item subtype.", default="", index=True)
    description: str = Field(description="Item description.", default="")
    effects: list[Effect] | None = Relationship(back_populates="items_effect", link_model=ItemEffectLink)
    craft: Craft | None = Relationship(back_populates="items_craft", link_model=ItemCraftLink)


class EffectResponse(SQLModel, ordered=True):
    name: Annotated[str, Field(description="Effect name.")]
    value: Annotated[int, Field(description="Effect value.")]


class CraftItemResponse(SQLModel, ordered=True):
    code: Annotated[str, Field(description="Item code.", regex=r'^[a-zA-Z0-9_-]+$')]
    quantity: Annotated[int, Field(description="Item quantity.", ge=1)]


class CraftResponse(SQLModel, ordered=True):
    skill: Annotated[CraftSkill, Field(description="Skill required to craft the item.")]
    level: Annotated[int, Field(description="The skill level required to craft the item.", ge=1)]
    items: Annotated[list[CraftItemResponse], Field(description="List of items required to craft the item.")]
    quantity: Annotated[int, Field(description="Quantity of items crafted.", ge=1)]


class ItemResponse(SQLModel, ordered=True):
    name: Annotated[str, Field(description="Item name.")]
    code: Annotated[str, Field(description="Item code. This is the item's unique identifier (ID).", regex=r'^[a-zA-Z0-9_-]+$')]
    level: Annotated[int, Field(description="Item level.", ge=1)]
    type: Annotated[ItemType, Field(description="Item type.")]
    subtype: Annotated[str, Field(description="Item subtype.")]
    description: Annotated[str, Field(description="Item description.")]
    effects: Annotated[list[EffectResponse], Field(description="List of object effects. For equipment, it will include item stats.", default=[])] = []
    craft: Annotated[CraftResponse | None, Field(description="Craft information. If applicable.", default=None)] = None


def load_items_data(session: Session):
    with open('app/Data/items.json') as f:
        items_data = json.load(f)
        for item_data in items_data:
            effects_data = item_data.pop('effects', [])
            craft_data = item_data.pop('craft', None)

            # Create Item
            item = Item(**item_data)
            session.add(item)
            session.commit()
            session.refresh(item)

            # Add Effects
            for effect_data in effects_data:
                effect = Effect(name=effect_data['name'], value=effect_data['value'])
                session.add(effect)
                session.commit()
                session.refresh(effect)

                # Link Item and Effect
                item_effect_link = ItemEffectLink(item_id=item.id, effect_id=effect.id)
                session.add(item_effect_link)

            # Add Craft
            if craft_data:
                craft = Craft(skill=craft_data['skill'], level=craft_data['level'], quantity=craft_data['quantity'])
                session.add(craft)
                session.commit()
                session.refresh(craft)

                # Add Craft Items
                for craft_item_data in craft_data['items']:
                    craft_item = CraftItem(code=craft_item_data['code'], quantity=craft_item_data['quantity'])
                    session.add(craft_item)
                    session.commit()
                    session.refresh(craft_item)

                    # Link Craft and CraftItem
                    craft_item_link = CraftItemLink(craft_id=craft.id, craft_item_id=craft_item.id)
                    session.add(craft_item_link)

                # Link Item and Craft
                item_craft_link = ItemCraftLink(item_id=item.id, craft_id=craft.id)
                session.add(item_craft_link)

            session.commit()
    return True


@router.get(
    name="Get All Items",
    path="/items",
    tags=["Items"],
    response_model=list[ItemResponse],
    description="Fetch items details.",
    response_description="Fetch items details.",
    responses={
        404: {"description": "Items not found."}
    },
)
async def get_all_items(
        session: SessionDep,
        craft_material: Annotated[
            str, Query(description="Item code of items used as material for crafting.", pattern=r'^[a-zA-Z0-9_-]+$')
        ] = None,
        craft_skill: Annotated[
            CraftSkill, Query(description="Skill to craft items.")
        ] = None,
        max_level: Annotated[
            int, Query(description="Maximum level items.", ge=1)
        ] = None,
        min_level: Annotated[
            int, Query(description="Minimum level items.", ge=1)
        ] = None,
        name: Annotated[
            str, Query(description="Name of the item.", pattern=r'^[a-zA-Z0-9_-]+$')
        ] = None,
        type: Annotated[
            ItemType, Query(description="Type of items.")
        ] = None,
):
    query = select(Item)
    if craft_material:
        query = query.where(Item.craft.has(Craft.items.any(CraftItem.code == craft_material)))  # noqa: F821
    if craft_skill:
        query = query.where(Item.craft.has(Craft.skill == craft_skill.value))
    if max_level:
        query = query.where(Item.level <= max_level)
    if min_level:
        query = query.where(Item.level >= min_level)
    if name:
        query = query.where(
            or_(
                Item.name.like(f"%{name}%"),  # noqa: F821
                Item.code.like(f"%{name}%")  # noqa: F821
            )
        )
    if type:
        query = query.where(Item.type == type.value)
    items = session.exec(query).all()
    if not items:
        return error_response(404, "Items not found.")
    return items


@router.get(
    name="Get Item",
    path="/items/{code}",
    tags=["Items"],
    response_model=ItemResponse,
    description="Retrieve the details of a item.",
    response_description="Successfully fetched item.",
    responses={
        404: {"description": "Item not found."}
    },
)
async def get_item(
    session: SessionDep,
    code: Annotated[
        str, Path(description="The code of the item.", pattern=r'^[a-zA-Z0-9_-]+$')
    ]
):
    item = session.exec(select(Item).where(Item.code == code)).first()
    if not item:
        return error_response(404, "Item not found.")
    return item



