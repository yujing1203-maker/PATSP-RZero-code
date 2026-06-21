import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Path
from sqlmodel import Field, Relationship, Session, SQLModel, select

from app.db import SessionDep
from app.routers import error_response

router = APIRouter()


class MonsterDropLink(SQLModel, table=True):
    monster_id: int | None = Field(default=None, foreign_key="monster.id", primary_key=True)
    drop_id: int | None = Field(default=None, foreign_key="monsterdrop.id", primary_key=True)


class MonsterDrop(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    code: str = Field(description="Item code.", index=True, regex=r'^[a-zA-Z0-9_-]+$')
    rate: int = Field(description="Chance rate.", ge=1, default=1)
    min_quantity: int = Field(description="Minimum quantity.", ge=1, default=1)
    max_quantity: int = Field(description="Maximum quantity.", ge=1, default=1)

    monsters: list["Monster"] = Relationship(back_populates="drops", link_model=MonsterDropLink)


class Monster(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(description="Monster name.")
    code: str = Field(description="The code of the monster. This is the monster's unique identifier (ID).", index=True, unique=True, regex=r'^[a-zA-Z0-9_-]+$')
    level: int = Field(description="Monster level.", index=True, ge=1)
    hp: int = Field(description="Monster hit points.", default=0, ge=0)
    attack_fire: int = Field(description="Monster fire attack.", default=0)
    attack_earth: int = Field(description="Monster earth attack.", default=0)
    attack_water: int = Field(description="Monster water attack.", default=0)
    attack_air: int = Field(description="Monster air attack.", default=0)
    res_fire: int = Field(description="Monster % fire resistance.", default=0)
    res_earth: int = Field(description="Monster % earth resistance.", default=0)
    res_water: int = Field(description="Monster % water resistance.", default=0)
    res_air: int = Field(description="Monster % air resistance.", default=0)
    min_gold: int = Field(description="Monster minimum gold drop.", default=0)
    max_gold: int = Field(description="Monster maximum gold drop.", default=0)
    drops: list[MonsterDrop] = Relationship(back_populates="monsters", link_model=MonsterDropLink)


class DropResponse(SQLModel, ordered=True):
    code: Annotated[str, Field(description="Item code.", regex=r'^[a-zA-Z0-9_-]+$')]
    rate: Annotated[int, Field(description="Chance rate.", ge=1, default=1)]
    min_quantity: Annotated[int, Field(description="Minimum quantity.", ge=1, default=1)]
    max_quantity: Annotated[int, Field(description="Maximum quantity.", ge=1, default=1)]


class MonsterResponse(SQLModel, ordered=True):
    name: Annotated[str, Field(description="Name of the monster.")]
    code: Annotated[str, Field(description="The code of the monster. This is the monster's unique identifier (ID).", regex=r'^[a-zA-Z0-9_-]+$')]
    level: Annotated[int, Field(description="Monster level.", ge=1)]
    hp: Annotated[int, Field(description="Monster hit points.", default=0, ge=0)]
    attack_fire: Annotated[int, Field(description="Monster fire attack.", default=0)]
    attack_earth: Annotated[int, Field(description="Monster earth attack.", default=0)]
    attack_water: Annotated[int, Field(description="Monster water attack.", default=0)]
    attack_air: Annotated[int, Field(description="Monster air attack.", default=0)]
    res_fire: Annotated[int, Field(description="Monster % fire resistance.", default=0)]
    res_earth: Annotated[int, Field(description="Monster % earth resistance.", default=0)]
    res_water: Annotated[int, Field(description="Monster % water resistance.", default=0)]
    res_air: Annotated[int, Field(description="Monster % air resistance.", default=0)]
    min_gold: Annotated[int, Field(description="Monster minimum gold drop.", default=0, ge=0)]
    max_gold: Annotated[int, Field(description="Monster maximum gold drop.", default=0, ge=0)]
    drops: Annotated[list[DropResponse], Field(description="Monster drops. This is a list of items that the monster drops after killing the monster.")]


def load_monsters_data(session: Session):
    with open('app/Data/monsters.json') as f:
        monsters_data = json.load(f)
        for monster_data in monsters_data:
            drops_data = monster_data.pop('drops', [])
            drops = []
            for drop_data in drops_data:
                drop = MonsterDrop(**drop_data)
                session.add(drop)
                drops.append(drop)
            new_monster = Monster(**monster_data, drops=drops)
            session.add(new_monster)
            # for drop in drops:
            #     monster_drop_link = MonsterDropLink(monster_id=new_monster.id, drop_id=drop)
            #     session.add(monster_drop_link)
            session.commit()
    return True


@router.get(
    name="Get All Monsters",
    path='/monsters',
    tags=['Monsters'],
    response_model=list[MonsterResponse],
    description="Fetch monsters details.",
    response_description="Successfully fetched monsters details.",
    responses={
        404: {"description": "Monsters not found."}
    },
)
async def get_all_monsters(
        session: SessionDep,
        drop: Annotated[
            str, Query(description="Item code of the drop.", regex=r'^[a-zA-Z0-9_-]+$', example="green_slimeball")
        ] = None,
        max_level: Annotated[
            int, Query(description="Monster maximum level.", ge=1)
        ] = None,
        min_level: Annotated[
            int, Query(description="Monster minimum level.", ge=1)
        ] = None,
):
    query = select(Monster)
    if drop:
        query = query.where(Monster.drops.any(MonsterDrop.code == drop))  # noqa: F821
    if max_level:
        query = query.where(Monster.level <= max_level)
    if min_level:
        query = query.where(Monster.level >= min_level)
    monsters = session.exec(query).all()
    if not monsters:
        return error_response(404, "Monsters not found.")
    return monsters


@router.get(
    name="Get Monster",
    path="/monsters/{code}",
    tags=['Monsters'],
    response_model=MonsterResponse,
    description="Retrieve the details of a monster.",
    response_description="Successfully fetched monster.",
    responses={
        404: {"description": "Monster not found."}
    },
)
async def get_monster(
        session: SessionDep,
        code: Annotated[
            str, Path(description="The code of the monster.", regex=r'^[a-zA-Z0-9_-]+$')
        ]
):
    monster = session.exec(select(Monster).where(Monster.code == code)).first()
    if not monster:
        return error_response(404, "Monster not found.")
    return monster
