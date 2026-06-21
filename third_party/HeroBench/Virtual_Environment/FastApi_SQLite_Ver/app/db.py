import os
from typing import Annotated

from fastapi import Depends
from sqlmodel import Session, SQLModel, create_engine, Field

from enum import Enum

# HEROBENCH_DB lets each server instance own its SQLite file, so parallel PATSP /
# R-Zero arms don't share (and on-startup rm_db-wipe) one another's world. Default unchanged.
sqlite_file_name = os.environ.get("HEROBENCH_DB", "artifact.db")
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


class ActionType(str, Enum):
    create_character = "create_character"
    create_custom_character = "create_custom_character"
    delete_character = "delete_character"
    move = "move"
    equip_item = "equip_item"
    unequip_item = "unequip_item"
    fight = "fight"
    gather = "gather"
    craft = "craft"
    delete_item = "delete_item"
    give_item = "give_item"
    buy_item = "buy_item"
   
class CharacterLog(SQLModel, table=True):
    id: Annotated[int | None, Field(default=None, primary_key=True)]
    character_name: Annotated[str, Field(description="character name", index=True)]
    action_type: Annotated[ActionType, Field(description="action type", index=True)]
    log: Annotated[str, Field(description="log of the performed action")]


def init_db():
    """
    Initializes the SQLite database.
    """
    SQLModel.metadata.create_all(engine)


def rm_db():
    """
    Deletes the SQLite database.
    """
    SQLModel.metadata.drop_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
