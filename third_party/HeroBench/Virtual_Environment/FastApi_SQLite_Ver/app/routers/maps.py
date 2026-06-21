import json
from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Path
from sqlmodel import Field, Relationship, Session, SQLModel, select

from app.db import SessionDep
from app.routers import error_response


class ContentType(str, Enum):
    monster = "monster"
    resource = "resource"
    workshop = "workshop"
    bank = "bank"
    grand_exchange = "grand_exchange"
    tasks_master = "tasks_master"


router = APIRouter()


class MapContentLink(SQLModel, table=True):
    map_id: int | None = Field(default=None, foreign_key="map.id", primary_key=True)
    map_content_id: int | None = Field(default=None, foreign_key="mapcontent.id", primary_key=True)


class MapContent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    type: ContentType = Field(index=True, description="Type of the content.")
    code: str = Field(index=True, description="Code of the content.", regex=r'^[a-zA-Z0-9_-]+$')

    maps: list["Map"] = Relationship(back_populates="content", link_model=MapContentLink)


class Map(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(description="Name of the map.")
    skin: str = Field(description="Skin of the map.")
    x: int = Field(index=True, description="The position x of the map.")
    y: int = Field(index=True, description="The position y of the map.")
    content: MapContent | None = Relationship(back_populates="maps", link_model=MapContentLink)


class MapContentResponse(SQLModel, ordered=True):
    type: Annotated[ContentType, Field(description="Type of the content.")]
    code: Annotated[str, Field(description="Code of the content.", regex=r'^[a-zA-Z0-9_-]+$')]


class MapResponse(SQLModel, ordered=True):
    name: Annotated[str, Field(description="Name of the map.")]
    skin: Annotated[str, Field(description="Skin of the map.")]
    x: Annotated[int, Field(description="Position X of the map.")]
    y: Annotated[int, Field(description="Position Y of the map.")]
    content: Annotated[MapContentResponse | None, Field(description="Content of the map.")]


def load_maps_data(session: Session):
    with open("app/Data/maps.json") as file:
        maps_data = json.load(file)
        for map_data in maps_data:
            map_content = None
            map_content_data = map_data.pop('content', None)
            if map_content_data:
                map_content = MapContent(**map_content_data)
                session.add(map_content)
            new_map = Map(**map_data, content=map_content)
            session.add(new_map)
            session.commit()
    return True


@router.get(
    name="Get All Maps",
    path="/maps",
    tags=["Maps"],
    response_model=list[MapResponse],
    description="Fetch maps details.",
    response_description="Successfully fetched maps details.",
    responses={
        404: {"description": "Maps not found."}
    }
)
async def get_all_maps(
        session: SessionDep,
        content_code: Annotated[
            str | None, Query(description="Content code on the map.", regex=r'^[a-zA-Z0-9_-]+$')
        ] = None,
        content_type: Annotated[
            ContentType | None, Query(description="Type of content on the map.")
        ] = None
):
    query = select(Map)
    if content_code:
        query = query.where(Map.content.has(MapContent.code == content_code))
    if content_type:
        query = query.where(Map.content.has(MapContent.type == content_type.value))
    maps = session.exec(query).all()
    if not maps:
        return error_response(404, "Maps not found.")
    return maps


@router.get(
    name="Get Map",
    path="/maps/{x}/{y}",
    tags=["Maps"],
    response_model=MapResponse,
    description="Retrieve the details of a map.",
    response_description="Successfully fetched map.",
    responses={
        404: {"description": "Map not found."}
    }
)
async def get_map(
        session: SessionDep,
        x: Annotated[int, Path(description="The position x of the map.")],
        y: Annotated[int, Path(description="The position X of the map.")]
):
    map_tile = session.exec(select(Map).filter_by(x=x, y=y)).first()
    if not map_tile:
        return error_response(404, "Map not found.")
    return map_tile




