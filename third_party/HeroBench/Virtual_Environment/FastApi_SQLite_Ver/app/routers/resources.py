import json
from enum import Enum
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Path
from sqlmodel import Field, Relationship, Session, SQLModel, select

from app.db import SessionDep
from app.routers import error_response


class GatherSkill(str, Enum):
    mining = "mining"
    woodcutting = "woodcutting"
    fishing = "fishing"


router = APIRouter()


class ResourceDropLink(SQLModel, table=True):
    resource_id: int | None = Field(default=None, foreign_key="resource.id", primary_key=True)
    drop_id: int | None = Field(default=None, foreign_key="resourceratedrop.id", primary_key=True)


class ResourceRateDrop(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    code: str = Field(description="Item code.", index=True, regex=r'^[a-zA-Z0-9_-]+$')
    rate: int = Field(description="Chance rate.", ge=1, default=1)
    min_quantity: int = Field(description="Minimum quantity.", ge=1, default=1)
    max_quantity: int = Field(description="Maximum quantity.", ge=1, default=1)

    resources: list["Resource"] = Relationship(back_populates="drops", link_model=ResourceDropLink)


class Resource(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(description="The name of the resource")
    code: str = Field(description="The code of the resource. This is the resource's unique identifier (ID).", index=True, unique=True, regex=r'^[a-zA-Z0-9_-]+$')
    skill: GatherSkill = Field(description="The skill required to gather this resource.", index=True)
    level: int = Field(description="The skill level required to gather this resource.", index=True, default=1, ge=1)
    drops: list[ResourceRateDrop] = Relationship(back_populates="resources", link_model=ResourceDropLink)


class DropRateResponse(SQLModel, ordered=True):
    code: Annotated[str, Field(description="Item code.", regex=r'^[a-zA-Z0-9_-]+$')]
    rate: Annotated[int, Field(description="Chance rate.", ge=1, default=1)]
    min_quantity: Annotated[int, Field(description="Minimum quantity.", ge=1, default=1)]
    max_quantity: Annotated[int, Field(description="Maximum quantity.", ge=1, default=1)]


class ResourceResponse(SQLModel, ordered=True):
    name: Annotated[str, Field(description="The name of the resource")]
    code: Annotated[str, Field(description="The code of the resource. This is the resource's unique identifier (ID).", regex=r'^[a-zA-Z0-9_-]+$')]
    skill: Annotated[GatherSkill, Field(description="The skill required to gather this resource.")]
    level: Annotated[int, Field(description="The skill level required to gather this resource.", default=1, ge=1)]
    drops: Annotated[list[DropRateResponse], Field(description="The drops of this resource.")]


def load_resources_data(session: Session):
    with open('app/Data/resources.json') as f:
        resources_data = json.load(f)
        for resource_data in resources_data:
            drops_data = resource_data.pop('drops', [])
            drops = []
            for drop_data in drops_data:
                drop = ResourceRateDrop(**drop_data)
                session.add(drop)
                session.commit()
                session.refresh(drop)
                drops.append(drop)
            new_resource = Resource(**resource_data)
            session.add(new_resource)
            session.commit()
            session.refresh(new_resource)
            for drop in drops:
                resource_drop_link = ResourceDropLink(resource_id=new_resource.id, drop_id=drop.id)
                session.add(resource_drop_link)
                session.commit()
    return True


@router.get(
    name="Get All Resources",
    path='/resources',
    tags=['Resources'],
    response_model=list[ResourceResponse],
    description="Fetch resources details.",
    response_description="Successfully fetched resources details.",
    responses={
        404: {"description": "Resources not found."}
    },
)
async def get_all_resources(
        session: SessionDep,
        drop: Annotated[
            str, Query(description="Item code of the drop.", regex=r'^[a-zA-Z0-9_-]+$', example="copper_ore")
        ] = None,
        max_level: Annotated[
            int, Query(description="Skill maximum level.", ge=1)
        ] = None,
        min_level: Annotated[
            int, Query(description="Skill minimum level.", ge=1)
        ] = None,
        skill: Annotated[
            GatherSkill, Query(description="The code of the skill.")
        ] = None,
):
    query = select(Resource)
    if drop:
        query = query.where(Resource.drops.any(ResourceRateDrop.code == drop))  # noqa: F821
    if max_level:
        query = query.where(Resource.level <= max_level)
    if min_level:
        query = query.where(Resource.level >= min_level)
    if skill:
        query = query.where(Resource.skill == skill.value)
    resources = session.exec(query).all()
    if not resources:
        return error_response(404, "Resources not found.")
    return resources


@router.get(
    name="Get Resource",
    path="/resources/{code}",
    tags=['Resources'],
    response_model=ResourceResponse,
    description="Retrieve the details of a resource.",
    response_description="Successfully fetched resource.",
    responses={
        404: {"description": "Resource not found."}
    },
)
async def get_resource(
        session: SessionDep,
        code: Annotated[
            str, Path(description="The code of the resource.", regex=r'^[a-zA-Z0-9_-]+$', example="copper_rocks")
        ]
):
    resource = session.exec(select(Resource).where(Resource.code == code)).first()
    if not resource:
        return error_response(404, "Resource not found.")
    return resource
