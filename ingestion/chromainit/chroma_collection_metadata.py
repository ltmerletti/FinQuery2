import datetime
from typing import Optional
from pydantic import BaseModel, Field


def get_current_time_in_iso_8601_format_utc() -> str:
    # returns the current UTC time in ISO format
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class CollectionMetadata(BaseModel):
    data_source: str = Field(
        ...,
        description="Document source description"
    )

    embedding_model_name: str = Field(
        ...,
        description="Name of embedding model"
    )

    created_by: str = Field(
        description="Author of the collection",
        default="unknown"
    )

    project_id: Optional[str] = Field(
        None,
        description="Name of the project"
    )

    parser_version: Optional[str] = Field(
        None,
        description="Version of the document parser (in case it's updated)"
    )

    created_at_iso: str = Field(
        default_factory=get_current_time_in_iso_8601_format_utc(),
        description="The ISO 8601 timestamp of creation time in UTC"
    )
