from typing import Annotated
import pandas as pd
from pydantic import UUID4, BaseModel, Field
from pandantic import Pandantic
from binuma import Dataset


class MetadataDataset:
    """A dataframe with information about each sequenced colony (i.e. sample or cell) for a specific [`Experiment`]."""

    metadata: pd.DataFrame

    def __init__(self, metadata: pd.DataFrame) -> None:
        validator = Pandantic(schema=Metadata)
        try:
            self.metadata = validator.validate(dataframe=metadata, errors="raise")
        except ValueError as e:
            e.add_note(f"`metadata` doesn't match its schema\n{metadata}")
            raise e


class Metadata(BaseModel):
    idx: UUID4
    name: str
    sample: Annotated[int, Field(int, strict=True, ge=0)]
    pop_size: Annotated[int, Field(int, strict=True, ge=0)]
    age: Annotated[int, Field(int, strict=True, ge=0, le=120)]
    dataset: Dataset
    status: str
