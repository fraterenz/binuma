from dataclasses import dataclass
from typing import Optional
import pandas as pd
from pydantic import BaseModel
from pydantic.types import StrictInt
from pandantic import Pandantic
from binuma import Experiment


@dataclass
class MetadataSchema(BaseModel):
    donor: str
    sample: str
    age: StrictInt
    status: Optional[str] = None


class Metadata:
    """A dataframe with information about each sequenced colony (i.e. sample or cell) for a specific [`Experiment`]."""

    experiment: Experiment
    df: pd.DataFrame

    def __init__(self, experiment: Experiment, metadata: pd.DataFrame) -> None:
        validator = Pandantic(schema=MetadataSchema)
        try:
            self.df = validator.validate(dataframe=metadata, errors="raise")
        except ValueError as e:
            e.add_note("`metadata` doesn't match its schema {:Metadata().__repr__()}")
            raise e
        self.experiment = experiment
