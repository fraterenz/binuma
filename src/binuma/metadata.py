import logging

import pandas as pd
from pandera.dtypes import Category
import pandera.pandas as pa
from pandera.typing import Series
from pandera import errors

log = logging.getLogger(__name__)


class MetadataDataset:
    """A dataframe with information about each mutation and each cell for a
    collection of donors from a specific [`Experiment`]."""

    # cell_idx:
    metadata: pd.DataFrame

    def __init__(self, metadata: pd.DataFrame) -> None:
        log.info("Creating the metadata")
        try:
            self.metadata = Metadata.validate(metadata)
        except errors.SchemaError as e:
            e.add_note(f"`metadata` doesn't match its schema\n{metadata}")
            raise e


class Metadata(pa.DataFrameModel):
    donor_id: Series[Category]
    donor: Series[Category]
    age: Series[int]
    dataset: Series[Category]
    mutation: Series[str]
    mutation_status: Series[Category]
    cell: Series[Category]
    cell_status: Series[Category]
