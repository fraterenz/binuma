from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from binuma.datasets import mitchell2022
from pandantic import Pandantic
from pydantic import BaseModel


BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")
DONORNAME = "CB001"


class MutType(str, Enum):
    SNV = "SNV"
    INDEL = "INDEL"


@dataclass
class MutMatrixSchema(BaseModel):
    x: MutType


class Mitchell2022Sample:
    def __init__(self) -> None:
        self.name = DONORNAME
        self.path2mut_type = Path(BASEPATH / f"mutType{self.name}.csv")
        self.path2mut_matrix = Path(BASEPATH / f"mutMatrix{self.name}.csv")
        self.mut_matrix = pd.read_csv(self.path2mut_matrix)
        validator = Pandantic(schema=MutMatrixSchema)
        df = pd.read_csv(self.path2mut_type)[["x"]]
        self.mut_type = validator.validate(dataframe=df, errors="raise")
        self.mut_type = self.mut_type["x"].squeeze()


def test_load_genotype_wrong_format(monkeypatch):
    def mock_df(*args, **kwargs):
        # hack
        if "index_col" in kwargs:
            return pd.DataFrame(np.random.random_sample(size=(10, 10)))
        else:
            return pd.Series(np.random.random_sample(size=(10,)))

    # this will patch both calls to pd.read_csv, but one should return a
    # DataFrame whereas the other should return a Series, hence the hack above
    monkeypatch.setattr(pd, "read_csv", mock_df)
    with pytest.raises(ValueError):
        mitchell2022.load_genotype(Path(""), Path(""), keep_indels=False)


def test_load_genotype_without_indels():
    donor = Mitchell2022Sample()
    processed = mitchell2022.load_genotype(
        donor.path2mut_matrix,
        donor.path2mut_type,
        keep_indels=False,
    )
    assert not len(
        set(processed.genotype.to_numpy().ravel()).symmetric_difference({0, 1})
    )


def test_load_genotype_with_indels():
    donor = Mitchell2022Sample()
    processed = mitchell2022.load_genotype(
        donor.path2mut_matrix,
        donor.path2mut_type,
        keep_indels=True,
    )
    assert not len(
        set(processed.genotype.to_numpy().ravel()).symmetric_difference({0, 1})
    )


def test_load_mitchell():
    donors = mitchell2022.load_mitchell2022(BASEPATH)
    assert [d.metadata.name for d in donors.donors] == [DONORNAME]


def test_sfs_nb_cells_mitchell():
    donors = mitchell2022.load_mitchell2022(BASEPATH)
    assert (
        mitchell2022.get_donors()[0]["cells"]
        == donors.donors[0].genotype.get_nb_cells()
    )
