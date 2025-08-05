import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from binuma.datasets import mitchell2022


BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")
DONORNAME = "CB001"


class Mitchell2022Sample:
    def __init__(self) -> None:
        self.name = DONORNAME
        self.path2mut_type = Path(BASEPATH / f"mutType{self.name}.csv")
        self.path2mut_matrix = Path(BASEPATH / f"mutMatrix{self.name}.csv")
        self.mut_matrix = pd.read_csv(self.path2mut_matrix)
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
    with pytest.raises(AssertionError):
        mitchell2022.LoadMitchell().load_dataset(Path(""))


# def test_load_genotype_without_indels():
#    donor = Mitchell2022Sample()
#    processed = mitchell2022.load_genotype(
#        donor.path2mut_matrix,
#        donor.path2mut_type,
#        keep_indels=False,
#    )
#    assert not len(
#        set(processed.genotype.to_numpy().ravel()).symmetric_difference({0, 1})
#    )


# def test_load_genotype_with_indels():
#    donor = Mitchell2022Sample()
#    processed = mitchell2022.load_genotype(
#        donor.path2mut_matrix,
#        donor.path2mut_type,
#        keep_indels=True,
#    )
#    assert not len(
#        set(processed.genotype.to_numpy().ravel()).symmetric_difference({0, 1})
#    )


def test_load_mitchell():
    metadata, donors = mitchell2022.LoadMitchell().load_dataset(BASEPATH)
    idx = list(donors.keys())
    assert len(idx) == 1
    assert metadata.metadata.donor_id.unique().tolist() == [idx[0]]
    assert [d.name for d in donors.values()] == [DONORNAME]


def test_burden_nb_cells_mitchell():
    metadata, donors = mitchell2022.LoadMitchell().load_dataset(BASEPATH)
    donors = list(donors.values())
    assert len(donors) == 1
    assert mitchell2022.get_donors()[0]["cells"] == donors[0].genotype.get_nb_cells()


def test_sfs_nb_cells_mitchell(capsys):
    metadata, donors = mitchell2022.LoadMitchell().load_dataset(BASEPATH)
    donors = list(donors.values())
    assert len(donors) == 1
    assert mitchell2022.get_donors()[0]["cells"] == donors[0].genotype.get_nb_cells()
    donor_sfs = donors[0].sfs
    some_cells = metadata.metadata.cell.sample(10)
    donor_sfs_subsample = donors[0].compute_sfs(set(some_cells.to_list()))
    assert donor_sfs_subsample.sfs != donor_sfs
    assert len(donor_sfs_subsample.sfs.keys()) < len(donor_sfs.sfs.keys())
