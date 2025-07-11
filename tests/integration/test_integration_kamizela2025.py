from pathlib import Path

from binuma.datasets.kamizela2025 import (
    load_kamizela2025_genotype,
    load_metadata_kamizela,
)


BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")


def test_load_metadata():
    meta = load_metadata_kamizela(BASEPATH / "per_sample_statistics.n834.csv").metadata
    assert meta.shape[0] > 0
    assert meta.status.all()


def test_load_sfs():
    donors = load_kamizela2025_genotype(BASEPATH)
    assert len(donors.donors) == 1
    donors, meta = donors.into_sfs()
    assert {k for k in donors.keys() if k in meta.metadata["idx"].unique()}
