import sys
import json
from uuid import uuid4
import pandas as pd
from pathlib import Path
from futils import snapshot
from typing import NewType, Tuple

from binuma import Dataset
from binuma.datasets.simulations import parameters
from binuma.metadata import Metadata, MetadataDataset
from binuma.sfs import DonorsSfs, Sfs

AgeSims = NewType("AgeSims", float)


def load_sfs_from_path(path: Path) -> snapshot.Histogram:
    try:
        hist = snapshot.histogram_from_file(path)
    except json.JSONDecodeError as e:
        print(f"Error in opening {path} {e}")
        sys.exit(1)
    return hist


class SimulationHsc:
    """The SFS simulated with the binary `hsc`."""

    def __init__(
        self,
        name: str,
        age: int,
        is_healthy: bool,
        sfs: Sfs,
        cells: int,
        pop_size: int,
    ) -> None:
        self.metadata = Metadata(
            idx=uuid4(),
            name=name,
            age=age,
            sample=cells,
            pop_size=pop_size,
            status="healthy" if is_healthy else "cancer",
            dataset=Dataset.HSCSIMULATIONS,
        )
        self.sfs = sfs


def load_simulation_from_path(path: Path) -> SimulationHsc:
    assert path.is_file(), f"cannot find SFS file {path}"
    params = parameters.parameters_from_path(path)
    print(parameters)
    return SimulationHsc(
        path.stem,
        params.age,
        True,
        Sfs(load_sfs_from_path(path)),
        params.sample,
        params.cells,
    )


def load_simulations(
    path2dir: Path,
) -> Tuple[DonorsSfs, MetadataDataset]:
    assert path2dir.is_dir()
    realisations, metadata = dict(), list()

    for path in path2dir.iterdir():
        if path.is_dir():
            for p in path.glob("*.json"):
                sim = load_simulation_from_path(p)
                metadata.append(sim.metadata)
                realisations[sim.metadata.idx] = sim.sfs

    print(f"loaded {len(realisations)} SFS from {path2dir}")

    return DonorsSfs(realisations), MetadataDataset(pd.DataFrame.from_records(metadata))


def load_all_sfs_by_age(path2dir: Path):
    return load_simulations(path2dir)
