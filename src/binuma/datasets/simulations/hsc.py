import sys
import json
from pathlib import Path
from futils import snapshot
from typing import NewType, Sequence

from binuma import Experiment
from binuma.datasets.simulations import parameters
from binuma.sfs import DonorSfs, DonorsSfs, Sfs

AgeSims = NewType("AgeSims", float)


def load_sfs_from_path(path: Path) -> snapshot.Histogram:
    try:
        hist = snapshot.histogram_from_file(path)
    except json.JSONDecodeError as e:
        print(f"Error in opening {path} {e}")
        sys.exit(1)
    return hist


def get_idx_from_params(params: parameters.Parameters) -> str:
    return f"{params.cells}_{params.age}_{params.idx}"


class SimulationHsc(DonorSfs):
    """The SFS simulated with the binary `hsc`."""

    def __init__(
        self,
        name: str,
        age: int,
        is_healthy: bool,
        sfs: Sfs,
        idx: str,
        cells: int,
        pop_size: int,
    ) -> None:
        super().__init__(name, age, is_healthy, Experiment.HSCSIMULATIONS, sfs, cells)
        self.idx = idx
        self.pop_size = pop_size


def load_simulation_from_path(path: Path) -> SimulationHsc:
    assert path.is_file(), f"cannot find SFS file {path}"
    params = parameters.parameters_from_path(path)
    print(parameters)
    return SimulationHsc(
        path.stem,
        params.age,
        True,
        Sfs(load_sfs_from_path(path)),
        f"{params.age}-{params.cells}-{params.sample}-{params.idx}",
        params.sample,
        params.cells,
    )


class SimulationsHsc(DonorsSfs):
    """A collection of SFS simulated with the binary `hsc`."""

    def __init__(self, donors: Sequence[SimulationHsc]) -> None:
        super().__init__(donors)
        list_idx = [s.idx for s in donors]
        set_idx = set(list_idx)
        assert len(list_idx) == len(set_idx), "Found non unique simulations' idx"
        self.idx = set_idx

    def get_simulation_by_idx(self, idx) -> SimulationHsc:
        assert idx in self.idx, "`idx` not in these simulations"
        # we know that there is only one entry because we checked in the
        # constructor that id are unique
        return [s for s in self.donors if s.idx == idx][0]


def load_simulations(
    path2dir: Path,
) -> SimulationsHsc:
    assert path2dir.is_dir()
    realisations = list()

    for path in path2dir.iterdir():
        if path.is_dir():
            for p in path.glob("*.json"):
                realisations.append(load_simulation_from_path(p))

    print(f"loaded {len(realisations)} SFS from {path2dir}")

    return SimulationsHsc(realisations)


def load_all_sfs_by_age(path2dir: Path):
    return load_simulations(path2dir)
