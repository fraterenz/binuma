import pytest
from binuma.datasets.simulations import hsc
from pathlib import Path

BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")


@pytest.mark.parametrize(
    "path,age,cells,sample,idx",
    [
        (
            BASEPATH
            / Path(
                "10000cells/sfs/0dot0years/4mu_0dot01mean_0dot01std_1tau_10000cells_260idx.json"
            ),
            0,
            10_000,
            10_000,
            "0-10000-10000-260",
        ),
        (
            BASEPATH
            / Path(
                "5000cells/sfs/5dot0years/4mu_0dot01mean_0dot01std_1tau_10000cells_260idx.json"
            ),
            5,
            10_000,
            5000,
            "5-10000-5000-260",
        ),
    ],
)
def test_load_simulation_from_path(path, age, cells, sample, idx):
    simulation = hsc.load_simulation_from_path(path)
    assert simulation.age == age
    assert simulation.cells == sample
    assert simulation.idx == idx
    assert simulation.pop_size == cells
