import pytest
from binuma.datasets.simulations import hsc
from pathlib import Path

BASEPATH = Path(Path(__file__).resolve().parent / "fixtures")


@pytest.mark.parametrize(
    "path,age,cells,sample",
    [
        (
            BASEPATH
            / Path(
                "10000cells/sfs/0dot0years/4mu_0dot01mean_0dot01std_1tau_10000cells_260idx.json"
            ),
            0,
            10_000,
            10_000,
        ),
        (
            BASEPATH
            / Path(
                "5000cells/sfs/5dot0years/4mu_0dot01mean_0dot01std_1tau_10000cells_260idx.json"
            ),
            5,
            10_000,
            5000,
        ),
    ],
)
def test_load_simulation_from_path(path, age, cells, sample):
    simulation = hsc.load_simulation_from_path(path)
    assert simulation.metadata.age == age
    assert simulation.metadata.sample == sample
    assert simulation.metadata.pop_size == cells
