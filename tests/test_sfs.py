from typing import Dict
from futils.snapshot import Histogram
import pytest
from scipy import stats

from binuma.sfs import Sfs


@pytest.mark.parametrize(
    "sfs_dict,entropy",
    [
        ({1: 2}, stats.entropy([1, 1], base=2)),  # 2 cells, with 1 mut each
        ({1: 1}, stats.entropy([1], base=2)),  # 1 cell, 1 mutation
        (
            {1: 1, 2: 1},
            stats.entropy([1, 2], base=2),
        ),  # 2 cells, 1 with 1 mut, 1 with 2 muts
    ],
)
def test_entropy(sfs_dict: Dict[int, int], entropy: float):
    mysfs = Sfs(Histogram(sfs_dict))
    assert mysfs.compute_entropy() == entropy
