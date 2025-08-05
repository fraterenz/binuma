from typing import Dict
from futils.snapshot import Histogram
import pytest
import pandas as pd
from scipy import stats

from binuma.genotype import BinaryMutationMatrix
from binuma.sfs import Sfs, compute_sfs


def valid_age(n: int) -> bool:
    return (n > 0) & (n < 120)


def valid_size(n: int) -> bool:
    return n > 0


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


def test_compute_sfs():
    sfs = compute_sfs(
        genotype=BinaryMutationMatrix(genotype=pd.DataFrame([[0, 1, 0], [1, 1, 0]]))
    )
    assert list(sfs.sfs.keys()) == [1, 2]
    assert list(sfs.sfs.values()) == [1, 1]
    assert sfs.compute_entropy() > 0
