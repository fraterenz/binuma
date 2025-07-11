from typing import Dict
from futils.snapshot import Histogram
from hypothesis import given, strategies as st
import pytest
import pandas as pd
from scipy import stats

from binuma import Dataset
from binuma.genotype import BinaryMutationMatrix, DonorGenotype
from binuma.sfs import DonorsGenotype, Sfs


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


@given(
    st.text(),
    st.integers().filter(valid_size),
    st.integers().filter(valid_age),
    st.sampled_from(["healthy", "cancer"]),
    st.sampled_from(Dataset),
)
def test_from_genotyped_donors_to_donors(name, pop_size, age, status, dataset):
    donors = DonorsGenotype(
        [
            DonorGenotype(
                name=name,
                age=age,
                status=status,
                dataset=dataset,
                genotype=BinaryMutationMatrix(
                    genotype=pd.DataFrame([[0, 1, 0], [1, 1, 0]])
                ),
            )
        ]
    )
    donors, metadata = donors.into_sfs()
    idx = metadata.metadata["idx"][0]
    sfs = donors[idx]
    assert list(sfs.sfs.keys()) == [1, 2]
    assert list(sfs.sfs.values()) == [1, 1]
    assert sfs.compute_entropy() > 0
    assert metadata.metadata["name"].iloc[0] == name
