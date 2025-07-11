import numpy as np
from typing import Tuple
import pandas as pd
import pytest

from binuma.genotype import BinaryMutationMatrix
from binuma.datasets.mitchell2022 import map_unsure_reads_to_0_counts


def create_float_data_without_unsure_reads() -> pd.DataFrame:
    return pd.DataFrame([[1, 1, 1], [0, 1, 0]], dtype=float)


def create_int_data_without_unsure_reads() -> pd.DataFrame:
    return pd.DataFrame([[1, 1, 1], [0, 1, 0]], dtype=int)


def create_data_with_unsure_reads() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame([[1, 1, 1], [0.5, 1, 0]], dtype=float),
        pd.DataFrame([[1, 1, 1], [0, 1, 0]], dtype=int),
    )


def create_data_with_random_unsure_reads() -> pd.DataFrame:
    data = np.random.rand(4, 2)
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "mut_matrix,expected",
    [
        (
            create_float_data_without_unsure_reads(),
            BinaryMutationMatrix(create_float_data_without_unsure_reads()),
        ),
        (
            create_int_data_without_unsure_reads(),
            BinaryMutationMatrix(create_int_data_without_unsure_reads()),
        ),
    ],
)
def test_map_unsure_reads_to_0_counts_without_unsure_reads(
    mut_matrix: pd.DataFrame,
    expected,
):
    assert (
        (map_unsure_reads_to_0_counts(mut_matrix).genotype == expected.genotype)
        .all()
        .all()
    )


def test_map_unsure_reads_to_0_counts_random_wrong_data():
    with pytest.raises(ValueError):
        map_unsure_reads_to_0_counts(create_data_with_random_unsure_reads())


def test_map_unsure_reads_to_0_counts_wrong_data():
    with pytest.raises(ValueError):
        map_unsure_reads_to_0_counts(pd.DataFrame([[1, 10]]))


def test_map_unsure_reads_to_0_counts():
    data, expected = create_data_with_unsure_reads()
    assert (map_unsure_reads_to_0_counts(data).genotype == expected).all().all()
