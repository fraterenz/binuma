import pandas as pd
import numpy as np
from hypothesis import given, strategies as st
import pytest
from io import StringIO
from binuma.datasets import Dataset
from binuma.donor import Donor
from binuma.genotype import (
    BinaryMutationMatrix,
    EntryIsNan,
    NonUniqueCells,
    NonUniqueMutations,
    filter_cells_from_matrix,
)


def valid_age(n: int) -> bool:
    return (n > 0) & (n < 120)


def valid_size(n: int) -> bool:
    return n > 0


def test_genotype_floats():
    BinaryMutationMatrix(pd.DataFrame([[1, 1, 1], [0, 1, 0]], dtype=float))


def test_wrong_genotype():
    with pytest.raises(ValueError):
        BinaryMutationMatrix(pd.DataFrame(np.random.random_sample(size=(40, 40))))


def test_get_nb_cells():
    geno = pd.DataFrame([[1, 1, 1, 0], [0, 1, 0, 0]], dtype=float)
    print(geno)
    assert BinaryMutationMatrix(geno).get_nb_cells() == geno.shape[1] - 1


def test_get_nb_mutations():
    geno = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    print(geno)
    assert BinaryMutationMatrix(geno).get_nb_cells() == geno.shape[0] - 1


def create_binary_mut() -> pd.DataFrame:
    data = """PD43947h_lo0002_hum\tPD43947h_lo0003_hum\tPD43947h_lo0004_hum\tPD43947h_lo0005_hum
10-110257574-G-A\t0.5\t1\t0\t0
10-114594383-C-T\t0\t1\t0\t0
10-124679941-T-C\t0\t0\t0\t0
10-12684595-G-C\t0\t0\t1\t1
10-132754174-G-A\t0\t0\t0\t0
10-133246550-G-A\t0\t1\t0\t0
10-133335354-C-T\t0\t0\t0\t0
10-134888630-G-A\t0\t1\t1\t1
10-15817209-T-C\t0\t0\t1\t1
10-1937569-G-C\tNaN\t0\t0\t0
 """
    return pd.read_csv(StringIO(data), sep="\t")


def create_cleaned_binary_mut() -> BinaryMutationMatrix:
    return BinaryMutationMatrix(create_binary_mut().fillna(0).map(int))


def test_empty_matrix():
    with pytest.raises(AssertionError):
        BinaryMutationMatrix(pd.DataFrame())


def test_nan_0_dot_5_values():
    binary_mut = create_binary_mut()
    with pytest.raises(EntryIsNan):
        BinaryMutationMatrix(binary_mut)


def test_duplicated_cells():
    binary_mut = create_cleaned_binary_mut()
    binary_mut = (
        pd.concat([binary_mut.genotype, binary_mut.genotype.iloc[:2]], axis=1)
        .fillna(0)
        .map(int)
    )

    with pytest.raises(NonUniqueCells):
        BinaryMutationMatrix(binary_mut)


def test_duplicated_mutations():
    binary_mut = create_cleaned_binary_mut()
    binary_mut = (
        pd.concat([binary_mut.genotype, binary_mut.genotype.iloc[:2]], axis=0)
        .fillna(0)
        .map(int)
    )
    with pytest.raises(NonUniqueMutations):
        BinaryMutationMatrix(binary_mut)


# def test_polytomies():
#    assert create_cleaned_binary_mut().polytomies == 2


def test_filter_cells():
    binary_mut = create_cleaned_binary_mut()

    cells2keep = set(binary_mut.genotype.columns[:2].to_list())
    filtered = filter_cells_from_matrix(binary_mut, cells2keep)
    assert len(set(filtered.genotype.columns.to_list()) - cells2keep) == 0


@given(
    st.uuids(version=4),
    st.text(),
    st.integers().filter(valid_size),
    st.integers().filter(valid_age),
    st.sampled_from(["healthy", "cancer", "bcr_abl1"]),
    st.sampled_from(Dataset),
)
def test_donor_genotype(idx, name, pop_size, age, status, dataset):
    binary_mut = create_cleaned_binary_mut()
    d = Donor(
        idx=idx,
        name=name,
        age=age,
        status=status,
        pop_size=pop_size,
        dataset=dataset,
        genotype=binary_mut,
    )
    assert d.name == name
    assert d.age == age
    assert d.status == status
    assert d.pop_size == pop_size
    assert d.dataset == dataset
