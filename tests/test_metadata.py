from hypothesis import given, strategies as st
import pandas as pd
import pytest
from binuma.datasets import Dataset
from binuma.metadata import MetadataDataset


def valid_age(n: int) -> bool:
    return (n > 0) & (n < 120)


def valid_size(n: int) -> bool:
    return n > 0


@pytest.mark.skip(reason="no way of currently testing this")
@given(
    st.text(),
    st.text(),
    st.text(),
    st.integers(),
    st.text(),
    st.sampled_from(Dataset),
    st.text(),
    st.text(),
)
def test_metadata(
    name, mutation, mutation_status, age, status, dataset, cell, cell_status
):
    MetadataDataset(
        pd.DataFrame.from_dict(
            dict(
                donor=name,
                age=age,
                dataset=dataset,
                mutation=mutation,
                mutation_status=mutation_status,
                cell=cell,
                cell_status=cell_status,
            )
        )
    )
