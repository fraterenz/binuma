from hypothesis import given, strategies as st
from uuid import uuid4
from binuma import Dataset
from binuma.metadata import Metadata


def valid_age(n: int) -> bool:
    return (n > 0) & (n < 120)


def valid_size(n: int) -> bool:
    return n > 0


@given(
    st.text(),
    st.integers().filter(valid_size),
    st.integers().filter(valid_size),
    st.integers().filter(valid_age),
    st.text(),
    st.sampled_from(Dataset),
)
def test_metadata(name, sample, pop_size, age, status, dataset):
    Metadata(
        idx=uuid4(),
        name=name,
        sample=sample,
        pop_size=pop_size,
        status=status,
        age=age,
        dataset=dataset,
    )
