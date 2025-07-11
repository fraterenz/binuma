from typing import List, Set
import uuid
import pandas as pd
from binuma import Dataset
from binuma.metadata import Metadata


class EntryIsNan(ValueError):
    """Found at least one nan entry in the genotype: entries must be either 0 or 1"""


class NonUniqueMutations(ValueError):
    """Non unique mutations found in the genotype"""


class NonUniqueCells(ValueError):
    """Non unique cells found in the genotype"""


class BinaryMutationMatrix:
    """A binary matrix containing only 0s and 1s.
    Rows are unique mutations and columns are unique samples (i.e. cells/colonies)."""

    def __init__(self, genotype: pd.DataFrame) -> None:
        """The `genotype` is the binary matrix with unique mutations/samples as
        index/columns."""

        def get_unique_quantities(quantities: List[str]) -> Set[str]:
            unique_q = set(quantities)
            assert len(unique_q) == len(quantities)
            return unique_q

        assert isinstance(genotype, pd.DataFrame), "Wrong input `genotype`"
        assert genotype.shape[0], "Empty `genotype`"
        if genotype.isna().any().any():
            raise EntryIsNan()
        genotype = genotype.astype("uint8")
        categories = set(genotype.to_numpy().ravel())
        if len(categories.symmetric_difference({0, 1})) != 0:
            raise ValueError("`genotype` should contain either 0 or 1")
        self.genotype = genotype.astype(bool)
        """A `pd.DataFrame` with boolean values."""

        try:
            self.cells = get_unique_quantities(
                self.genotype.loc[:, (self.genotype > 0).any(axis=0)].columns.to_list()
            )
            """Unique cells in the genotype with at least one mutation in the genotype matrix"""
        except AssertionError:
            raise NonUniqueCells
        try:
            self.mutations = get_unique_quantities(self.genotype.index.to_list())
            """Unique mutations in the genotype"""
        except AssertionError:
            raise NonUniqueMutations
        self.polytomies = self.genotype.T.duplicated(keep=False).sum()
        """Polytomies or artefacts: different cells with the same genotype"""

    # def get_mutations(self) -> List[str]:
    #    print((self.df > 0).any(axis=1))
    #    return self.df.loc[:, (self.df > 0).any(axis=1)].index.to_list()

    # def get_nb_mutations(self) -> int:
    #    return len(self.get_mutations())

    def get_nb_cells(self) -> int:
        """Returns the number of all cells that have at least one mutation in this genotype matrix."""
        return len(self.cells)


def filter_cells_from_matrix(
    binary_mut: BinaryMutationMatrix, cells2keep: Set[str]
) -> BinaryMutationMatrix:
    data = binary_mut.genotype
    data = data.drop(columns=data.loc[:, ~data.columns.isin(cells2keep)].columns)
    assert data.shape[1], "Dropped all cells"
    return BinaryMutationMatrix(data)


class DonorGenotype:
    """Donors' data are provided in form a genotype matrix. This class stores this
    matrix for a donor."""

    def __init__(
        self,
        name: str,
        age: int,
        status: str,
        dataset: Dataset,
        genotype: BinaryMutationMatrix,
        pop_size: int = 100_000,
    ) -> None:
        self.metadata = Metadata(
            name=name,
            age=age,
            status=status,
            pop_size=pop_size,
            dataset=dataset,
            idx=uuid.uuid4(),
            sample=genotype.get_nb_cells(),
        )
        self.genotype = genotype
