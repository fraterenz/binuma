import logging
from typing import List, Set


import pandas as pd

log = logging.getLogger(__name__)


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
        log.debug(
            "Finding genotype entries by mapping categories (0/1) from mutation matrix"
        )
        if len(categories.symmetric_difference({0, 1})) != 0:
            raise ValueError("`genotype` should contain either 0 or 1")
        self.genotype = genotype.astype(bool)
        log.debug(
            "Converted genotype entries from floats to booleans (0/1) in mutation matrix"
        )
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
        log.debug("Computing polytomies")
        self.polytomies = self.genotype.T.duplicated(keep=False).sum()
        log.debug("Polytomies computed")
        """Polytomies or artefacts: different cells with the same genotype"""

    # def get_mutations(self) -> List[str]:
    #    print((self.df > 0).any(axis=1))
    #    return self.df.loc[:, (self.df > 0).any(axis=1)].index.to_list()

    # def get_nb_mutations(self) -> int:
    #    return len(self.get_mutations())

    def get_nb_cells(self) -> int:
        """Returns the number of all cells that have at least one mutation in this genotype matrix."""
        return len(self.cells)

    def melt(self) -> pd.DataFrame:
        """Transform a binary matrix into a long format by unpivotting using `pd.melt`.
        >>> from binuma.genotype import BinaryMutationMatrix
        >>> binary_mut = BinaryMutationMatrix(pd.DataFrame([[1, 0, 1], [0, 0, 1]], columns=["cell1", "cell2", "cell3"], index=["mut1", "mut2"]))
        >>> binary_mut.melt()
          mutation   cell
        0     mut1  cell1
        1     mut1  cell3
        2     mut2  cell3
        """
        log.info("Melting the mutation matrix")
        # first bring mutation into a column
        df2 = self.genotype.reset_index().rename(columns={"index": "mutation"})
        # melt -> long form with a 'presence' column
        long = df2.melt(id_vars="mutation", var_name="cell", value_name="presence")
        # filter and drop the presence column
        log.debug("Mutation matrix melted")
        return (
            long[long["presence"] == 1].drop(columns="presence").reset_index(drop=True)
        )


def filter_cells_from_matrix(
    binary_mut: BinaryMutationMatrix, cells2keep: Set[str]
) -> BinaryMutationMatrix:
    log.debug("Removing cells from mutation matrix")
    data = binary_mut.genotype
    data = data.drop(columns=data.loc[:, ~data.columns.isin(cells2keep)].columns)
    assert data.shape[1], "Dropped all cells"
    return BinaryMutationMatrix(data)
