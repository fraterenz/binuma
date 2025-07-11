import pandas as pd
from typing import Dict, NewType, List, Tuple
from futils import snapshot
from pydantic import UUID4
from scipy import stats

from binuma.genotype import BinaryMutationMatrix, DonorGenotype
from binuma.metadata import MetadataDataset


class Sfs:
    """The frequency of mutations in a genotype matrix grouped into occurences.

    We first compute the number of mutations in each cells j: S_j.
    Then we group S_j, by counting how many S_j mutations are present in j cells.
    """

    def __init__(self, sfs: snapshot.Histogram) -> None:
        self.sfs = sfs
        """The SFS is implemented as a [`futils.snapshot.Histogram`](https://github.com/fraterenz/futils/tree/master),
        that is a dict of integers. Keys are the j cells (the x-axis of the
        SFS) and values are the counts of mutations (the y-axis of the SFS)."""

    def __repr__(self) -> str:
        return self.sfs.__repr__()

    def compute_entropy(self) -> float:
        """Compute the base 2 entropy of the SFS using [`scipy.stats.entropy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)"""
        return float(stats.entropy(snapshot.array_from_hist(self.sfs), base=2))


def sfs_from_genotype(geno_dnr: DonorGenotype) -> Sfs:
    return compute_sfs(geno_dnr.genotype)


def compute_sfs(genotype: BinaryMutationMatrix) -> Sfs:
    sfs_donor = genotype.genotype.sum(axis=1).value_counts()
    # drop mutations non occurring in any cell
    sfs_donor.drop(index=sfs_donor[sfs_donor.index == 0].index, inplace=True)
    x_sfs = sfs_donor.index.to_numpy(dtype=int)
    return Sfs(
        snapshot.histogram_from_dict(
            {x: y for x, y in zip(x_sfs, sfs_donor.to_numpy())},
        ),
    )


DonorsSfs = NewType("DonorsSfs", Dict[UUID4, Sfs])


class DonorsGenotype:
    def __init__(self, donors: List[DonorGenotype]) -> None:
        self.donors = donors

    def into_sfs(self) -> Tuple[DonorsSfs, MetadataDataset]:
        donors_sfs = DonorsSfs(
            {d.metadata.idx: sfs_from_genotype(d) for d in self.donors}
        )
        metadata = MetadataDataset(
            pd.DataFrame.from_records([dict(d.metadata) for d in self.donors])
        )
        return donors_sfs, metadata
