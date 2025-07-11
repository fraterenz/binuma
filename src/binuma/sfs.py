from abc import ABC
from dataclasses import dataclass
from typing import Sequence
from futils import snapshot
from scipy import stats

from binuma import Experiment
from binuma.genotype import BinaryMutationMatrix, DonorGenotype, DonorsGenotype


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


class DonorSfs:
    def __init__(
        self,
        name: str,
        age: int,
        is_healthy: bool,
        experiment: Experiment,
        sfs: Sfs,
        cells: int,
    ) -> None:
        self.name = name
        self.age = age
        self.is_healthy = is_healthy
        self.experiment = experiment
        self.sfs = sfs
        self.cells = cells


@dataclass
class DonorsSfs(ABC):
    """A collection of donors representing a dataset of site frequency spectra."""

    donors: Sequence[DonorSfs]


def sfs_from_genotype(geno_dnr: DonorGenotype) -> DonorSfs:
    return DonorSfs(
        geno_dnr.name,
        geno_dnr.age,
        geno_dnr.is_healthy,
        geno_dnr.experiment,
        compute_sfs(geno_dnr.genotype),
        geno_dnr.genotype.get_nb_cells(),
    )


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


def from_genotyped_donors_to_donors(
    genotyped_donors: DonorsGenotype,
) -> DonorsSfs:
    return DonorsSfs([sfs_from_genotype(d) for d in genotyped_donors.donors])
