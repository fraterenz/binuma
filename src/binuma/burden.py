from futils import snapshot
import logging

from binuma.genotype import BinaryMutationMatrix

log = logging.getLogger(__name__)


class Burden:
    """The number of mutations within one cell, aka the single-cell mutational
    burden."""

    def __init__(self, burden: snapshot.Histogram) -> None:
        self.burden = burden

    def __repr__(self) -> str:
        return self.burden.__repr__()


def compute_burden(genotype: BinaryMutationMatrix) -> Burden:
    log.info("Computing the burden from the mutation matrix")
    burden_donor = genotype.genotype.sum(axis=0).value_counts()
    x_burden, burden_donor = (
        burden_donor.index.to_numpy(),
        burden_donor.to_numpy(),
    )
    return Burden(
        snapshot.histogram_from_dict({x: y for x, y in zip(x_burden, burden_donor)})
    )
