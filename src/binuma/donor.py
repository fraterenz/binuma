import abc
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, NewType, Set, Tuple
import uuid
import logging

from pydantic import UUID4
from binuma.datasets import Dataset
from binuma.burden import Burden, compute_burden
from binuma.genotype import BinaryMutationMatrix, filter_cells_from_matrix
from binuma.metadata import MetadataDataset
from binuma.sfs import Sfs, compute_sfs


log = logging.getLogger(__name__)


@dataclass
class Donor:
    """Donors' data are provided in form a genotype matrix. This class stores this
    matrix for a donor."""

    idx: UUID4
    name: str
    age: int
    status: str
    dataset: Dataset
    genotype: BinaryMutationMatrix
    pop_size: int = 100_000

    @cached_property
    def sfs(self):
        return compute_sfs(self.genotype)

    @cached_property
    def burden(self):
        return compute_burden(self.genotype)

    def compute_burden(self, idx: Set[str] | None = None) -> Burden:
        log.info("Computing the burden for the donor")
        if idx:
            return compute_burden(filter_cells_from_matrix(self.genotype, idx))
        return self.burden

    def compute_sfs(self, idx: Set[str] | None = None) -> Sfs:
        """The SFS for this donor using only the cells with `idx`."""
        log.info("Computing the SFS for the donor")
        if idx:
            return compute_sfs(filter_cells_from_matrix(self.genotype, idx))
        return self.sfs


Donors = NewType("Donors", Dict[uuid.uuid4, Donor])


class DataLoader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "load_dataset")
            and callable(subclass.load_dataset)
            or NotImplemented
        )

    @abc.abstractmethod
    def load_dataset(self, path2dir: Path) -> Tuple[MetadataDataset, Donors]:
        """Load in the data set"""
        raise NotImplementedError
