"""A package to compute site frequency spectra (SFS) from genotype matrices from
samples collected from either healthy donors or cancer patients.

The SFS requires mutational data obtained at single-cell resolution.
This information is stored in genotype matrices.
Filled with 0s or 1s, a genotype matrix has rows/columns indicating mutations/cells
respectively."""

from enum import Enum, auto


class Dataset(Enum):
    """The data sources used to load and compute the site frequency spectrum (SFS)."""

    MITCHELL2022 = auto()
    """Single-cell resolution WGS data of healthy donors from Mitchell et al. 2022"""
    KAMIZELA2025 = auto()
    """Single-cell resolution WGS data of CML patients from Kamizela et al. 2025"""
    HSCSIMULATIONS = auto()
    """Simulations generated with the binary [`hsc`](https://github.com/fraterenz/hsc)"""
