"""All the datasets from which we want to load and analyse the single-cell
genotype matrices."""

from enum import Enum, auto


class Dataset(Enum):
    """The data sources used to load and compute the site frequency spectrum (SFS)."""

    MITCHELL2022 = auto()
    """Single-cell resolution WGS data of healthy donors from [Mitchell et al. 2022](https://www.nature.com/articles/s41586-022-04786-y)."""
    KAMIZELA2025 = auto()
    """Single-cell resolution WGS data of CML patients from [Kamizela et al. 2025](https://www.nature.com/articles/s41586-025-08817-2)."""
