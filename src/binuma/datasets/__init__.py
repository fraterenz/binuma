from typing import Tuple
from pydantic import DirectoryPath, validate_call

from binuma import Dataset
from binuma.datasets import kamizela2025, mitchell2022
from binuma.datasets.simulations import hsc
from binuma.metadata import MetadataDataset
from binuma.sfs import DonorsSfs


@validate_call
def load_sfs(
    path2dir: DirectoryPath, dataset: Dataset
) -> Tuple[DonorsSfs, MetadataDataset]:
    """Load the site frequency spectra (SFS) from the `path2dir` directory,
    which contains all the SFS of an `Experiment`."""
    if dataset == Dataset.HSCSIMULATIONS:
        return hsc.load_simulations(path2dir)
    if dataset == Dataset.MITCHELL2022:
        return mitchell2022.load_mitchell2022(path2dir).compute_mitchell2022_sfs()
    if dataset == Dataset.KAMIZELA2025:
        return kamizela2025.load_kamizela2025(path2dir)
    raise ValueError
