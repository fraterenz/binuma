import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, NewType


from binuma import Dataset
from binuma.genotype import DonorGenotype, BinaryMutationMatrix
from binuma.sfs import DonorsGenotype


GenotypeMitchellRaw = NewType("GenotypeMitchellRaw", pd.DataFrame)


def load_genotype(
    path2matrix: Path,
    path2type: Path,
    keep_indels: bool,
) -> BinaryMutationMatrix:
    """Genotype is a binary matrix with rows being mutations and columns being samples"""
    # f"../mutMatrix{patient}.csv"
    # f"../mutType{patient}.csv"
    mut_matrix = pd.read_csv(path2matrix, index_col=0)
    assert isinstance(mut_matrix, pd.DataFrame)
    # map 0.5 to 0
    mut_matrix = map_unsure_reads_to_0_counts(mut_matrix)
    if not keep_indels:
        mut_type = pd.read_csv(path2type, usecols=[1], dtype="category").squeeze()
        assert isinstance(mut_type, pd.Series)
        assert mut_matrix.genotype.shape[0] == mut_type.shape[0]
        filter_mutations(mut_matrix, mut_type)
    return mut_matrix


def map_unsure_reads_to_0_counts(mut_matrix: pd.DataFrame) -> BinaryMutationMatrix:
    """Nans are encoded as 0.5 in Mitchell et al. 2022. We map the 0.5 to 0.

    We asssume that nans correspond to mutations not present in the sample
    """
    # is this slow? can we remove it
    if mut_matrix.sum().sum() > mut_matrix.shape[0] * mut_matrix.shape[1]:
        raise ValueError("Data is not between 0 and 1. Is this from Mitchell et al.?")
    # check if only 0, 0.5 and 1 are present
    mask = (
        np.isclose(mut_matrix, 0.0)
        | np.isclose(mut_matrix, 0.5)
        | np.isclose(mut_matrix, 1.0)
    )
    if not mask.all().all():
        raise ValueError(
            "Data should be either 0, 0.5 or 1. Is this from Mitchell et al. 2022?"
        )
    return BinaryMutationMatrix(pd.DataFrame(mut_matrix.map(int), dtype=int))


def get_donors() -> List[Dict[str, Any]]:
    return [
        {"name": "CB001", "age": 0, "cells": 216, "clones": 0},
        {"name": "CB002", "age": 0, "cells": 390, "clones": 0},
        {"name": "KX001", "age": 29, "cells": 407, "clones": 0},
        {"name": "KX002", "age": 38, "cells": 380, "clones": 1},
        {"name": "SX001", "age": 48, "cells": 362, "clones": 0},
        {"name": "AX001", "age": 63, "cells": 361, "clones": 1},
        {"name": "KX008", "age": 76, "cells": 367, "clones": 12},
        {"name": "KX004", "age": 77, "cells": 451, "clones": 15},
        {"name": "KX003", "age": 81, "cells": 328, "clones": 13},
    ]


class DonorsMitchell2022(DonorsGenotype):
    """Mutational matrices from the paper Mitchell 2022 et al. published in
    Nature.

    The matrices are WGS at single-cell resolution of 9 healthy individuals
    obtained by *in vitro* expansion of individual HSC/HPP cells into clonal
    colonies (clonogenic assay).
    """


def filter_mutations(
    m_matrix: BinaryMutationMatrix, m_type: pd.Series
) -> BinaryMutationMatrix:
    return BinaryMutationMatrix(
        m_matrix.genotype.iloc[m_type[m_type == "SNV"].dropna().index, :]
    )


def donor_mut_matrix(
    name: str, path2mitchell: Path, keep_indels: bool
) -> BinaryMutationMatrix:
    """Load the genotype matrix for donor `name`.

    Each column represents a cell and each row represents a mutation.
    """
    return load_genotype(
        path2mitchell / f"mutMatrix{name}.csv",
        path2mitchell / f"mutType{name}.csv",
        keep_indels,
    )


def load_mitchell2022(
    path2mitchell: Path, keep_indels: bool = True
) -> DonorsMitchell2022:
    """Load the data from the paper Mitchell et al. 2022"""
    donors = list()
    for donor in get_donors():
        try:
            geno = load_genotype(
                path2mitchell / f"mutMatrix{donor['name']}.csv",
                path2mitchell / f"mutType{donor['name']}.csv",
                keep_indels,
            )
        except FileNotFoundError:
            continue
        donors.append(
            DonorGenotype(
                dataset=Dataset.MITCHELL2022,
                genotype=geno,
                age=donor["age"],
                status="healthy",
                name=donor["name"],
            )
        )
    donors_loaded = len(donors)
    assert donors_loaded, f"Found 0 donors in `path2mitchell`: {path2mitchell}"
    print(f"Loaded {donors_loaded} donors")

    return DonorsMitchell2022(donors)
