"""Mutational matrices from the paper Mitchell 2022 et al. published in Nature.

The matrices are WGS at single-cell resolution of 9 healthy individuals
obtained by *in vitro* expansion of individual HSC/HPP cells into clonal
colonies (clonogenic assay).
"""

from uuid import uuid4
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple

from binuma.datasets import Dataset
from binuma.donor import DataLoader, Donor, BinaryMutationMatrix, Donors
from binuma.metadata import MetadataDataset


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


def filter_mutations(
    m_matrix: BinaryMutationMatrix, m_type: pd.Series
) -> BinaryMutationMatrix:
    return BinaryMutationMatrix(
        m_matrix.genotype.iloc[m_type[m_type == "SNV"].dropna().index, :]
    )


class LoadMitchell(DataLoader):
    def load_dataset(self, path2dir: Path) -> Tuple[MetadataDataset, Donors]:
        donors, metadata = dict(), list()
        for donor in get_donors():
            try:
                idx = uuid4()

                # geno
                bin_mtx_file = f"mutMatrix{donor['name']}.csv"
                path2matrix = path2dir / bin_mtx_file
                assert path2matrix.is_file(), (
                    f"cannot find binary matrix {bin_mtx_file} for donor {donor} in {path2matrix}"
                )
                mut_matrix = pd.read_csv(path2matrix, index_col=0)
                assert isinstance(mut_matrix, pd.DataFrame)
                # map 0.5 to 0
                mut_matrix = map_unsure_reads_to_0_counts(mut_matrix)
                donors[idx] = Donor(
                    idx=idx,
                    dataset=Dataset.MITCHELL2022,
                    genotype=mut_matrix,
                    age=donor["age"],
                    status="healthy",
                    name=donor["name"],
                )

                # meta
                meta_file = f"mutType{donor['name']}.csv"
                path2type = path2dir / meta_file
                assert path2type.is_file(), (
                    f"cannot find metadata {meta_file} for donor {donor} in {path2type}"
                )
                metadata_raw = pd.read_csv(path2type, usecols=[1], dtype="category")
                assert mut_matrix.genotype.shape[0] == metadata_raw.shape[0]
                metadata_raw.rename(columns={"x": "mutation_status"}, inplace=True)
                metadata_raw["mutation"] = mut_matrix.genotype.index.to_list()
                metadata_d = mut_matrix.melt()
                metadata_d = pd.merge(
                    right=metadata_raw,
                    left=metadata_d,
                    how="left",
                    on="mutation",
                    validate="many_to_one",
                )
                metadata_d["donor_id"] = idx
                metadata_d["donor"] = donor["name"]
                metadata_d["age"] = donor["age"]
                metadata_d["dataset"] = Dataset.MITCHELL2022
                metadata_d["cell_status"] = "healthy"
                metadata.append(metadata_d)

            except AssertionError as e:
                print(e)
                print(f"--skipping donor {donor}")
                continue

        donors_loaded = len(donors)
        assert donors_loaded, f"Found 0 donors in `path2dir`: {path2dir}"
        print(f"Loaded {donors_loaded} donors")
        metadata = pd.concat(metadata)
        for c in ["donor_id", "donor", "cell_status", "mutation_status", "dataset"]:
            metadata[c] = metadata[c].astype("category")
        return MetadataDataset(metadata), Donors(donors)
