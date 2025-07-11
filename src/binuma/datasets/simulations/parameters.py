"""Parse the filename of the output of a SFS simulated in rust.
Rust saves the output of the simulations by encoding the parameters used to
simulate into the filename.
Here we want to extract the parameters from the filenames.
Assume the file `myfile.myext` is saved as:
    `/path/to/dir/{number}cells/sfs/{age}years/myfile.json`
"""

import re
import pandas as pd
from pathlib import Path

from typing import Any, Dict, List, Set


class Parameters:
    def __init__(
        self,
        path: Path,
        sample: int,
        cells: int,
        tau: float,
        mu: float,
        mean: float,
        std: float,
        idx: int,
        age: int,
    ):
        self.sample = sample
        self.path = path
        self.cells = cells
        self.tau = tau
        self.mu = mu
        self.s = mean
        self.std = std
        self.idx = idx
        self.age = age

    def into_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def stringify(self, some_params: Set[str]) -> str:
        return ", ".join(
            [f"{k}={v}" for k, v in self.into_dict().items() if k in some_params]
        )


class SampleSizeIsZero(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


class AgeNotParsed(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


class SampleSizeNotParsed(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


def parse_age(part_with_age: str) -> int:
    """
    >>> from sfs_data.datasets.simulations import parameters
    >>> [parameters.parse_age(age) for age in ['0dot0years', '0dot5years', '10dot0years', '10dot5years']]
    [int(0), int(0), int(10), int(10)]
    """
    match_sample = re.compile(r"^(\d+)dot(\d+)(years)$", re.IGNORECASE)
    matched = match_sample.search(part_with_age)
    if matched:
        return int(matched.group(1))
    raise AgeNotParsed(
        f"Cannot match regex to parse the age from {part_with_age}",
        error_code=1,
    )


def parse_sample_size(part_with_sample_size: str) -> int:
    """
    >>> from sfs_data.datasets.simulations import parameters
    >>> [parameters.parse_sample_size(age) for age in ['10cells', '100cells']]
    [int(10), int(100)]
    """
    match_sample = re.compile(r"^(\d+)(cells)$", re.IGNORECASE)
    matched = match_sample.search(part_with_sample_size)

    if matched:
        sample_size = int(matched.group(1))
        # neg sample size wont match
        if sample_size == 0:
            raise SampleSizeIsZero(
                f"Found sample size of 0 from file {part_with_sample_size}",
                error_code=1,
            )
        return sample_size
    raise SampleSizeNotParsed(
        f"Cannot match regex to parse the sample size from file {part_with_sample_size}",
        error_code=1,
    )


def parameters_from_path(path: Path) -> Parameters:
    """The main method to use: take a path as input and returns a dict.
    The path must follow the convention:
    `/path/to/dir/{number}cells/sfs/{age}dot{age2}years/myfile.json`
    """
    # assume the first (\d+)(cells) is the sample size
    params_file = parse_filename(path.stem)
    parts = path.parts
    params_file["sample"] = parse_sample_size(parts[-4])
    params_file["age"] = parse_age(parts[-2])
    params_file["path"] = path
    return Parameters(**params_file)


def parse_filename(file_stemmed: str) -> Dict[str, Any]:
    """
    >>> from sfs_data.datasets.simulations import parameters
    >>> filename = "4mu_0dot01mean_0dot01std_1tau_10000cells_260idx.json"
    >>> parameters.parse_filename(filename)
    {'mu': 4.0, 's': 0.01, 'std': 0.01, 'tau': 1.0, 'cells': 10_000', 'idx': 260}
    """
    match_nb = re.compile(r"(\d+\.?\d*)([a-z]+\d*)", re.IGNORECASE)
    filename_str = file_stemmed.replace("dot", ".").split("_")
    my_dict = dict()
    for ele in filename_str:
        matched = match_nb.search(ele)
        if matched:
            if matched.group(2) in ["idx", "cells"]:
                my_dict[matched.group(2)] = int(matched.group(1))
            else:
                my_dict[matched.group(2)] = float(matched.group(1))
        else:
            raise ValueError(
                f"Syntax <NUMBER><VALUE>_ not respected in filename {file_stemmed}"
            )
    return my_dict


def params_into_dataframe(params: List[Parameters]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([param for param in params])
    df.idx = df.idx.astype(int)
    df.cells = df.cells.astype(int)
    df["samples"] = df["samples"].astype(int)
    return df
