# Binuma
Summary statistics from single-cell BINary mUation MAtrices  in Python.

A binary muation matrix $A$ (also known as a genotype matrix) is a NxD array where the rows are unique mutations and columns unique cells found in the sequencing sample.
The entry $a_{ij}$ is 1 if the mutation $i$ is present in the cell $j$, otherwise $a_{ij}$ is 0.

## Statistics
The following statistics are implemented for now:
1. site frequency spectrum (SFS): the frequency of the number mutations found in each cell
2. single-cell mutational burden: the number of mutations found in each cell

## Datasets
This package provides allows to compute the statistics for the following datasets:
- single-cell resolution WGS data of healthy donors from [Mitchell et al. 2022](https://www.nature.com/articles/s41586-022-04786-y)
- single-cell resolution WGS data of CML patients from [Kamizela et al. 2025](https://www.nature.com/articles/s41586-025-08817-2)
