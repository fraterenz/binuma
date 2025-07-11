import pytest
from binuma.datasets.simulations import parameters
from pathlib import Path


def test_neg_sample_size():
    path = "-1cells"
    with pytest.raises(
        parameters.SampleSizeNotParsed,
        match=f"Cannot match regex to parse the sample size from file {path}",
    ):
        parameters.parse_sample_size(path)


def test_zero_sample_size():
    path = "0cells"
    with pytest.raises(
        parameters.SampleSizeIsZero, match=f"Found sample size of 0 from file {path}"
    ):
        parameters.parse_sample_size(path)


def test_sample_size():
    path = "10cells"
    assert parameters.parse_sample_size(path) == 10


def test_big_sample_size():
    path = "10000cells"
    assert parameters.parse_sample_size(path) == 10**4


def test_parse_filename_into_dict_missing_value():
    path = Path("/path/to/10000cells/sfs/300").stem
    with pytest.raises(ValueError):
        parameters.parse_filename(path)


def test_parse_filename_into_dict_missing_extension():
    path = Path("/path/to/10000cells/sfs/300rate_").stem
    with pytest.raises(ValueError):
        parameters.parse_filename(path)


def test_parse_filename_into_dict_trailing_sep():
    path = Path("/path/to/10000cells/sfs/300rate_.json").stem
    with pytest.raises(ValueError):
        parameters.parse_filename(path)


def test_parse_filename_into_dict_int():
    path = Path("/path/to/10000cells/sfs/300rate.json").stem
    assert parameters.parse_filename(path) == {"rate": 300}


def test_parse_filename_into_dict_float_int():
    path = Path("/path/to/10000cells/sfs/300dot1rate_20mu.json").stem
    assert parameters.parse_filename(path) == {"rate": 300.1, "mu": 20}


def test_parse_filename_into_dict_float():
    path = Path("/path/to/10000cells/sfs/300dot1rate.json").stem
    assert parameters.parse_filename(path) == {"rate": 300.1}


def test_parse_filename_into_dict_borderline():
    path = Path("/path/to/10000cells/entropy/300dot32re1_20nu.json").stem
    assert parameters.parse_filename(path) == {"re1": 300.32, "nu": 20}


@pytest.mark.parametrize(
    "path,params",
    [
        (
            Path(
                "/path/to/10000cells/sfs/0dot0years/4mu_0dot01mean_0dot01std_1tau_10000cells_260idx.json"
            ),
            parameters.Parameters(
                Path(
                    "/path/to/10000cells/sfs/0dot0years/4mu_0dot01mean_0dot01std_1tau_10000cells_260idx.json"
                ),
                10000,
                10000,
                1.0,
                4.0,
                0.01,
                0.01,
                260,
                0,
            ),
        ),
        (
            Path(
                "/path/to/100cells/sfs/0dot0years/4mu_0dot01mean_0dot01std_1tau_10000cells_20idx.json"
            ),
            parameters.Parameters(
                Path(
                    "/path/to/100cells/sfs/0dot0years/4mu_0dot01mean_0dot01std_1tau_10000cells_20idx.json"
                ),
                100,
                10_000,
                1.0,
                4.0,
                0.01,
                0.01,
                20,
                0,
            ),
        ),
    ],
)
def test_parameters_from_path(path, params):
    assert params.__dict__ == parameters.parameters_from_path(path).__dict__
