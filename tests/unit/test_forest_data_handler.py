import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fscve import forest_data_handler


def test_take_forest_dataset_and_convert_to_pd(test_data_dir):
    with pytest.raises(FileNotFoundError, match="Cannot find file wrongpath"):
        forest_data_handler.take_forest_dataset_and_convert_to_pd("wrongpath", [])

    test = forest_data_handler.take_forest_dataset_and_convert_to_pd(
        os.path.join(test_data_dir, "type_of_high_vegetation_PATHFINDER.nc"),
        ["tvh"],
    )
    print(test.columns)
    print(test.shape)
    print(test.head())
    print(test.index.nlevels)
    print(type(test))
    assert set(test.columns) == set(["latitude", "longitude","time", "tvh"])
    assert test.index.nlevels == 1
    assert len(test.index) < 1e5


def test_variable_mappings():
    assert set(forest_data_handler.variable_mapping("lat", ["LAT", "LON"])) == set(
        (True, "LAT")
    )
    assert set(forest_data_handler.variable_mapping("LON", ["lat", "lon"])) == set(
        (True, "lon")
    )
    assert set(forest_data_handler.variable_mapping("Hello", ["lat", "lon"])) == set(
        (False, "Hello")
    )
    assert forest_data_handler.variable_mapping_era5("elevation") == "HGT-ERA"
    assert forest_data_handler.variable_mapping_era5("TAVG") == "T2M"
    assert forest_data_handler.variable_mapping_era5("Hello") == "HELLO"
    assert forest_data_handler.variable_mapping_era5("Rsds") == "RSDS-ERA"
    assert forest_data_handler.variable_mapping_era5("skin_temperature") == "TS-ERA"


def test_add_variables_to_forest_dataset(test_data_dir):
    test = forest_data_handler.add_variables_to_forest_dataset(
        os.path.join(test_data_dir, "type_of_high_vegetation_PATHFINDER.nc"),
        ["LAT", "LON", "ELEVATION", "TAVG", "tvh"],
        ["tvh"],
    )
    assert set(test.columns) == set(
        ["LAT", "LON", "ELEVATION", "TAVG", "time", "tvh"]
    )


def test_make_sparse_forest_df_xarray():
    df_test = pd.DataFrame(
        data=[
            [59.9, 10.7, 1, 0.5],
            [52.3, 4.9, 1, 0.5],
            [38.0, 23.7, 0, 0],
            [59.3, 18.1, 0, 0],
        ],
        columns=["LAT", "LON", "agb", "conifer"],
        index=["Oslo", "Amsterdam", "Athens", "Stockholm"],
    )
    ds_in_xr = forest_data_handler.make_sparse_forest_df_xarray(df_test)
    print(ds_in_xr)
    print(ds_in_xr["conifer"].shape)
    assert isinstance(ds_in_xr, xr.Dataset)
    assert set(ds_in_xr.coords.keys()) == set(["lat", "lon"])
    assert set(ds_in_xr.variables.keys()) == set(["lat", "lon", "agb", "conifer"])
    assert np.isnan(ds_in_xr["conifer"].sel(lat=60.0, lon=10.0))
    print("Hei")
    print(ds_in_xr["conifer"])
    assert ds_in_xr["conifer"].sel(lat=slice(59.89, 59.91), lon=10.7).values[0] == 0.5
    assert (
        ds_in_xr["conifer"]
        .sel(lat=slice(52.29, 52.31), lon=slice(4.89, 4.91))
        .values[0]
        == 0.5
    )

    assert ds_in_xr["agb"].shape == (1801, 3600)


def test_make_full_grid():
    test, round_lat, round_lon, outlat, outlon = forest_data_handler.make_full_grid()
    assert test.shape == (1801 * 3600, 2)
    assert len(outlat) == 1801
    assert len(outlon) == 3600
    assert round_lon == 3
    assert round_lat == 3
    assert set(test.columns) == set(["lat", "lon"])
    # test = forest_data_handler.make_full_grid()


def test_rename_lat_lon():
    df_test = pd.DataFrame(
        data=[
            [59.9, 10.7, 1, 0.5],
            [52.3, 4.9, 1, 0.5],
            [38.0, 23.7, 0, 0],
            [59.3, 18.1, 0, 0],
        ],
        columns=["LAT", "LON", "agb", "conifer"],
        index=["Oslo", "Amsterdam", "Athens", "Stockholm"],
    )
    df_test2 = forest_data_handler.rename_lat_lon(df_test)
    assert set(df_test2.columns) == set(["lat", "lon", "agb", "conifer"])
    df_test3 = forest_data_handler.rename_lat_lon(df_test)

    assert set(df_test3.columns) == set(["lat", "lon", "agb", "conifer"])
