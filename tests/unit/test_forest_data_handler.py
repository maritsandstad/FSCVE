import pytest

from fscve import forest_data_handler


def test_take_forest_dataset_and_convert_to_pd():
    with pytest.raises(FileNotFoundError, match="Cannot find file wrongpath"):
        forest_data_handler.take_forest_dataset_and_convert_to_pd("wrongpath", [])

    test = forest_data_handler.take_forest_dataset_and_convert_to_pd(
        "/div/no-backup-nac/users/masan/PATHFINDER/PathFinder_WP3_Task3.5_emulator/agb_and_confier_on_era5-land_resolution_test.nc",
        ["agb_era5land", "conifer_era5land"],
    )
    print(test.columns)
    print(test.shape)
    print(test.head())
    print(test.index.nlevels)
    print(type(test))
    assert set(test.columns) == set(["lat", "lon", "agb_era5land", "conifer_era5land"])
    assert test.index.nlevels == 1
    assert len(test.index) < 1e5


def test_variable_mappings():
    assert set(forest_data_handler.variable_mapping("lat", ["LAT", "LON"])) == set(
        (True, "LAT")
    )
    assert set(forest_data_handler.variable_mapping("LON", ["lat", "lon"])) == set(
        (True, "lon")
    )
    assert forest_data_handler.variable_mapping_era5("elevation") == "HGT-ERA"
    assert forest_data_handler.variable_mapping_era5("TAVG") == "T2M"
    assert forest_data_handler.variable_mapping_era5("Hello") == "HELLO"
    assert forest_data_handler.variable_mapping_era5("Rsds") == "RSDS-ERA"


def test_add_variables_to_forest_dataset():
    test = forest_data_handler.add_variables_to_forest_dataset(
        "/div/no-backup-nac/users/masan/PATHFINDER/PathFinder_WP3_Task3.5_emulator/agb_and_confier_on_era5-land_resolution_test.nc",
        ["LAT", "LON", "ELEVATION", "TAVG", "agb_era5land", "conifer_era5land"],
        ["agb_era5land", "conifer_era5land"],
    )
    assert set(test.columns) == set(
        ["LAT", "LON", "ELEVATION", "TAVG", "agb_era5land", "conifer_era5land"]
    )
