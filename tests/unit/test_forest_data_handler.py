import pytest
import pandas as pd
import numpy as np
from fscve import forest_data_handler

def test_take_forest_dataset_and_convert_to_pd():
    with pytest.raises(FileNotFoundError, match="Cannot find file wrongpath"):
        forest_data_handler.take_forest_dataset_and_convert_to_pd("wrongpath", [])

    test = forest_data_handler.take_forest_dataset_and_convert_to_pd("/div/no-backup-nac/users/masan/PATHFINDER/PathFinder_WP3_Task3.5_emulator/agb_and_confier_on_era5-land_resolution_test.nc", ["agb_era5land", "conifer_era5land"])
    print(test.columns)
    print(test.shape)
    print(test.head())
    print(test.index.nlevels)
    print(type(test))
    assert False