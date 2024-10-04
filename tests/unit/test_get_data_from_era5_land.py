from fscve import get_data_from_era5land


def test_get_era5_land_methods_misc():
    test_single = get_data_from_era5land.get_data_for_lat_lon("T2M", 59.9, 10.7)
    print(test_single)
    assert test_single > 0
    assert test_single < 283
    test_monthly = get_data_from_era5land.get_monthly_data_for_var_lat_lon_vector(
        "T2M", [59.9, 52.3], [10.7, 4.9]
    )
    print(test_monthly)
    assert test_monthly.shape == (12, 2)
