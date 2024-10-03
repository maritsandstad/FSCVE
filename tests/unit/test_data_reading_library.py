
import pytest
import pandas as pd
import numpy as np
from fscve import data_readying_library


def test_check_data():
    predictor_list = ["LAT", "LON", "conifer_ratio"]
    with pytest.raises(TypeError, match = "Predictor data must be in the form of a Pandas DataFrame"):
        data_readying_library.check_data(predictor_list, "Hello")
    data1 = pd.DataFrame(data = [[59.9,10.7,0,0],[52.3,4.9,0,0]], columns = ["LAT", "LON", "agb", "conifer"], index = ["Oslo", "Amsterdam"])
    print(isinstance(data1, pd.DataFrame))
    with pytest.raises(ValueError):
        data_readying_library.check_data(predictor_list, data1)

    data1.rename(columns={"conifer":"conifer_ratio"}, inplace=True)
    data_cut = data_readying_library.check_data(predictor_list, data1)
    assert set(data_cut.columns) == set(predictor_list)

    data_cut2 = data_readying_library.check_data(predictor_list, data_cut)
    assert set(data_cut2.columns) == set(data_cut.columns)
    assert set(data_cut2.index) == set(data_cut.index)
    assert np.all(data_cut2.values == data_cut.values)

def test_cut_data_points_where_all_equal():
    data1 = pd.DataFrame(data = [[59.9,10.7,0,0],[52.3,4.9,0,0]], columns = ["LAT", "LON", "agb", "conifer"], index = ["Oslo", "Amsterdam"])
    data2 = pd.DataFrame(data = [[59.9,10.7,0,1],[52.3,4.9,0,0]], columns = ["LAT", "LON", "agb", "conifer"], index = ["Oslo", "Amsterdam"])
    data1_cut, data2_cut = data_readying_library.cut_data_points_where_all_equal(data1, data2)
    assert(data1_cut.shape == (1,4))
    assert(data2_cut.shape == (1,4))
    assert(data1_cut.index == ["Oslo"])
    assert(data2_cut.index == ["Oslo"])
    data1 = pd.DataFrame(data = [[59.9,10.7,0,0],[52.3,4.9,0,0], [38.0,23.7,0,0],[59.3,18.1,0,0]], columns = ["LAT", "LON", "agb", "conifer"], index = ["Oslo", "Amsterdam", "Athens", "Stockholm"])
    data2 = pd.DataFrame(data = [[59.9,10.7,0,1],[52.3,4.9,0,0], [38.0,23.7,0,0],[59.3,18.1,0.5,0]], columns = ["LAT", "LON", "agb", "conifer"], index = ["Oslo", "Amsterdam", "Athens", "Stockholm"])
    data1_cut, data2_cut = data_readying_library.cut_data_points_where_all_equal(data1, data2)
    assert(data1_cut.shape == (2,4))
    assert(data2_cut.shape == (2,4))
    assert(set(data1_cut.index) == set(["Oslo", "Stockholm"]))
    assert(set(data2_cut.index) == set(["Oslo", "Stockholm"]))


def test_prepare_datacolumn_from_netcdf():
    pass