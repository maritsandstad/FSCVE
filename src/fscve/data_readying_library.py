import logging

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)


def check_data(predictor_list, data):
    """
    Code to check that data is on a suitable format

    Data should be pandas and have the right predictor variables
    found in the predictor_list, additional variables should be cut


    """
    if not isinstance(data, pd.DataFrame):
        LOGGER.error("Predictor data must be in the form of a Pandas DataFrame")
        raise TypeError("Predictor data must be in the form of a Pandas DataFrame")

    if not all(predictors in data.columns for predictors in predictor_list):
        LOGGER.error(
            f"Incomplete predictor data. {data.columns} do not match {predictor_list}"
        )
        raise ValueError(
            f"Incomplete predictor data. {data.columns} do not match {predictor_list}"
        )
    if len(data.columns) == len(predictor_list):
        return data
    for col in data.columns:
        if col not in predictor_list:
            data.drop(columns=[col], inplace=True)
    return data


def cut_data_points_where_all_equal(data_base, data_forest_change):
    """
    Code to cut datapoints that don't provide information

    I.e. all datapoints which have same exact values for base and forest


    """
    diff = data_base.compare(data_forest_change)
    print(diff.index)
    print(data_base.loc[diff.index])
    print(data_forest_change.loc[diff.index])
    return data_base.loc[diff.index], data_forest_change.loc[diff.index]


def fill_zeros(result, data_base):
    complete = result.copy()
    for index in data_base.index:
        if index not in complete.index:
            complete.loc[index] = np.zeros(len(complete.columns))
    return complete


def prepare_datacolumn_from_netcdf():
    pass
