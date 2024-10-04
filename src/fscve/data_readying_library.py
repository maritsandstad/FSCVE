"""
Functionality to sanity check, read and manipulate data for making forest emulation
"""
import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def check_data(predictor_list, data):
    """
    Code to check that data is on a suitable format

    Data should be pandas and have the right predictor variables
    found in the predictor_list, additional variables should be cut

    Parameters
    ----------
    predictor_list : list
        List of predictors needed
    data: pd.DataFrame
        Data for which to make predictions

    Returns
    -------
    pd.DataFrame
        Sanity checked DataFrame in which superflous columns have been dropped

    Raises
    ------
    TypeError
        If data are not in pandas.DataFrame
    ValueError
        If DataFrame does not contain all the columns specified in the predictor_list
    """
    if not isinstance(data, pd.DataFrame):
        LOGGER.error("Predictor data must be in the form of a Pandas DataFrame")
        raise TypeError("Predictor data must be in the form of a Pandas DataFrame")

    if not all(predictors in data.columns for predictors in predictor_list):
        LOGGER.error(  # pylint: disable=logging-fstring-interpolation
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

    Parameters
    ----------
    data_base : pd.DataFrame
        DataFrame from a forest base run for predictions
    data_forest_change: pd.DataFrame
        Data for a run with forest changes

    Returns
    -------
    list
        of cut dataFrames for both data_base and data_forest_change in which
        all points where nothing has been changed are dropped
    """
    diff = data_base.compare(data_forest_change)
    return data_base.loc[diff.index], data_forest_change.loc[diff.index]


def fill_zeros(result, data_base):
    """
    Fill in zeros for points with no change in input data, hence diff will
    be zero, and no prediction has been explicitly made

    Parameters
    ----------
    result: pd.DataFrame
        Containing predictions for the diff points
    data_base : pd.DataFrame
        DataFrame from a forest base run for predictions

    Returns
    -------
        pd.DataFrame
        Containing actual diff results for points that changed and
        zeros where there was no change in inputs
    """
    complete = result.copy()
    # TODO: Is there a more efficient way to do this?
    for index in data_base.index:
        if index not in complete.index:
            complete.loc[index] = np.zeros(len(complete.columns))
    return complete
