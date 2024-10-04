"""
Functionality to read in Forest data on lat lon format, and adding era5land predictor variables
"""

import logging
import os

import pandas as pd
import xarray as xr

from .get_data_from_era5land import get_data_for_lat_lon_vector

LOGGER = logging.getLogger(__name__)


def take_forest_dataset_and_convert_to_pd(filepath, variable_list):
    """
    Take forest dataset from netcdf file and convert to dataFrame

    Parameters
    ----------
    filepath: str
        Path to netcdf file with forest data to be read from
    variable_list: list
        List of variables to include from forest file

    Returns
    -------
        pd.DataFrame
        Containing the requested data and it's coordinates in a pd.DataFrame
        with one line per datapoint

    Raises
    ------
    FileNotFoundError
        If the filepath does not correspond to an existing file
    ValueError
        If netcdf file does not contain all the variables specified in the variable_list
    """
    if not os.path.exists(filepath):
        LOGGER.error(  # pylint: disable=logging-fstring-interpolation
            f"Cannot find file {filepath}"
        )
        raise FileNotFoundError(f"Cannot find file {filepath}")
    ds = xr.open_dataset(filepath)
    df = ds.to_dataframe()
    if isinstance(df.index, pd.MultiIndex):
        # df = df.reset_index(level = np.arange(df.index.nlevels))
        df = df.reset_index(level=list(ds.coords))
    if not all(variable in df.columns for variable in variable_list):
        LOGGER.error(  # pylint: disable=logging-fstring-interpolation
            f"Cannot find file {filepath}"
        )
        raise ValueError(
            f"All variables in {variable_list} must be in the dataset, but here are only {df.columns}"
        )
    df_to_evaluate = df[variable_list]
    df_zeros = df_to_evaluate * 0.0
    diff = df_to_evaluate.compare(df_zeros)
    df = df.loc[diff.index]
    return df


def variable_mapping(variable, forest_columns):
    """
    Map variables to upper or lower which can be found in forest_columns

    I.e. for variables that are not in forest_columns it identify mappings
    that can be performed to match them. So far this only includes
    changing to all uppercase or all lowercase lettesr

    Parameters
    ----------
    variable : str
        Variable not found in the forest_column list as is
    forest_columns: list
        List of variables that are wanted (defined predictor variables typically)

    Returns
    -------
    list
        First item is a bool to state if there is a remapping of the variable
        The second is a string that either states what the variable should be
        mapped to if a remapping is found. Otherwise it just sends back the
        original variable in this placement
    """
    if variable.upper() in forest_columns:
        return True, variable.upper()
    if variable.lower() in forest_columns:
        return True, variable.lower()
    return False, variable


def variable_mapping_era5(variable):
    """
    ERA5-Land reading specific mapping for variable names

    Parameters
    ----------
    variable : str
        Variable to remap

    Returns
    -------
    str
        Specialised ERA-5 remapping or variable, or if
        one such is not found, just the all uppercase
        version of the variable string
    """
    if variable.upper() in ["ELEVATION", "HGT"]:
        return "HGT-ERA"
    if variable.upper() in ["TAS", "TAVG"]:
        return "T2M"
    if variable.upper() in ["PR", "RSDS", "RLDS"]:
        return f"{variable.upper()}-ERA"
    return variable.upper()


def add_variables_to_forest_dataset(filepath, full_variable_list, forest_variables):
    """
    Read forest file and add other predictor variables to the resulting dataset

    Parameters
    ----------
    filepath: str
        Path to netcdf file with forest data to be read from
    full_variable_list: list
        List of variables to include (typically all predictor variables)
    forest_variables: list
        List of forest file variables

    Returns
    -------
    pd.DataFrame
        Including both forest variables and additional ERA5-Land data for
        other predictor variables
    """
    df_forest = take_forest_dataset_and_convert_to_pd(filepath, forest_variables)
    variables_missing = []
    for variable in full_variable_list:
        if variable not in df_forest.columns:
            rename, to_rename = variable_mapping(variable, df_forest.columns)
            if rename:
                df_forest.rename(columns={to_rename: variable}, inplace=True)
            else:
                variables_missing.append(variable)
    for variable in variables_missing:
        df_forest[variable] = get_data_for_lat_lon_vector(
            variable_mapping_era5(variable),
            df_forest["LAT"].values,
            df_forest["LON"].values,
        )
    return df_forest
