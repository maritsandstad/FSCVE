"""
Functionality to read in Forest data on lat lon format, and adding era5land predictor variables
"""

import logging
import os

import pandas as pd
import numpy as np
import xarray as xr

from .get_data_from_era5land import get_data_for_lat_lon_vector

LOGGER = logging.getLogger(__name__)


def take_forest_dataset_and_convert_to_pd(filepath, variable_list):
    """
    Take forest dataset from netcdf file and convert to dataFrame
    Datapoints with no forest data will be taken out of dataFrame

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
    if "lat" not in df_forest.columns or "lon" not in df_forest.columns:
        df_forest = rename_lat_lon(df_forest)
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

def make_full_grid(resolution_file= "era5_land"):
    """
    Make a full latxlon grid of a resolution, either era5_land or from netcdf file

    Parameters
    ----------
    resolution_file : str
        Path to file with netcdf data on wanted resolution/extent
        A regular lat-lon grid will be assumed
        If not sent a global era5_land resolution will be used

    Returns
    -------
    pd.DataFrame
        Dataframe with entries for each combination of lat and lon
        in the extent and resolution in each row
    """
    if resolution_file == "era5_land":
        max_lat = 90
        min_lat = -90
        num_lat = 1801
        max_lon = 359.9
        min_lon = 0
        num_lon = 3600
    else:
        ds_grid = xr.open_dataset(resolution_file)
        max_lat = ds_grid["lat"].max()
        min_lat = ds_grid["lat"].min()
        num_lat = len(ds_grid["lat"])
        max_lon = ds_grid["lon"].max()
        min_lon = ds_grid["lon"].min()
        num_lon = len(ds_grid["lon"])

    lats = np.linspace(min_lat, max_lat, num=num_lat)
    lons = np.linspace(min_lon, max_lon, num=num_lon)
    cross = [(A,B) for A in lats for B in lons]
    cross = np.array(cross)
    full_grid = pd.DataFrame({
        'lat': cross[:,0], 
        'lon': cross[:,1]
        })
    full_grid = full_grid.sort_values(by=['lat','lon'])
    full_grid = full_grid.reset_index(drop=True)
    return full_grid, lats, lons

def rename_lat_lon(df_wrong):
    """
    Rename lat and lon with different names in dataframe

    Parameters
    ----------
    df_wrong : pd.DataFrame
        Dataframe in which lat and lon is to be renamed
        First and best reasonable match will be used
    
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed lat and lon if suitable alternatives exist
    """
    if "lat" not in df_wrong.columns:
        for col in df_wrong.columns:
            if col.lower() in ["lat", "lati", "latitude"]:
                df_wrong.rename(columns={col:"lat"}, inplace=True)
                break
    if "lon" not in df_wrong.columns:
        for col in df_wrong.columns:
            if col.lower() in ["lon", "long", "longitude"]:
                df_wrong.rename(columns={col:"lon"}, inplace=True)
                break 
    return df_wrong

def get_indices_from_lat_lon(df_sparse, outlats, outlons):
    print(len(outlats))
    print(len(outlons))
    lat_spacing = int(len(outlats)-1)/(outlats[-1] - outlats[0])
    lon_spacing = int(len(outlons)/(outlons[-1] - outlons[0]))
    df_sparse["calc_index"] = ((df_sparse["lat"] - outlats[0])*lat_spacing)*len(outlons) + ((df_sparse["lon"] - outlons[0])*lon_spacing).astype(int)
    df_sparse.set_index("calc_index", inplace = True)
    #df_sparse.drop(columns = ["lat", "lon"], inplace = True)
    return df_sparse.sort_values(by="calc_index")

def make_sparse_forest_df_xarray(df_sparse, resolution_file= "era5_land"):

    full_grid, outlats, outlons = make_full_grid(resolution_file)
    df_sparse_w_index = get_indices_from_lat_lon(rename_lat_lon(df_sparse), outlats, outlons)
    print(df_sparse_w_index.index.max())
    print(df_sparse_w_index.index.min())
    print(df_sparse_w_index.index.has_duplicates)
    ids = df_sparse_w_index.index
    print(df_sparse_w_index[ids.isin(ids[ids.duplicated()])])#df_sparse_w_index.index.duplicated])
    data_onto_full_grid = pd.merge(full_grid, df_sparse_w_index, how= "left", left_index=True, right_index=True)#how='outer', on = ["lat", "lon"])
    print(data_onto_full_grid.index.max())
    print(data_onto_full_grid.index.min())
    variables = {}
    coords = {"lat": outlats, "lon": outlons}
    for col in data_onto_full_grid.columns:
        if col in ["lat", "lon"]:
            continue
        target_variable_2D = data_onto_full_grid[col].values.reshape((len(outlats),len(outlons)))
        target_variable_xr = xr.DataArray(target_variable_2D, coords=[('lat', outlats),('lon', outlons)])
        target_variable_xr = target_variable_xr.rename(col)
        variables[col] = target_variable_xr

    data_ds = xr.Dataset(data_vars=variables, coords=coords)
    return data_ds.fillna(0)
