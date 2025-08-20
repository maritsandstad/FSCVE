"""
Functionality to read in Forest data on lat lon format, and adding era5land predictor variables
"""

import logging
import os, sys

import numpy as np
import pandas as pd
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
    if variable.starts_with("skin_temperature"):
        return "TS-ERA"
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

def add_timeseriesdata_to_forest_dataset(filepath, full_variable_list, forest_variables, drop_data= None, rename_dict = None):
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
    print(forest_variables)
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
    if drop_data is not None:
        df_forest.drop(columns=drop_data, inplace=True)
    if rename_dict is not None:
        df_forest.rename(columns=rename_dict, inplace=True)

    df_forest["month"] =[timestamp.month for timestamp in df_forest["time"]]
    df_forest["year"] = [timestamp.year for timestamp in df_forest["time"]]
    df_forest_monvar = None
    for month, group in df_forest.groupby(by=["month"]):
        to_rework = group.copy()
        for fvariable_orig in forest_variables:
            if fvariable_orig in rename_dict:
                fvariable = rename_dict[fvariable_orig]
            else:
                fvariable = fvariable_orig
            to_rework.rename(columns={fvariable:f"{fvariable}_{month[0]}"}, inplace=True)
        to_rework.drop(columns=["time", "month"], inplace=True)
        if df_forest_monvar is None:
            df_forest_monvar = to_rework
        else:
            df_forest_monvar = pd.merge(df_forest_monvar, to_rework, how="outer", on=["LON", "LAT", "year"])
    #print(df_forest_monvar)
    #sys.exit(4)
    return df_forest_monvar


def make_full_grid(resolution_file="era5_land"):
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
    list
        Containing first a Dataframe with entries for each combination
        of lat and lon in the extent and resolution in each row, second
        and third the number of digits of decimal precision needed
        for this resolution in latitudes or longitudes. The DataFrame
        is already adjusted to this precision, but the same precision
        is needed when other data is to be matched into the same DataFrame
        otherwise merging and transforming to xarray won't work.
        Fourth and fifth are np.ndarrays with the latitudes and longitudes
        of the DataFrame. These will be used to initialise coordinates
        when transforming to an xarray Dataset.
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
    cross = [(A, B) for A in lats for B in lons]
    cross = np.array(cross)
    full_grid = pd.DataFrame({"lat": cross[:, 0], "lon": cross[:, 1]})
    lat_round = int(np.ceil(np.log10(len(lats) - 1) / (lats[-1] - lats[0])))
    lon_round = int(np.ceil(np.log10(len(lons) / (lons[-1] - lons[0]))))
    full_grid = full_grid.sort_values(by=["lat", "lon"])
    full_grid = full_grid.reset_index(drop=True).round(
        {"lat": lat_round, "lon": lon_round}
    )
    return full_grid, lat_round, lon_round, lats, lons


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
                df_wrong.rename(columns={col: "lat"}, inplace=True)
                break
    if "lon" not in df_wrong.columns:
        for col in df_wrong.columns:
            if col.lower() in ["lon", "long", "longitude"]:
                df_wrong.rename(columns={col: "lon"}, inplace=True)
                break
    return df_wrong

def merge_dfs_with_float_lat_lons(lat_round, lon_round, df_left, df_right, additional_on_merge=None, merge_how = 'left'):
    df_left = rename_lat_lon(df_left)
    df_right = rename_lat_lon(df_right)
    lat_factor = 10**(lat_round)
    lon_factor = 10**(lon_round)
    df_right["LAT_merge"] = np.round(df_right["lat"]*lat_factor).astype(int)
    df_left["LAT_merge"] = np.round(df_left["lat"]*lat_factor).astype(int)
    df_right["LON_merge"] = np.round(df_right["lon"]*lon_factor).astype(int)
    df_left["LON_merge"] = np.round(df_left["lon"]*lon_factor).astype(int)
    df_left.drop(columns=["lat", "lon"], inplace= True)
    df_right.drop(columns=["lat", "lon"], inplace= True)
    print(f"Shape of left: {df_left.shape} and shape of right: {df_right.shape}")
    if additional_on_merge is None:
        print("Just on lat and lon")
        duplicates = df_right[df_right.duplicated(subset=["LAT_merge", "LON_merge"], keep=False)]
        print(duplicates.shape)
        print(df_right[(df_right["LAT_merge"] == 350) & (df_right["LON_merge"] == 0)]  )
        merged = pd.merge(
            df_left, df_right, how=merge_how, on=["LAT_merge", "LON_merge"]
        )
    else:
        onlist = ["LAT_merge", "LON_merge"] + additional_on_merge
        merged = pd.merge(
            df_left, df_right, how=merge_how, on=onlist
        )
    print(f"Finally merged befor dropping: {merged.shape}")
    merged["lat"] = merged["LAT_merge"]/lat_factor
    merged["lon"] = merged["LON_merge"]/lon_factor
    merged.drop(columns=["LAT_merge", "LON_merge"], inplace=True)
    return merged

def make_sparse_forest_df_xarray(df_sparse, resolution_file="era5_land"):
    """
    Make a sparse forest DataFrame into a fully filled in xarray

    Parameters
    ----------
    df_sparse : pd.DataFrame
        Dataset with longitude, latitude and data output columns, can be
        predicted climate variables, forest coverage or something else
    resolution_file : str
        Directions for full grid resolution. Default is era5_land and if
        this is sent, an era5_land 0.1 degree global coverage resolution
        will be used. Otherwise the path for a netcdf file with the
        expected resolution is expected

    Returns
    -------
    xr.Dataset
        With the data from df_sparse on the lat-lon gridpoints where it has
        data, and zeros otherwise
    """
    full_grid, lat_round, lon_round, outlats, outlons = make_full_grid(resolution_file)
    df_sparse_w_index = rename_lat_lon(df_sparse).round(
        #{"lat": lat_round, "lon": lon_round}
        {"lat": lat_round, "lon": lon_round}
    )
    data_onto_full_grid = pd.merge(
        full_grid, df_sparse_w_index, how="inner", on=["lat", "lon"]
    )
    data_onto_full_grid = merge_dfs_with_float_lat_lons(lat_round, lon_round, full_grid, df_sparse_w_index)

    #sys.exit(4)
    variables = {}
    coords = {"lat": outlats, "lon": outlons}
    for col in data_onto_full_grid.columns:
        if col in ["lat", "lon"]:
            continue
        target_variable_2d = data_onto_full_grid[col].values.reshape(
            (len(outlats), len(outlons))
        )
        target_variable_xr = xr.DataArray(
            target_variable_2d, coords=[("lat", outlats), ("lon", outlons)]
        )
        target_variable_xr = target_variable_xr.rename(col)
        variables[col] = target_variable_xr

    data_ds = xr.Dataset(data_vars=variables, coords=coords)
    return data_ds.fillna(0)

def add_static_data_to_tsdata(df_ts, df_static, ts_var = "year"):
    df_ts = rename_lat_lon(df_ts)
    df_static = rename_lat_lon(df_static)
    #sys.exit(4)
    #print(df_static)

    list_concat = []

    for tgroup, group in df_ts.groupby(by=[ts_var]):
        print(f"{tgroup} with shape {group.shape}")
        df_merge = merge_dfs_with_float_lat_lons(1, 1, group.copy(), df_static.copy())
        print(f"Shape after the merge {df_merge.shape}")
        list_concat.append(df_merge)
    df_full = pd.concat(list_concat)
    print(df_full.shape)
    #sys.exit(4)
    return df_full



def combine_forest_datasets(forest_datasets):

    if isinstance(forest_datasets, dict):
        forest_dataset_list = []
        for data_path, arguments in forest_datasets.items():
            ds = add_variables_to_forest_dataset(
                data_path,
                full_variable_list=arguments["full_variable_list"],
                forest_variables=arguments["forest_variables"]
            )
            print(ds)
            ds.drop(columns=arguments["drop_data"], inplace=True)
            forest_dataset_list.append(ds.copy())
    else:
        forest_dataset_list = forest_datasets
    
    combined_dataset = None
    for ds in forest_dataset_list:
        print(ds)
        if combined_dataset is None:
            combined_dataset = ds
        else:
            print("In else")
            combined_dataset = merge_dfs_with_float_lat_lons(2, 2, combined_dataset, ds, merge_how='outer')

    return combined_dataset