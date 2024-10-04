import os
import logging

import xarray as xr
import pandas as pd
import numpy as np

from .get_data_from_era5land import get_data_for_lat_lon_vector

LOGGER = logging.getLogger(__name__)

def take_forest_dataset_and_convert_to_pd(filepath, variable_list):
    if not os.path.exists(filepath):
        LOGGER.error(f"Cannot find file {filepath}")
        raise FileNotFoundError(
                f"Cannot find file {filepath}"    
            )
    ds = xr.open_dataset(filepath)
    print(list(ds.coords))
    df = ds.to_dataframe()
    if isinstance(df.index, pd.MultiIndex):
        print("Hello")
        #df = df.reset_index(level = np.arange(df.index.nlevels))
        df = df.reset_index(level = list(ds.coords))
    if not all( variable in df.columns for variable in variable_list):
        LOGGER.error(f"Cannot find file {filepath}")
        raise ValueError(
                f"All variables in {variable_list} must be in the dataset, but here are only {df.columns}"    
            )       
    df_to_evaluate = df[variable_list]
    df_zeros = df_to_evaluate*0.0
    diff = df_to_evaluate.compare(df_zeros)
    df = df.loc[diff.index]
    return df

def variable_mapping(variable, forest_columns):
    if variable.upper() in forest_columns:
        return True, variable.upper()
    if variable.lower() in forest_columns:
        return True, variable.lower()
    return False, variable
    
def variable_mapping_era5(variable):
    if variable.upper() in ["ELEVATION", "HGT"]:
        return "HGT-ERA"
    if variable.upper() in ["TAS", "TAVG"]:
        return "T2M"
    if variable.upper() in ["PR", "RSDS", "RLDS"]:
        return f"{variable.upper()}-ERA"
    return variable.upper()

def add_variables_to_forest_dataset(filepath, full_variable_list, forest_variables):
    df_forest = take_forest_dataset_and_convert_to_pd(filepath, forest_variables)
    variables_missing = []
    for variable in full_variable_list:
        if variable not in df_forest.columns:
            rename, to_rename = variable_mapping(variable, df_forest.columns)
            if rename:
                df_forest.rename(columns={to_rename:variable}, inplace=True)
            else:
                variables_missing.append(variable)
    for variable in variables_missing:
        df_forest[variable] = get_data_for_lat_lon_vector(variable_mapping_era5(variable), df_forest["LAT"].values, df_forest["LON"].values)

    return df_forest