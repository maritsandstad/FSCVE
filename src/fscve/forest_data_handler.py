import os
import logging

import xarray as xr
import pandas as pd
import numpy as np

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

    # TODO, drop rows with zeros in variable list
    return df
