"""
Functionality to read from era5land data
"""
from math import ceil, floor

import numpy as np
import xarray as xr

# TODO: Generalize this (both file and file-placement, but also to non ERA5 data)?
# This path needs to be changed to specific folder
# Also if ERA5-land datafiles are differently named that too needs to change
FOLDER = "/div/no-backup-nac/reanalysis/ERA5-land/"


varstring = {
    "T2M": "tas",
    "T2DEW": "t2dew",
    "RLDS-ERA": "rlds",
    "HGT-ERA": "hgt",
    "PR-ERA": "pr",
    "RSDS-ERA": "rsds",
}


def round_up(n, decimals=0):
    """
    Round up number to given number of decimals

    Parameters
    ----------
    n : float
        Number to round up
    decimals : int
        Number of decimals to keep
    Returns
    -------
    float
        Number rounded up to nearest decimals decimals
    """
    multiplier = 10**decimals
    return ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    """
    Round down number to given number of decimals

    Parameters
    ----------
    n : float
        Number to round down
    decimals : int
        Number of decimals to keep

    Returns
    -------
    float
        Number rounded down to nearest decimals decimals
    """
    multiplier = 10**decimals
    return floor(n * multiplier) / multiplier


def lat_to_coord(lat):
    """
    Find coordinate placement of latitude

    Parameters
    ----------
    lat : float
        latitude

    Returns
    -------
    float
        Coordinate of latitude
    """
    return int(900 - round_up(lat, 1) * 10)


lat_to_coord_vector = np.vectorize(lat_to_coord)


def lon_to_coord(lon):
    """
    Find coordinateplacement of latitude

    Parameters
    ----------
    lon : float
        longitude

    Returns
    -------
    float
        Coordinate of longitude
    """
    return int(round_down(lon, 1) * 10)


lon_to_coord_vector = np.vectorize(lon_to_coord)


def get_data_for_lat_lon(variable, lat, lon):
    """
    Get variable data for given latitude and longitude

    Parameters
    ----------
    variable : str
        Variable to get data for, must be a key in the varstring dictionary
    lat : float
        latitude
    lon : float
        longitude

    Returns
    -------
    float
        Variable data in ERA5-land_2020.nc
    """
    filepath = f"{FOLDER}{varstring[variable]}_yearmean_ERA5-land_2020.nc"
    ds = xr.open_dataset(filepath)
    tas_data = ds[varstring[variable]][0, lat_to_coord(lat), lon_to_coord(lon)]
    return tas_data.values


def get_data_for_lat_lon_vector(variable, lat, lon):
    """
    Get variable data for given latitude and longitude

    Parameters
    ----------
    variable : str
        Variable to get data for, must be a key in the varstring dictionary
    lat : np.ndarray
        latitude, should be same length as lon-vector. Each entry should
        correspond to a single geographical point with the longitude
        with the same array placement.
    lon : np.ndarray
        longitue, should be same length as lon-vector.Each entry should
        correspond to a single geographical point with the latitude
        with the same array placement.

    Returns
    -------
    np.ndarray
        Variable data as vector for all the latitude and longitude pairs
    """
    filepath = f"{FOLDER}{varstring[variable]}_yearmean_ERA5-land_2020.nc"
    ds = xr.open_dataset(filepath)
    lat_coord = lat_to_coord_vector(lat)
    lon_coord = lon_to_coord_vector(lon)
    tas_data = ds[varstring[variable]][0].values[lat_coord, lon_coord]
    return tas_data


def get_monthly_data_for_var_lat_lon_vector(variable, lat, lon):
    """
    Get monthly variable data for given latitude and longitude

    Parameters
    ----------
    variable : str
        Variable to get data for, must be a key in the varstring dictionary
    lat : np.ndarray
        latitude, should be same length as lon-vector. Each entry should
        correspond to a single geographical point with the longitude
        with the same array placement.
    lon : np.ndarray
        longitue, should be same length as lon-vector.Each entry should
        correspond to a single geographical point with the latitude
        with the same array placement.

    Returns
    -------
    np.ndarray
        Monthly variable data as vector for all the latitude and longitude pairs
    """
    filepath = f"{FOLDER}{varstring[variable]}_Amon_ERA5-Land_2020.nc"
    ds = xr.open_dataset(filepath)
    lat_coord = lat_to_coord_vector(lat)
    lon_coord = lon_to_coord_vector(lon)
    var_data = ds[varstring[variable]][:].values[:, lat_coord, lon_coord]
    return var_data
