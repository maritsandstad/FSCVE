import numpy as np
import pandas as pd
import xarray as xr
from math import ceil, floor

folder = "/div/no-backup-nac/reanalysis/ERA5-land/"
varstring = {"T2M": "tas","T2DEW": "t2dew","RLDS-ERA": "rlds", "HGT-ERA": "hgt", "PR-ERA": "pr", "RSDS-ERA": "rsds", "volume": "vol", "agb": "agb"}
def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return ceil(n * multiplier) / multiplier

def round_down(n, decimals=0): 
    multiplier = 10 ** decimals 
    return floor(n * multiplier) / multiplier

def lat_to_coord(lat):
    return int(900 -round_up(lat,1)*10)

lat_to_coord_vector = np.vectorize(lat_to_coord)

def lon_to_coord(lon):
    return int(round_down(lon, 1)*10)

lon_to_coord_vector = np.vectorize(lon_to_coord)

def get_data_for_lat_lon(variable, lat,lon):
    filepath = f"{folder}{varstring[variable]}_yearmean_ERA5-land_2020.nc"
    ds = xr.open_dataset(filepath)
    tas_data = ds[varstring[variable]][0, lat_to_coord(lat), lon_to_coord(lon)]
    return tas_data.values

def get_data_for_lat_lon_vector(variable, lat,lon):
    filepath = f"{folder}{varstring[variable]}_yearmean_ERA5-land_2020.nc"
    ds = xr.open_dataset(filepath)
    lat_coord = lat_to_coord_vector(lat)
    lon_coord = lon_to_coord_vector(lon)
    tas_data = ds[varstring[variable]][0].values[lat_coord,lon_coord]
    print(tas_data.shape)
    return tas_data

def get_monthly_data_for_var_lat_lon_vector(variable, lat, lon):
    filepath = f"{folder}{varstring[variable]}_Amon_ERA5-Land_2020.nc"
    var_data_full = np.zeros((12, len(lat)))
    ds = xr.open_dataset(filepath)
    lat_coord = lat_to_coord_vector(lat)
    lon_coord = lon_to_coord_vector(lon)
    print(ds[varstring[variable]][:])
    var_data = ds[varstring[variable]][:].values[:,lat_coord,lon_coord]
    print(var_data.shape)
    return var_data

if __name__ == "__main__":
    test = get_data_for_lat_lon("T2M", 52.377956, 4.897070)
    print(lat_to_coord(89.91))
    print(test)
    test2 = get_monthly_data_for_var_lat_lon_vector("RSDS-ERA", [52.377956, 60], [4.897070, 10.75])
    print(test2)
    test_df = pd.DataFrame({"Station": ["Amsterdam", "Oslo"], "Country":["Netherlands", "Norway"]})
    month_columns = [f"RSDS_ERA_{m:02}" for m in range(1,13)]
    test_df = test_df.reindex(columns=test_df.columns.to_list()+ month_columns)
    test_df[month_columns] = test2.T
    print(test_df)