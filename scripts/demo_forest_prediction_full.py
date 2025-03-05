import os
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr


# Adding location of source code to system path
# os.path.dirname(__file__) gives the directory of
# current file. Put in updated path if running script from elsewhere
# os.path joins all the folders of a path together in a
# system independent way (i.e. will work equally well on Windows, linux etc)
sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from fscve import FSCVE, ml_modelling_infrastructure, forest_data_handler

# Reading in traning data and organising it a bit, keeping columns of interest
# and throwing out rows with nans in predictors and targets
training_data = pd.read_csv(
    "/div/nac/users/kjetisaa/PATHFINDER/PathFinder_WP3_Task3.5_emulator/All_data_GHCNdaily_wForestAndCorineFromFile_Europe_2016_chelsa_wCI.csv"
)
keep = ["LAT", "LON", "ELEVATION", "conifer-ratio_smooth_101", "agb_smooth_101"]
keep_no_forest = ["LAT", "LON", "ELEVATION"]
predict = "TAVG_7"
for column in training_data.columns:
    if column in keep or column == predict:
        continue
    training_data.drop(columns=column, inplace=True)
training_data.dropna(inplace=True)
X_unscaled = training_data[keep]
y_unscaled = training_data[predict]

# Defining and training a machine learning model with the training data
ml_model_linear = ml_modelling_infrastructure.MLMODELINTERFACE(RandomForestRegressor)
ml_model_linear.train_new_model_instance(X_unscaled, y_unscaled)

# Forest emulator from the trained ML model
forest_sensitive_emulator = FSCVE(ml_model_linear, keep, [predict])

def get_forest_diff_and_plot(data_base, data_change, change_name_short, change_name_long = None):

    # Making a diff prediction and putting it into an xarray format
    july_temp_change = forest_sensitive_emulator.predict_and_get_variable_diff(
        data_base, data_change
    )
    data_july_temp = data_base[["LAT", "LON"]].copy()
    data_july_temp["july_temp_change"] = july_temp_change
    ds_july_temp = forest_data_handler.make_sparse_forest_df_xarray(data_july_temp)

    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds_july_temp["july_temp_change"].plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap="bwr", vmin=-3, vmax=3
    )
    if change_name_long is None:
        change_name_long = change_name_short

    ax.coastlines()
    ax.set_extent([-10, 30, 35, 70])
    ax.set_title(f"Temperature change when {change_name_long}")
    plt.savefig(f"temp_change_{change_name_short}.png")

# Getting a datafile to demo on. Here we just use a land mask to get all of Europe
# era_datafile = "/div/no-backup-nac/users/masan/PATHFINDER/PathFinder_WP3_Task3.5_emulator/agb_and_confier_on_era5-land_resolution_test.nc"
# era_datafile = "/div/no-backup-nac/users/masan/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc"
conifer_file = "/div/no-backup-nac/PATHFINDER/PathFinder_data/2020_agb_conifers_E_regridded.nc"
agb_file = "/div/no-backup-nac/PATHFINDER/PathFinder_data/2020_agb_E_regridded.nc"
vol_file = "/div/no-backup-nac/PATHFINDER/PathFinder_data/2020_vol_E_regridded.nc"
# Making it into a DataFrame and keeping only non-zero values (i.e. land)
"""
data_forest = forest_data_handler.add_variables_to_forest_dataset(
    era_datafile,
    full_variable_list=["LAT", "LON", "ELEVATION"],
    forest_variables=["lsm"],
)
# Cutting to Europeish box
data_forest = data_forest.loc[
    ((data_forest.LON < 30.0) | (data_forest.LON > 350.0))
    & (data_forest.LAT > 35)
    & (data_forest.LAT < 70)
]

# Dropping unused columns and makeing a version with no forest and one with all forest
data_forest.drop(columns=["time", "lsm"], inplace=True)
data_noforest = data_forest.copy()[keep_no_forest]
data_noforest["forest_fraction_101"] = np.zeros(data_forest.shape[0])
data_allforest = data_forest.copy()[keep_no_forest]
data_allforest["forest_fraction_101"] = np.ones(data_forest.shape[0])
"""
def plot_map(dataset, varname, filename, cmp="Greens"):
    # Interlude to plot a map of the forrested dataset
    ds_europe_land = forest_data_handler.make_sparse_forest_df_xarray(dataset.copy())
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds_europe_land[varname].plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap=cmp, #vmin=-3, vmax=3
    )
    ax.set_extent([-10, 30, 35, 70])
    ax.coastlines()
    ax.set_title(f"Europe {varname}")
    plt.savefig(filename)
    plt.clf()

def plot_lat_lon_values(dataset, varname):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].hist(dataset["LAT"])
    axs[0].set_xlabel("Lat values")
    axs[1].hist(dataset["LON"])
    axs[1].set_xlabel("Lon values")
    fig.suptitle(f"Histograms of lat and lons for {varname}")
    axs[0].set_title("LAT")
    axs[1].set_title("LON")
    fig.savefig(f"{varname}_lat_lon_hist.png")
    plt.clf()

data_conifer = forest_data_handler.add_variables_to_forest_dataset(
    conifer_file,
    full_variable_list=["LAT", "LON", "ELEVATION"],
    forest_variables=["2020_agb_conifers_E_regridded"],
)


#data_conifer.rename(columns= {"LAT":"LON", "LON":"LAT"}, inplace=True)
#print(data_conifer)
#data_conifer["2020_agb_conifers_E_regridded"] = data_conifer["2020_agb_conifers_E_regridded"].where(data_conifer["2020_agb_conifers_E_regridded"] < 65533.0, 0)
data_agb = forest_data_handler.add_variables_to_forest_dataset(
    agb_file,
    full_variable_list=["LAT", "LON", "ELEVATION"],
    forest_variables=["2020_agb_E_regridded"],
)

data_vol = forest_data_handler.add_variables_to_forest_dataset(
    vol_file,
    full_variable_list=["LAT", "LON", "ELEVATION"],
    forest_variables=["2020_vol_E_regridded"],
)


datasets = {"agb": [data_agb, "2020_agb_E_regridded"], "conifers": [data_conifer, "2020_agb_conifers_E_regridded"], "vol": [data_vol, "2020_vol_E_regridded"],}
"""
for datasetname, dataset_list in datasets.items():
    dataset_list[0][dataset_list[1]] = dataset_list[0][dataset_list[1]].where(dataset_list[0][dataset_list[1]] < 65533.0, 0)
    plot_map(dataset_list[0], dataset_list[1], f"{datasetname}_europe_2020_map.png")
    plot_lat_lon_values(dataset_list[0], datasetname)
    print(f"{datasetname} values: {np.min(dataset_list[0][dataset_list[1]].values)} - {np.max(dataset_list[0][dataset_list[1]].values)} ")
    print(f"Lat lon extent: {np.min(dataset_list[0]['LAT'].values)} - {np.max(dataset_list[0]['LAT'].values)} and {np.min(dataset_list[0]['LON'].values)} - {np.max(dataset_list[0]['LON'].values)}")
"""
def make_change_dataset(orig_dataset, changev, change):
    change_ds = orig_dataset.copy()
    for cvar in changev:
        if change == "halved":
            change_ds[cvar] = change_ds[cvar]/2.0
        elif change == "set_to_zero":
            change_ds[cvar] = 0.0
        elif change == "doubled":
            change_ds[cvar] = change_ds[cvar]*2.0
            if "conifer" in cvar:
                change_ds[cvar] = change_ds[cvar].where(change_ds[cvar] > 100., 100.)
    return change_ds
                
composite_base_dataset = data_agb.copy().merge(data_conifer, how="outer", on =['LAT', 'LON', 'ELEVATION', "crs"])
composite_base_dataset.drop(columns=["crs"], inplace = True)
composite_base_dataset['2020_agb_conifers_E_regridded'] = composite_base_dataset['2020_agb_conifers_E_regridded'].fillna(0)
composite_base_dataset['2020_agb_E_regridded'] = composite_base_dataset['2020_agb_E_regridded'].fillna(0)
composite_base_dataset.rename(columns={'2020_agb_E_regridded': 'agb_smooth_101', "2020_agb_conifers_E_regridded": 'conifer-ratio_smooth_101'}, inplace=True)
composite_base_dataset = composite_base_dataset.reindex(columns=['LAT', 'LON', 'ELEVATION', 'conifer-ratio_smooth_101', 'agb_smooth_101'])

short_to_longname_dictionary = {'agb': 'agb_smooth_101', "conifers": 'conifer-ratio_smooth_101'}
print(composite_base_dataset)
print(composite_base_dataset.shape)
print(data_conifer.shape)
print(data_conifer)
print(data_agb.shape)
print(data_agb)
print(composite_base_dataset)
print(composite_base_dataset.shape)
change_list = ["set_to_zero", "halved", "doubled"]
variables_to_change = ["agb", "conifers", "both"]
for changes in change_list:
    for changev in variables_to_change:
        change_name_short = f"{changev}_{changes}"
        change_name_long = f"{changev} variable is {(' ').join(changes.split('_'))}"
        if changev == "both":
            changev_list =  [short_to_longname_dictionary["agb"], short_to_longname_dictionary["conifers"]]
        else:
            changev_list = [short_to_longname_dictionary[changev]]
        change_ds = make_change_dataset(composite_base_dataset, changev_list, change=changes)
        get_forest_diff_and_plot(composite_base_dataset, change_ds, change_name_short, change_name_long = change_name_long)



