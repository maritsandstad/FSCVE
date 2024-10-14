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
    "/div/nac/users/kjetisaa/PATHFINDER/PathFinder_WP3_Task3.5_emulator/All_data_GHCNdaily_wForestAndCorine_Europe_2018-2022.csv"
)
keep = ["LAT", "LON", "ELEVATION", "forest_fraction_101"]
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


# Getting a datafile to demo on. Here we just use a land mask to get all of Europe
# era_datafile = "/div/no-backup-nac/users/masan/PATHFINDER/PathFinder_WP3_Task3.5_emulator/agb_and_confier_on_era5-land_resolution_test.nc"
era_datafile = "/div/no-backup-nac/users/masan/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc"

# Making it into a DataFrame and keeping only non-zero values (i.e. land)
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

# Interlude to plot a map of the forrested dataset
ds_europe_land = forest_data_handler.make_sparse_forest_df_xarray(data_allforest.copy())
fig = plt.figure(figsize=(10, 5))
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ds_europe_land["forest_fraction_101"].plot(
    ax=ax, transform=ccrs.PlateCarree(), cmap="bwr", vmin=-3, vmax=3
)
ax.coastlines()
ax.set_title("Europe land")
plt.savefig("europe_land.png")

# And the non forested
ds_europe_no_land = forest_data_handler.make_sparse_forest_df_xarray(
    data_noforest.copy()
)
fig = plt.figure(figsize=(10, 5))
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ds_europe_no_land["forest_fraction_101"].plot(
    ax=ax, transform=ccrs.PlateCarree(), cmap="bwr", vmin=-3, vmax=3
)
ax.coastlines()
ax.set_title("No land")
plt.savefig("europe_no_land.png")

# Making a diff prediction and putting it into an xarray format
july_temp_change = forest_sensitive_emulator.predict_and_get_variable_diff(
    data_noforest, data_allforest
)
data_july_temp = data_forest[["LAT", "LON"]].copy()
data_july_temp["july_temp_change"] = july_temp_change
ds_july_temp = forest_data_handler.make_sparse_forest_df_xarray(data_july_temp)

# Plotting
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ds_july_temp["july_temp_change"].plot(
    ax=ax, transform=ccrs.PlateCarree(), cmap="bwr", vmin=-3, vmax=3
)
ax.coastlines()
ax.set_title("Temperature change all forest to no forest")
plt.savefig("temp_change_max_change.png")
