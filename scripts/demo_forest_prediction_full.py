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

training_data = pd.read_csv("/div/nac/users/kjetisaa/PATHFINDER/PathFinder_WP3_Task3.5_emulator/All_data_GHCNdaily_wForestAndCorine_Europe_2018-2022.csv")
print(training_data)
keep = ["LAT", "LON", "ELEVATION","forest_fraction_101"]
keep_no_forest = ["LAT", "LON", "ELEVATION"]
predict = "TAVG_7"
for column in training_data.columns:
    if column in keep or column == predict:
        continue
    training_data.drop(columns=column, inplace=True)
training_data.dropna(inplace=True)
X_unscaled = training_data[keep]
y_unscaled = training_data[predict]
ml_model_linear = ml_modelling_infrastructure.MLMODELINTERFACE(RandomForestRegressor)
ml_model_linear.train_new_model_instance(X_unscaled, y_unscaled)

#sys.exit(4)
forest_sensitive_emulator = FSCVE(ml_model_linear, keep, [predict])

#era_datafile = "/div/no-backup-nac/users/masan/PATHFINDER/PathFinder_WP3_Task3.5_emulator/agb_and_confier_on_era5-land_resolution_test.nc"
era_datafile = "/div/no-backup-nac/users/masan/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc"

data_forest = forest_data_handler.add_variables_to_forest_dataset(
        era_datafile,
        full_variable_list=["LAT", "LON", "ELEVATION"], forest_variables=["lsm"],
    )
print(data_forest.shape)
print(3600*1801)
#data_forest = data_forest.loc[(data_forest.LON < 30.0) | (data_forest.LON > 350.0)]

print("TESTTEST3")
data_forest["LON"] = data_forest["LON"].round(1)
data_forest["LAT"] = data_forest["LAT"].round(1)
print(data_forest)

data_forest = data_forest.loc[((data_forest.LON < 30.0) | (data_forest.LON > 350.0)) & (data_forest.LAT > 35) & (data_forest.LAT < 70)]
#print(data_forest.LON.max())
#print(data_forest.LON.min())
#sys.exit(4)

#print(data_forest.shape)
#data_forest = data_forest[data_forest.LAT > 35]
#print(data_forest.shape)
#data_forest = data_forest[data_forest.LAT < 70]
#data_forest = data_forest[data_forest.hgt > 0.1453892]
#data_forest['LON'] = (10*data_forest['LON']).astype(int)/10
#print(data_forest.shape)
#print(data_forest.columns)
#sys.exit(4)

data_forest.drop(columns = ["time", "lsm"], inplace=True)
data_noforest = data_forest.copy()[keep_no_forest]
data_noforest["forest_fraction_101"] = np.zeros(data_forest.shape[0])
data_allforest = data_forest.copy()[keep_no_forest]
data_allforest["forest_fraction_101"] = np.ones(data_forest.shape[0])

print(data_allforest)
#sys.exit(4)
ds_europe_land = forest_data_handler.make_sparse_forest_df_xarray(data_allforest.copy())
print(ds_europe_land)
print(ds_europe_land["forest_fraction_101"].where(ds_europe_land["forest_fraction_101"] > 0))
print(ds_europe_land["forest_fraction_101"].where(ds_europe_land["forest_fraction_101"] > 0).shape)
print(ds_europe_land["forest_fraction_101"].sel(lat = slice(35.1,70), lon=slice(10.1,30)))
print(ds_europe_land["forest_fraction_101"].sel(lat = slice(35.1,70), lon=slice(10.1,30)).shape)
print(ds_europe_land["forest_fraction_101"].sel(lat = slice(35.1,70), lon=slice(10.1,30)).min())
print(ds_europe_land["forest_fraction_101"].sel(lat = slice(35.1,70), lon=slice(10.1,30)).max())
print(ds_europe_land["forest_fraction_101"].sel(lat = slice(35.1,70), lon=slice(10.1,30)).mean())
print(ds_europe_land["forest_fraction_101"].mean())

fig = plt.figure(figsize=(10,5))
fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection=ccrs.PlateCarree())
#ax.contourf(data_forest["LON"], data_forest["LAT"], july_temp_change, transform=ccrs.PlateCarree().values)
ds_europe_land["forest_fraction_101"].plot(ax=ax, transform=ccrs.PlateCarree(), cmap ="bwr", vmin=-3, vmax=3)
ax.coastlines()
ax.set_title("Europe land")
plt.savefig("europe_land.png")

ds_europe_no_land = forest_data_handler.make_sparse_forest_df_xarray(data_noforest.copy())

fig = plt.figure(figsize=(10,5))
fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection=ccrs.PlateCarree())
#ax.contourf(data_forest["LON"], data_forest["LAT"], july_temp_change, transform=ccrs.PlateCarree().values)
ds_europe_no_land["forest_fraction_101"].plot(ax=ax, transform=ccrs.PlateCarree(), cmap ="bwr", vmin=-3, vmax=3)
ax.coastlines()
ax.set_title("No land")
plt.savefig("europe_no_land.png")

#sys.exit(4)

all_forest_pred = ml_model_linear.predict_with_current(data_allforest)
no_forest_pred = ml_model_linear.predict_with_current(data_noforest)
print(all_forest_pred-no_forest_pred)
#sys.exit(4)
july_temp_change = forest_sensitive_emulator.predict_and_get_variable_diff(data_noforest, data_allforest)
print(july_temp_change.mean())
print(july_temp_change.max())
print(july_temp_change.min())
print(july_temp_change)

#plt.scatter( data_forest["LAT"], july_temp_change)
#plt.savefig("July_temp_change_per_lat.png")
#plt.scatter( data_forest["LON"], july_temp_change)
#plt.savefig("July_temp_change_per_lon.png")
data_july_temp = data_forest[["LAT", "LON"]].copy()
data_july_temp["july_temp_change"] = july_temp_change
ds_july_temp = forest_data_handler.make_sparse_forest_df_xarray(data_july_temp)
#sys.exit(4)

fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection=ccrs.PlateCarree())
#ax.contourf(data_forest["LON"], data_forest["LAT"], july_temp_change, transform=ccrs.PlateCarree().values)
ds_july_temp["july_temp_change"].plot(ax=ax, transform=ccrs.PlateCarree(), cmap ="bwr", vmin=-3, vmax=3)
ax.coastlines()
ax.set_title("Temperature change no forest to all forest")
plt.savefig("temp_change_max_change.png")

