import os
import sys
import argparse

import xarray as xr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Extend path to include src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from fscve import FSCVE, ml_modelling_infrastructure, forest_data_handler

# --------------------------------
# Parse CLI argument for training data
# --------------------------------
parser = argparse.ArgumentParser(description="Train and run FSCVE emulator.")
parser.add_argument("training_data_path", type=str, help="Path to CSV training data.")
args = parser.parse_args()

# --------------------------------
# Load and clean training data
# --------------------------------
training_data = pd.read_csv(args.training_data_path)
required_columns = ["LATITUDE", "LONGITUDE", "ELEVATION", "AGB_ESA", "mean"]
training_data = training_data[required_columns].dropna()

resolution_file="/div/no-backup-nac/PATHFINDER/CERRA/CERRA_mean.nc"

X = training_data[["LATITUDE", "LONGITUDE", "ELEVATION", "AGB_ESA"]]
y = training_data["mean"]

# --------------------------------
# Train ML model
# --------------------------------
ml_model = ml_modelling_infrastructure.MLMODELINTERFACE(RandomForestRegressor)
#ml_model = ml_modelling_infrastructure.MLMODELINTERFACE(LinearRegression)
ml_model.train_new_model_instance(X, y)
#print(ml_model.evaluate_model(X,y))
#sys.exit(4)

# Wrap model in emulator
emulator = FSCVE(ml_model, list(X.columns), ["mean"])

# --------------------------------
# Strip Back Data just to Predictors so we can use these for predictions
# --------------------------------
map_data = pd.read_csv("/div/no-backup-nac/PATHFINDER/EMULATOR-DATA/FSCVE_5km_2015.csv")

# --------------------------------
# Define change scenarios
# --------------------------------
def apply_agb_change(df, change, var_to_change):
    df_changed = df.copy()
    if change == "halved":
        df_changed[var_to_change] = df_changed[var_to_change] / 2.0
    elif change == "doubled":
        df_changed[var_to_change] = df_changed[var_to_change] * 2.0
    elif change == "set_to_zero":
        df_changed[var_to_change] = 0.0
    elif change == "max_everywhere":
        df_changed[var_to_change] = df[var_to_change].max()
    return df_changed

# --------------------------------
# Plotting helpers
# --------------------------------
def plot_diff_map(base_df, changed_df, change_short, change_long=None):
    diff = emulator.predict_and_get_variable_diff(base_df, changed_df)
    #print(diff)
    #sys.exit(4)
    # Prepare DataFrame for plotting
    plot_df = base_df[["LATITUDE", "LONGITUDE"]].copy()
    plot_df["mean_diff"] = diff.values
    plot_df = plot_df.rename(columns={
    "LATITUDE": "lat",
    "LONGITUDE": "lon"
    })
    ##print(plot_df.head())
    #sys.exit(4)
    ds = forest_data_handler.make_sparse_forest_df_xarray(plot_df, resolution_file=resolution_file)
    #print(ds.head())
    #print(f"complete mean: {ds['mean_diff'].values.mean()}, complete max: {ds['mean_diff'].values.max()},complete min: {ds['mean_diff'].values.min()}")
    #sys.exit(4)
    # Ensure the subdirectory exists
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds["mean_diff"].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=-1,
        vmax=1
    )
    ax.coastlines()
    ax.set_extent([-10, 30, 35, 70])
    ax.set_title(f"Predicted change in mean when {change_long or change_short}")

    # Save plot in the subdirectory
    filename = f"prediction_change_{change_short}.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath)
    plt.clf()

# --------------------------------
# Standardize column names to what emulator expects
# --------------------------------
def prepare_emulator_df(df):
    return df.rename(columns={
        "lat_round": "LATITUDE",
        "lon_round": "LONGITUDE",
        "Elevation": "ELEVATION",
        "AGB_ESA": "AGB_ESA"  # unchanged
    })

def plot_vanilla(df,vname, figname, plot_dir = "."):
    ds = forest_data_handler.make_sparse_forest_df_xarray(df.copy(),resolution_file=resolution_file)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds[vname].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Greens",
        #vmin=-3,
        #vmax=3
    )
    ax.coastlines()
    ax.set_extent([-10, 30, 35, 70])
    ax.set_title(f"Data for {vname} in {figname.split('.')[0]}")

    # Save plot in the subdirectory
    filepath = os.path.join(plot_dir, figname)
    plt.savefig(filepath)
    plt.clf()
map_data.drop(columns="mean_mean", inplace=True)
map_data.drop_duplicates(inplace=True)

plot_vanilla(map_data, "AGB_ESA", "base_dataset_on_map.png")
map_data.rename(columns={"lat": "LATITUDE", "lon":"LONGITUDE"}, inplace=True)
map_data = map_data[["LATITUDE", "LONGITUDE", "ELEVATION", "AGB_ESA"]]
# --------------------------------
# Run prediction change simulations
# --------------------------------

predict_base_map = ml_model.predict_with_current(map_data)
map_data["mean"] = predict_base_map
plot_vanilla(map_data, "mean", "mean_prediction_base_map.png", plot_dir="plots")
for change in ["halved", "doubled", "set_to_zero", "max_everywhere"]:
    changed_map = apply_agb_change(map_data, change, var_to_change="AGB_ESA")
    plot_vanilla(changed_map, "AGB_ESA",f"agb_{change}_map.png", plot_dir="plots")
    base_emul_df = prepare_emulator_df(map_data)
    changed_emul_df = prepare_emulator_df(changed_map)
    plot_diff_map(base_emul_df, changed_emul_df, change, f"AGB_ESA is {change.replace('_', ' ')}")

