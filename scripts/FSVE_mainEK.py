import os
import sys
import argparse

import xarray as xr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

X = training_data[["LATITUDE", "LONGITUDE", "ELEVATION", "AGB_ESA"]]
y = training_data["mean"]

# --------------------------------
# Train ML model
# --------------------------------
ml_model = ml_modelling_infrastructure.MLMODELINTERFACE(RandomForestRegressor)
ml_model.train_new_model_instance(X, y)

# Wrap model in emulator
emulator = FSCVE(ml_model, list(X.columns), ["mean"])

# --------------------------------
# Strip Back Data just to Predictors so we can use these for predictions
# --------------------------------
map_data = training_data
map_data_unique = map_data.drop_duplicates(subset=['LATITUDE', 'LONGITUDE'])

# --------------------------------
# Define change scenarios
# --------------------------------
def apply_agb_change(df, change):
    df_changed = df.copy()
    if change == "halved":
        df_changed["AGB_ESA"] = df_changed["AGB_ESA"] / 2.0
    elif change == "doubled":
        df_changed["AGB_ESA"] = df_changed["AGB_ESA"] * 2.0
    elif change == "set_to_zero":
        df_changed["AGB_ESA"] = 0.0
    return df_changed

# --------------------------------
# Plotting helpers
# --------------------------------
def plot_diff_map(base_df, changed_df, change_short, change_long=None):
    diff = emulator.predict_and_get_variable_diff(base_df, changed_df)
    
    # Prepare DataFrame for plotting
    plot_df = base_df[["LATITUDE", "LONGITUDE"]].copy()
    plot_df["mean_diff"] = diff
    plot_df = plot_df.rename(columns={
    "LATITUDE": "lat",
    "LONGITUDE": "lon"
    })
    # print(plot_df)
    ds = forest_data_handler.make_sparse_forest_df_xarray(plot_df)
    # print(ds)

    # Ensure the subdirectory exists
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ds["mean_diff"].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="bwr",
        vmin=-3,
        vmax=3
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

# --------------------------------
# Run prediction change simulations
# --------------------------------
# def plot_vanilla(ds, name, plot_name):
#     ds = forest_data_handler.make_sparse_forest_df_xarray(ds)
#     # Plotting
#     fig = plt.figure(figsize=(10, 5))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ds[name].plot(
#         ax=ax, transform=ccrs.PlateCarree(), cmap="bwr", vmin=-3, vmax=3
#     )
#     ax.coastlines()
#     ax.set_extent([-10, 30, 35, 70])
#     ax.set_title(f"{plot_name}")
#     plt.savefig(f"temp_change_{plot_name}.png")
#     plt.clf()

# plot_vanilla(map_data, "mean", "base")


# def plot_agb_esa_scatter(df, save_path="agb_esa_scatter.png"):
#     fig = plt.figure(figsize=(10, 6))
#     ax = plt.axes(projection=ccrs.PlateCarree())
    
#     # Add map features
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=":")
#     ax.add_feature(cfeature.LAND, facecolor='lightgray')
#     ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

#     # Plot AGB_ESA values
#     sc = ax.scatter(
#         df["LONGITUDE"], df["LATITUDE"],
#         c=df["AGB_ESA"], cmap="YlGn", s=10, alpha=0.8,
#         transform=ccrs.PlateCarree()
#     )
    
#     # Add colorbar and title
#     cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.6, pad=0.05)
#     cbar.set_label("AGB_ESA")
#     ax.set_title("AGB_ESA Values Across Locations")

#     # Save plot
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
# plot_agb_esa_scatter(map_data[["LATITUDE", "LONGITUDE", "AGB_ESA"]])

for change in ["halved", "doubled", "set_to_zero"]:
    print(change)
    print(map_data)
    changed_map = apply_agb_change(map_data, change)

    base_emul_df = prepare_emulator_df(map_data)
    changed_emul_df = prepare_emulator_df(changed_map)
    # plot_vanilla(changed_map, "mean", f"Changes agb {change}")
    plot_diff_map(base_emul_df, changed_emul_df, change, f"AGB_ESA is {change.replace('_', ' ')}")

