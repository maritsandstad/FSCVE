import os
import sys
import argparse

import xarray as xr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    print("predicted")
    plot_df = base_df[["LATITUDE", "LONGITUDE"]].copy()
    plot_df["mean_diff"] = diff
    ds = forest_data_handler.make_sparse_forest_df_xarray(plot_df)
    print("sparsened")

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
    plt.savefig(f"prediction_change_{change_short}.png")
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
for change in ["halved", "doubled", "set_to_zero"]:
    print(change)
    changed_map = apply_agb_change(map_data, change)

    base_emul_df = prepare_emulator_df(map_data)
    changed_emul_df = prepare_emulator_df(changed_map)

    plot_diff_map(base_emul_df, changed_emul_df, change, f"AGB_ESA is {change.replace('_', ' ')}")

