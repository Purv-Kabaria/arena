"""
Highly Optimized XGBoost Training Script (xgb_1).
Features:
- Geospatial features (Haversine, Manhattan, Bearing, raw Lat/Lon).
- Multi-level Smoothed Target Encoding (Route, Pickup, Dropoff).
- Log-transformed target for stable tree building.
- Aggressive hyperparameter tuning & Early Stopping.

Run: python train/xgb_1.py
"""

import os
import io
import zipfile
import pickle
import time
import math
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import geopandas as gpd

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgb_1.pkl"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "pickup_zone", "dropoff_zone", "hour", "minute", "dow", "is_weekend", "month", "passenger_count",
    "p_lat", "p_lon", "d_lat", "d_lon", "haversine_dist", "manhattan_dist", "bearing",
    "route_mean", "route_count", "pickup_mean", "dropoff_mean"
]

# --- 1. GEOSPATIAL PROCESSING ---
def get_zone_centroids() -> dict:
    print("Downloading and processing NYC Taxi Zones shapefile...")
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    r = requests.get(url)
    
    extract_dir = DATA_DIR / "taxi_zones"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(extract_dir)
    
    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"Could not find any .shp file in {extract_dir}")
    
    gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
    centroids = gdf.geometry.centroid
    
    zone_centers = {
        row.LocationID: (centroid.y, centroid.x)
        for row, centroid in zip(gdf.itertuples(), centroids)
    }
    zone_centers[0] = (40.7128, -74.0060)
    zone_centers[264] = (40.7128, -74.0060)
    zone_centers[265] = (40.7128, -74.0060)
    return zone_centers

def add_spatial_features(df: pd.DataFrame, zone_centers: dict) -> pd.DataFrame:
    df["p_lat"] = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).astype("float32")
    df["p_lon"] = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).astype("float32")
    df["d_lat"] = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).astype("float32")
    df["d_lon"] = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).astype("float32")
    
    p_lat_r, p_lon_r, d_lat_r, d_lon_r = map(np.radians, [df["p_lat"], df["p_lon"], df["d_lat"], df["d_lon"]])
    
    dlon = d_lon_r - p_lon_r
    dlat = d_lat_r - p_lat_r
    a = np.sin(dlat/2.0)**2 + np.cos(p_lat_r) * np.cos(d_lat_r) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine_dist"] = (6371 * c).astype("float32")
    
    a_lat = np.sin(dlat/2.0)**2
    a_lon = np.cos(p_lat_r) * np.cos(d_lat_r) * np.sin(dlon/2.0)**2
    df["manhattan_dist"] = (6371 * (2 * np.arcsin(np.sqrt(a_lat)) + 2 * np.arcsin(np.sqrt(a_lon)))).astype("float32")
    
    y = np.sin(dlon) * np.cos(d_lat_r)
    x = np.cos(p_lat_r) * np.sin(d_lat_r) - np.sin(p_lat_r) * np.cos(d_lat_r) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y, x)).astype("float32")
    
    return df

# --- 2. TARGET ENCODING ---
def build_target_encodings(train_df: pd.DataFrame, smoothing: int = 15):
    global_mean = train_df["duration_seconds"].mean()
    
    # 1. Route Stats
    train_df["route_key"] = train_df["pickup_zone"].astype(str) + "_" + train_df["dropoff_zone"].astype(str)
    route_agg = train_df.groupby("route_key")["duration_seconds"].agg(["sum", "count"])
    route_stats = {
        k: ((row["sum"] + global_mean * smoothing) / (row["count"] + smoothing), row["count"])
        for k, row in route_agg.iterrows()
    }
    
    # 2. Pickup Stats
    pu_agg = train_df.groupby("pickup_zone")["duration_seconds"].agg(["sum", "count"])
    pu_stats = {
        k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing)
        for k, row in pu_agg.iterrows()
    }
    
    # 3. Dropoff Stats
    do_agg = train_df.groupby("dropoff_zone")["duration_seconds"].agg(["sum", "count"])
    do_stats = {
        k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing)
        for k, row in do_agg.iterrows()
    }
    
    return route_stats, pu_stats, do_stats, global_mean

def apply_target_encodings(df: pd.DataFrame, route_stats: dict, pu_stats: dict, do_stats: dict, global_mean: float) -> pd.DataFrame:
    route_keys = df["pickup_zone"].astype(str) + "_" + df["dropoff_zone"].astype(str)
    
    r_stats = route_keys.map(lambda k: route_stats.get(k, (global_mean, 0)))
    df["route_mean"] = r_stats.apply(lambda x: x[0]).astype("float32")
    df["route_count"] = r_stats.apply(lambda x: x[1]).astype("int32")
    
    df["pickup_mean"] = df["pickup_zone"].map(lambda k: pu_stats.get(k, global_mean)).astype("float32")
    df["dropoff_mean"] = df["dropoff_zone"].map(lambda k: do_stats.get(k, global_mean)).astype("float32")
    
    return df

# --- 3. MAIN PIPELINE ---
def main():
    zone_centers = get_zone_centroids()
    
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    
    print("Cleaning training data...")
    valid_duration = (train["duration_seconds"] >= 60) & (train["duration_seconds"] <= 7200)
    train = train[valid_duration].copy()
    
    print("Engineering time features...")
    for df in [train, dev]:
        ts = pd.to_datetime(df["requested_at"])
        df["hour"] = ts.dt.hour.astype("int8")
        df["minute"] = ts.dt.minute.astype("int8")
        df["dow"] = ts.dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dow"] >= 5).astype("int8")
        df["month"] = ts.dt.month.astype("int8")
        df["passenger_count"] = df["passenger_count"].astype("int8")
        
    print("Engineering spatial features...")
    train = add_spatial_features(train, zone_centers)
    dev = add_spatial_features(dev, zone_centers)
    
    print("Building target encodings...")
    route_stats, pu_stats, do_stats, global_mean = build_target_encodings(train)
    train = apply_target_encodings(train, route_stats, pu_stats, do_stats, global_mean)
    dev = apply_target_encodings(dev, route_stats, pu_stats, do_stats, global_mean)
    
    X_train = train[FEATURES]
    # LOG TRANSFORM TARGET
    y_train = np.log1p(train["duration_seconds"].to_numpy())
    
    X_dev = dev[FEATURES]
    y_dev = np.log1p(dev["duration_seconds"].to_numpy())
    y_dev_raw = dev["duration_seconds"].to_numpy() # Keep raw for final MAE calculation

    print("\nTraining XGBoost (xgb_1)...")
    model = xgb.XGBRegressor(
        n_estimators=3000,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=20,
        tree_method="hist",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        verbose=100
    )
    print(f"  trained in {time.time() - t0:.0f}s")

    # Predict and inverse log transform
    preds_log = model.predict(X_dev)
    preds_raw = np.expm1(preds_log)
    
    mae = float(np.mean(np.abs(preds_raw - y_dev_raw)))
    print(f"\nFinal Dev MAE: {mae:.1f} seconds")

    artifact = {
        "model": model,
        "zone_centers": zone_centers,
        "route_stats": route_stats,
        "pu_stats": pu_stats,
        "do_stats": do_stats,
        "global_mean": global_mean,
        "features": FEATURES,
        "model_type": "xgb_1"
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model artifact to {MODEL_PATH}")

if __name__ == "__main__":
    main()