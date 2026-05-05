#!/usr/bin/env python
"""
Veteran-Level XGBoost Training Script (xgb_2).
Features:
- Out-of-Fold (OOF) Smoothed Target Encoding to prevent data leakage.
- Advanced Spatial: Hub distances, Coordinate Rotation (45 deg).
- Advanced Context: Traffic Density, Rush Hour, Airport Flags, Holidays.
- Bayesian Hyperparameter Optimization via Optuna.
- Log1p target transformation.
- Reads all context dynamically from local data/ folder.

Run: python train/xgb_2.py
"""

import os
import json
import pickle
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import geopandas as gpd
from sklearn.model_selection import KFold
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "model.pkl"

FEATURES =[
    "pickup_zone", "dropoff_zone", "passenger_count",
    "hour", "minute", "dow", "month", "day_of_year", "is_weekend", "is_holiday",
    "is_rush_hour", "is_airport_pickup", "is_airport_dropoff", "traffic_density",
    "min_of_day_sin", "min_of_day_cos", "dow_sin", "dow_cos",
    "p_lat", "p_lon", "d_lat", "d_lon", 
    "p_rot45_x", "p_rot45_y", "d_rot45_x", "d_rot45_y",
    "haversine_dist", "manhattan_dist", "bearing",
    "p_dist_jfk", "p_dist_lga", "p_dist_ewr", "p_dist_tsq",
    "d_dist_jfk", "d_dist_lga", "d_dist_ewr", "d_dist_tsq",
    "route_mean", "route_count", "route_std",
    "pu_mean", "do_mean", "route_hour_mean"
]

def load_local_context():
    print("Loading local context data...")
    with open(DATA_DIR / "holidays.json", "r") as f:
        ny_holidays = set(json.load(f))
        
    with open(DATA_DIR / "hubs.json", "r") as f:
        hubs = json.load(f)
        
    zone_lookup = pd.read_csv(DATA_DIR / "taxi_zone_lookup.csv")
    airport_zones = set(zone_lookup[zone_lookup['Zone'].str.contains('Airport', case=False, na=False)]['LocationID'])
    
    return ny_holidays, hubs, airport_zones

def get_zone_centroids() -> dict:
    print("Loading local NYC Taxi Zones shapefile...")
    shp_files = list((DATA_DIR / "taxi_zones").rglob("*.shp"))
    gdf = gpd.read_file(shp_files[0]).to_crs(epsg=2263)
    centroids = gdf.geometry.centroid.to_crs(epsg=4326)
    
    zone_centers = {row.LocationID: (centroid.y, centroid.x) for row, centroid in zip(gdf.itertuples(), centroids)}
    zone_centers[0] = zone_centers[264] = zone_centers[265] = (40.7128, -74.0060)
    return zone_centers

def haversine_array(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians,[lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def add_spatial_features(df: pd.DataFrame, zone_centers: dict, hubs: dict) -> pd.DataFrame:
    df["p_lat"] = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).astype("float32")
    df["p_lon"] = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).astype("float32")
    df["d_lat"] = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).astype("float32")
    df["d_lon"] = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).astype("float32")
    
    angle = np.pi / 4
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    df["p_rot45_x"] = (df["p_lon"] * cos_a - df["p_lat"] * sin_a).astype("float32")
    df["p_rot45_y"] = (df["p_lon"] * sin_a + df["p_lat"] * cos_a).astype("float32")
    df["d_rot45_x"] = (df["d_lon"] * cos_a - df["d_lat"] * sin_a).astype("float32")
    df["d_rot45_y"] = (df["d_lon"] * sin_a + df["d_lat"] * cos_a).astype("float32")
    
    df["haversine_dist"] = haversine_array(df["p_lat"], df["p_lon"], df["d_lat"], df["d_lon"]).astype("float32")
    df["manhattan_dist"] = (haversine_array(df["p_lat"], df["p_lon"], df["d_lat"], df["p_lon"]) + 
                            haversine_array(df["d_lat"], df["p_lon"], df["d_lat"], df["d_lon"])).astype("float32")
    
    p_lat_r, p_lon_r, d_lat_r, d_lon_r = map(np.radians,[df["p_lat"], df["p_lon"], df["d_lat"], df["d_lon"]])
    dlon = d_lon_r - p_lon_r
    y = np.sin(dlon) * np.cos(d_lat_r)
    x = np.cos(p_lat_r) * np.sin(d_lat_r) - np.sin(p_lat_r) * np.cos(d_lat_r) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y, x)).astype("float32")
    
    for hub_name, (h_lat, h_lon) in hubs.items():
        df[f"p_dist_{hub_name}"] = haversine_array(df["p_lat"], df["p_lon"], h_lat, h_lon).astype("float32")
        df[f"d_dist_{hub_name}"] = haversine_array(df["d_lat"], df["d_lon"], h_lat, h_lon).astype("float32")
        
    return df

def add_temporal_base(df: pd.DataFrame, holidays_set: set, airport_zones: set) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    df["hour"] = ts.dt.hour.astype("int8")
    df["minute"] = ts.dt.minute.astype("int8")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["day_of_year"] = ts.dt.dayofyear.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_holiday"] = ts.dt.strftime("%Y-%m-%d").isin(holidays_set).astype("int8")
    
    df["is_rush_hour"] = ((df["dow"] < 5) & (((df["hour"] >= 7) & (df["hour"] <= 9)) | ((df["hour"] >= 16) & (df["hour"] <= 19)))).astype("int8")
    df["is_airport_pickup"] = df["pickup_zone"].isin(airport_zones).astype("int8")
    df["is_airport_dropoff"] = df["dropoff_zone"].isin(airport_zones).astype("int8")
    
    min_of_day = df["hour"] * 60 + df["minute"]
    df["min_of_day_sin"] = np.sin(2 * np.pi * min_of_day / 1440).astype("float32")
    df["min_of_day_cos"] = np.cos(2 * np.pi * min_of_day / 1440).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7).astype("float32")
    df["passenger_count"] = df["passenger_count"].astype("float32")
    return df

def add_traffic_density(df: pd.DataFrame, traffic_agg: dict) -> pd.DataFrame:
    keys = list(zip(df["dow"], df["hour"]))
    df["traffic_density"] = [traffic_agg.get(k, 0) for k in keys]
    df["traffic_density"] = df["traffic_density"].astype("float32")
    return df

def get_encoding_stats(df: pd.DataFrame, smoothing: int = 20):
    global_mean = df["duration_seconds"].mean()
    
    route_agg = df.groupby("route_key")["duration_seconds"].agg(["sum", "count", "std"])
    route_stats = {k: ((row["sum"] + global_mean * smoothing) / (row["count"] + smoothing), row["count"], row["std"] if pd.notnull(row["std"]) else 0.0) for k, row in route_agg.iterrows()}
    
    rh_agg = df.groupby("route_hour_key")["duration_seconds"].agg(["sum", "count"])
    rh_stats = {k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing) for k, row in rh_agg.iterrows()}
    
    pu_agg = df.groupby("pickup_zone")["duration_seconds"].agg(["sum", "count"])
    pu_stats = {k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing) for k, row in pu_agg.iterrows()}
    
    do_agg = df.groupby("dropoff_zone")["duration_seconds"].agg(["sum", "count"])
    do_stats = {k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing) for k, row in do_agg.iterrows()}
    
    return route_stats, rh_stats, pu_stats, do_stats, global_mean

def apply_encodings(df: pd.DataFrame, stats: tuple) -> pd.DataFrame:
    route_stats, rh_stats, pu_stats, do_stats, global_mean = stats
    
    r_stats = df["route_key"].map(lambda k: route_stats.get(k, (global_mean, 0, 0.0)))
    df["route_mean"] = r_stats.apply(lambda x: x[0]).astype("float32")
    df["route_count"] = r_stats.apply(lambda x: x[1]).astype("int32")
    df["route_std"] = r_stats.apply(lambda x: x[2]).astype("float32")
    
    df["route_hour_mean"] = df["route_hour_key"].map(lambda k: rh_stats.get(k, global_mean)).astype("float32")
    df["pu_mean"] = df["pickup_zone"].map(lambda k: pu_stats.get(k, global_mean)).astype("float32")
    df["do_mean"] = df["dropoff_zone"].map(lambda k: do_stats.get(k, global_mean)).astype("float32")
    return df

def build_oof_encodings(train: pd.DataFrame, dev: pd.DataFrame):
    print("Building Out-Of-Fold Target Encodings...")
    train["route_key"] = train["pickup_zone"].astype(str) + "_" + train["dropoff_zone"].astype(str)
    train["route_hour_key"] = train["route_key"] + "_" + train["hour"].astype(str)
    dev["route_key"] = dev["pickup_zone"].astype(str) + "_" + dev["dropoff_zone"].astype(str)
    dev["route_hour_key"] = dev["route_key"] + "_" + dev["hour"].astype(str)
    
    for col in["route_mean", "route_count", "route_std", "route_hour_mean", "pu_mean", "do_mean"]:
        train[col] = np.nan
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train):
        tr_fold, val_fold = train.iloc[tr_idx], train.iloc[val_idx]
        stats = get_encoding_stats(tr_fold)
        val_encoded = apply_encodings(val_fold.copy(), stats)
        for col in["route_mean", "route_count", "route_std", "route_hour_mean", "pu_mean", "do_mean"]:
            train.iloc[val_idx, train.columns.get_loc(col)] = val_encoded[col]
            
    full_stats = get_encoding_stats(train)
    dev = apply_encodings(dev, full_stats)
    return train, dev, full_stats

def main():
    ny_holidays, hubs, airport_zones = load_local_context()
    zone_centers = get_zone_centroids()
    
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    
    print("Cleaning training data...")
    valid_duration = (train["duration_seconds"] >= 60) & (train["duration_seconds"] <= 7200)
    train = train[valid_duration].copy()
    
    print("Engineering temporal and spatial features...")
    train = add_temporal_base(train, ny_holidays, airport_zones)
    dev = add_temporal_base(dev, ny_holidays, airport_zones)
    
    traffic_agg = train.groupby(["dow", "hour"]).size().to_dict()
    train = add_traffic_density(train, traffic_agg)
    dev = add_traffic_density(dev, traffic_agg)
    
    train = add_spatial_features(train, zone_centers, hubs)
    dev = add_spatial_features(dev, zone_centers, hubs)
    
    train, dev, full_stats = build_oof_encodings(train, dev)
    route_stats, rh_stats, pu_stats, do_stats, global_mean = full_stats
    
    X_train, y_train = train[FEATURES], np.log1p(train["duration_seconds"].to_numpy())
    X_dev, y_dev = dev[FEATURES], np.log1p(dev["duration_seconds"].to_numpy())
    y_dev_raw = dev["duration_seconds"].to_numpy()

    print("\nRunning Optuna Hyperparameter Tuning (10 Trials)...")
    sample_idx = np.random.choice(len(X_train), min(1_000_000, len(X_train)), replace=False)
    X_tune, y_tune = X_train.iloc[sample_idx], y_train[sample_idx]
    
    def objective(trial):
        params = {
            "n_estimators": 400,
            "max_depth": trial.suggest_int("max_depth", 7, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tune, y_tune, eval_set=[(X_dev, y_dev)], verbose=False)
        return np.mean(np.abs(np.expm1(model.predict(X_dev)) - y_dev_raw))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"Best params found: {best_params}")

    print("\nTraining Final Model with Best Params...")
    best_params.update({"n_estimators": 3000, "early_stopping_rounds": 50, "tree_method": "hist", "n_jobs": -1, "random_state": 42})
    final_model = xgb.XGBRegressor(**best_params)
    
    t0 = time.time()
    final_model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=100)
    print(f"  trained in {time.time() - t0:.0f}s")

    mae = float(np.mean(np.abs(np.expm1(final_model.predict(X_dev)) - y_dev_raw)))
    print(f"\nFinal Dev MAE: {mae:.1f} seconds")

    artifact = {
        "model": final_model,
        "zone_centers": zone_centers,
        "route_stats": route_stats,
        "rh_stats": rh_stats,
        "pu_stats": pu_stats,
        "do_stats": do_stats,
        "global_mean": global_mean,
        "traffic_agg": traffic_agg,
        "holidays": list(ny_holidays),
        "hubs": hubs,
        "airport_zones": list(airport_zones),
        "features": FEATURES,
        "model_type": "xgb_2"
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model artifact to {MODEL_PATH}")

if __name__ == "__main__":
    main()