#!/usr/bin/env python
"""
Veteran-Level Decision Tree Ensemble Script (dt_2).
Features:
- Random Forest Regressor (Ensemble of Decision Trees).
- Expected Duration Feature (Distance / Historical Route Speed).
- Coordinate Clustering (MiniBatchKMeans with 100 clusters, per PDF).
- Rotated Coordinates (30, 45, 60 degrees) to help trees split diagonally.
- Local Weather Data Integration.
- Bayesian Hyperparameter Optimization via Optuna.

Run: python train/dt_2.py
"""

import os
import json
import pickle
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "dt_2.pkl"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES =[
    "passenger_count", "hour", "minute", "dow", "month", "day_of_year",
    "is_weekend", "is_holiday", "is_rush_hour", "is_airport_pickup", "is_airport_dropoff",
    "traffic_density", "min_of_day_sin", "min_of_day_cos", "dow_sin", "dow_cos",
    "p_lat", "p_lon", "d_lat", "d_lon",
    "p_rot30_x", "p_rot30_y", "d_rot30_x", "d_rot30_y",
    "p_rot45_x", "p_rot45_y", "d_rot45_x", "d_rot45_y",
    "p_rot60_x", "p_rot60_y", "d_rot60_x", "d_rot60_y",
    "haversine_dist", "manhattan_dist", "bearing",
    "p_dist_jfk", "p_dist_lga", "p_dist_ewr", "p_dist_tsq",
    "d_dist_jfk", "d_dist_lga", "d_dist_ewr", "d_dist_tsq",
    "temp", "precip", "wind",
    "route_mean", "route_count", "route_speed_mean", "expected_duration",
    "cluster_route_mean", "pu_mean", "do_mean"
]

# --- 1. LOAD LOCAL CONTEXT ---
def load_local_context():
    print("Loading local context data...")
    with open(DATA_DIR / "holidays.json", "r") as f:
        ny_holidays = set(json.load(f))
    with open(DATA_DIR / "hubs.json", "r") as f:
        hubs = json.load(f)
    zone_lookup = pd.read_csv(DATA_DIR / "taxi_zone_lookup.csv")
    airport_zones = set(zone_lookup[zone_lookup['Zone'].str.contains('Airport', case=False, na=False)]['LocationID'])
    return ny_holidays, hubs, airport_zones

def get_local_weather():
    print("Loading local weather data...")
    df = pd.read_csv(DATA_DIR / "weather_2023_2024.csv")
    df['time'] = pd.to_datetime(df['time'])
    weather_dict = {(row['time'].year, row['time'].month, row['time'].day, row['time'].hour): 
                    (row['precip'], row['wind'], row['temp']) for _, row in df.iterrows()}
    return weather_dict, (df['precip'].mean(), df['wind'].mean(), df['temp'].mean())

def get_zone_centroids() -> dict:
    print("Loading local NYC Taxi Zones shapefile...")
    shp_files = list((DATA_DIR / "taxi_zones").rglob("*.shp"))
    gdf = gpd.read_file(shp_files[0]).to_crs(epsg=2263)
    centroids = gdf.geometry.centroid.to_crs(epsg=4326)
    zone_centers = {row.LocationID: (centroid.y, centroid.x) for row, centroid in zip(gdf.itertuples(), centroids)}
    zone_centers[0] = zone_centers[264] = zone_centers[265] = (40.7128, -74.0060)
    return zone_centers

def get_zone_clusters(zone_centers: dict) -> dict:
    print("Clustering zones into 100 neighborhoods (per PDF)...")
    coords = np.array([zone_centers[i] for i in range(1, 266) if i in zone_centers])
    kmeans = MiniBatchKMeans(n_clusters=100, random_state=42, n_init=3)
    kmeans.fit(coords)
    return {z: kmeans.predict([c])[0] for z, c in zone_centers.items()}

# --- 2. FEATURE ENGINEERING ---
def haversine_array(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def add_spatial_features(df: pd.DataFrame, zone_centers: dict, hubs: dict, zone_clusters: dict) -> pd.DataFrame:
    df["p_lat"] = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).astype("float32")
    df["p_lon"] = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).astype("float32")
    df["d_lat"] = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).astype("float32")
    df["d_lon"] = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).astype("float32")
    
    df["pickup_cluster"] = df["pickup_zone"].map(lambda x: zone_clusters.get(x, 0)).astype("int32")
    df["dropoff_cluster"] = df["dropoff_zone"].map(lambda x: zone_clusters.get(x, 0)).astype("int32")
    
    # Rotations to help trees split diagonally
    for angle_deg in[30, 45, 60]:
        angle = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        df[f"p_rot{angle_deg}_x"] = (df["p_lon"] * cos_a - df["p_lat"] * sin_a).astype("float32")
        df[f"p_rot{angle_deg}_y"] = (df["p_lon"] * sin_a + df["p_lat"] * cos_a).astype("float32")
        df[f"d_rot{angle_deg}_x"] = (df["d_lon"] * cos_a - df["d_lat"] * sin_a).astype("float32")
        df[f"d_rot{angle_deg}_y"] = (df["d_lon"] * sin_a + df["d_lat"] * cos_a).astype("float32")
    
    df["haversine_dist"] = haversine_array(df["p_lat"], df["p_lon"], df["d_lat"], df["d_lon"]).astype("float32")
    df["manhattan_dist"] = (haversine_array(df["p_lat"], df["p_lon"], df["d_lat"], df["p_lon"]) + 
                            haversine_array(df["d_lat"], df["p_lon"], df["d_lat"], df["d_lon"])).astype("float32")
    
    p_lat_r, p_lon_r, d_lat_r, d_lon_r = map(np.radians, [df["p_lat"], df["p_lon"], df["d_lat"], df["d_lon"]])
    dlon = d_lon_r - p_lon_r
    y = np.sin(dlon) * np.cos(d_lat_r)
    x = np.cos(p_lat_r) * np.sin(d_lat_r) - np.sin(p_lat_r) * np.cos(d_lat_r) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y, x)).astype("float32")
    
    for hub_name, (h_lat, h_lon) in hubs.items():
        df[f"p_dist_{hub_name}"] = haversine_array(df["p_lat"], df["p_lon"], h_lat, h_lon).astype("float32")
        df[f"d_dist_{hub_name}"] = haversine_array(df["d_lat"], df["d_lon"], h_lat, h_lon).astype("float32")
        
    return df

def add_temporal_and_weather(df: pd.DataFrame, holidays_set: set, airport_zones: set, weather_dict: dict, weather_avgs: tuple) -> pd.DataFrame:
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
    
    keys = list(zip(ts.dt.year, ts.dt.month, ts.dt.day, ts.dt.hour))
    weather_vals =[weather_dict.get(k, weather_avgs) for k in keys]
    df["precip"] = np.array([w[0] for w in weather_vals], dtype="float32")
    df["wind"] = np.array([w[1] for w in weather_vals], dtype="float32")
    df["temp"] = np.array([w[2] for w in weather_vals], dtype="float32")
    
    return df

# --- 3. OUT-OF-FOLD TARGET ENCODING ---
def get_encoding_stats(df: pd.DataFrame, smoothing: int = 20):
    global_mean = df["duration_seconds"].mean()
    global_speed = df["speed_kmh"].mean()
    
    route_agg = df.groupby("route_key")[["duration_seconds", "speed_kmh"]].agg(["sum", "count"])
    route_stats = {
        k: (
            (row[("duration_seconds", "sum")] + global_mean * smoothing) / (row[("duration_seconds", "count")] + smoothing),
            row[("duration_seconds", "count")],
            (row[("speed_kmh", "sum")] + global_speed * smoothing) / (row[("speed_kmh", "count")] + smoothing)
        ) for k, row in route_agg.iterrows()
    }
    
    c_route_agg = df.groupby("cluster_route_key")["duration_seconds"].agg(["sum", "count"])
    c_route_stats = {k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing) for k, row in c_route_agg.iterrows()}
    
    pu_agg = df.groupby("pickup_zone")["duration_seconds"].agg(["sum", "count"])
    pu_stats = {k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing) for k, row in pu_agg.iterrows()}
    
    do_agg = df.groupby("dropoff_zone")["duration_seconds"].agg(["sum", "count"])
    do_stats = {k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing) for k, row in do_agg.iterrows()}
    
    return route_stats, c_route_stats, pu_stats, do_stats, global_mean, global_speed

def apply_encodings(df: pd.DataFrame, stats: tuple) -> pd.DataFrame:
    route_stats, c_route_stats, pu_stats, do_stats, global_mean, global_speed = stats
    
    r_stats = df["route_key"].map(lambda k: route_stats.get(k, (global_mean, 0, global_speed)))
    df["route_mean"] = r_stats.apply(lambda x: x[0]).astype("float32")
    df["route_count"] = r_stats.apply(lambda x: x[1]).astype("int32")
    df["route_speed_mean"] = r_stats.apply(lambda x: x[2]).astype("float32")
    
    # The Golden Feature: Expected Duration based on historical speed
    df["expected_duration"] = (df["haversine_dist"] / (df["route_speed_mean"].clip(lower=1.0) / 3600)).astype("float32")
    
    df["cluster_route_mean"] = df["cluster_route_key"].map(lambda k: c_route_stats.get(k, global_mean)).astype("float32")
    df["pu_mean"] = df["pickup_zone"].map(lambda k: pu_stats.get(k, global_mean)).astype("float32")
    df["do_mean"] = df["dropoff_zone"].map(lambda k: do_stats.get(k, global_mean)).astype("float32")
    
    return df

def build_oof_encodings(train: pd.DataFrame, dev: pd.DataFrame):
    print("Building Out-Of-Fold Target Encodings & Speed Features...")
    train["route_key"] = train["pickup_zone"].astype(str) + "_" + train["dropoff_zone"].astype(str)
    train["cluster_route_key"] = train["pickup_cluster"].astype(str) + "_" + train["dropoff_cluster"].astype(str)
    train["speed_kmh"] = train["haversine_dist"] / (train["duration_seconds"] / 3600)
    
    dev["route_key"] = dev["pickup_zone"].astype(str) + "_" + dev["dropoff_zone"].astype(str)
    dev["cluster_route_key"] = dev["pickup_cluster"].astype(str) + "_" + dev["dropoff_cluster"].astype(str)
    
    cols_to_fill =["route_mean", "route_count", "route_speed_mean", "expected_duration", "cluster_route_mean", "pu_mean", "do_mean"]
    for col in cols_to_fill:
        train[col] = np.nan
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train):
        tr_fold, val_fold = train.iloc[tr_idx], train.iloc[val_idx]
        stats = get_encoding_stats(tr_fold)
        val_encoded = apply_encodings(val_fold.copy(), stats)
        for col in cols_to_fill:
            train.iloc[val_idx, train.columns.get_loc(col)] = val_encoded[col]
            
    full_stats = get_encoding_stats(train)
    dev = apply_encodings(dev, full_stats)
    return train, dev, full_stats

# --- 4. MAIN PIPELINE ---
def main():
    ny_holidays, hubs, airport_zones = load_local_context()
    weather_dict, weather_avgs = get_local_weather()
    zone_centers = get_zone_centroids()
    zone_clusters = get_zone_clusters(zone_centers)
    
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    
    print("Cleaning training data...")
    valid_duration = (train["duration_seconds"] >= 60) & (train["duration_seconds"] <= 7200)
    train = train[valid_duration].copy()
    
    print("Engineering temporal, spatial, and weather features...")
    train = add_temporal_and_weather(train, ny_holidays, airport_zones, weather_dict, weather_avgs)
    dev = add_temporal_and_weather(dev, ny_holidays, airport_zones, weather_dict, weather_avgs)
    
    traffic_agg = train.groupby(["dow", "hour"]).size().to_dict()
    train["traffic_density"] =[traffic_agg.get(k, 0) for k in zip(train["dow"], train["hour"])]
    dev["traffic_density"] = [traffic_agg.get(k, 0) for k in zip(dev["dow"], dev["hour"])]
    
    train = add_spatial_features(train, zone_centers, hubs, zone_clusters)
    dev = add_spatial_features(dev, zone_centers, hubs, zone_clusters)
    
    train, dev, full_stats = build_oof_encodings(train, dev)
    route_stats, c_route_stats, pu_stats, do_stats, global_mean, global_speed = full_stats
    
    # Sample for Random Forest (RF is memory intensive)
    if len(train) > 1_500_000:
        train = train.sample(n=1_500_000, random_state=42)
        
    X_train, y_train = train[FEATURES], np.log1p(train["duration_seconds"].to_numpy())
    X_dev, y_dev = dev[FEATURES], np.log1p(dev["duration_seconds"].to_numpy())
    y_dev_raw = dev["duration_seconds"].to_numpy()

    print("\nRunning Optuna Hyperparameter Tuning for Random Forest (5 Trials)...")
    sample_idx = np.random.choice(len(X_train), min(200_000, len(X_train)), replace=False)
    X_tune, y_tune = X_train.iloc[sample_idx], y_train[sample_idx]
    
    def objective(trial):
        params = {
            "n_estimators": 50,
            "max_depth": trial.suggest_int("max_depth", 15, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "n_jobs": -1,
            "random_state": 42
        }
        model = RandomForestRegressor(**params)
        # Pass .to_numpy() to prevent feature names from being saved
        model.fit(X_tune.to_numpy(), y_tune)
        return np.mean(np.abs(np.expm1(model.predict(X_dev.to_numpy())) - y_dev_raw))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params
    print(f"Best params found: {best_params}")

    print("\nTraining Final Random Forest Model...")
    best_params.update({"n_estimators": 150, "n_jobs": -1, "random_state": 42})
    final_model = RandomForestRegressor(**best_params)
    
    t0 = time.time()
    # Pass .to_numpy() to prevent feature names from being saved
    final_model.fit(X_train.to_numpy(), y_train)
    print(f"  trained in {time.time() - t0:.0f}s")

    mae = float(np.mean(np.abs(np.expm1(final_model.predict(X_dev.to_numpy())) - y_dev_raw)))
    print(f"\nFinal Dev MAE: {mae:.1f} seconds")

    artifact = {
        "model": final_model,
        "zone_centers": zone_centers,
        "zone_clusters": zone_clusters,
        "route_stats": route_stats,
        "c_route_stats": c_route_stats,
        "pu_stats": pu_stats,
        "do_stats": do_stats,
        "global_mean": global_mean,
        "global_speed": global_speed,
        "traffic_agg": traffic_agg,
        "holidays": list(ny_holidays),
        "hubs": hubs,
        "airport_zones": list(airport_zones),
        "weather_dict": weather_dict,
        "weather_avgs": weather_avgs,
        "features": FEATURES,
        "model_type": "dt_2"
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model artifact to {MODEL_PATH}")

if __name__ == "__main__":
    main()