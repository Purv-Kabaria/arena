#!/usr/bin/env python
"""
Optimized Decision Tree Training Script (dt_1).
Features:
- Local Weather Data Integration (Open-Meteo).
- Coordinate Clustering (MiniBatchKMeans) for neighborhood grouping.
- Geospatial features (Haversine, Manhattan, Bearing, Lat/Lon).
- Target Encoding (Route and Cluster-Route).
- Hyperparameter tuning for DecisionTreeRegressor.

Run: python train/dt_1.py
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import RandomizedSearchCV

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "dt_1.pkl"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES =[
    "passenger_count", "hour", "minute", "dow", "month", "is_weekend",
    "p_lat", "p_lon", "d_lat", "d_lon",
    "haversine_dist", "manhattan_dist", "bearing",
    "temp", "precip", "wind",
    "route_mean", "route_count", "cluster_route_mean"
]

def get_local_weather():
    print("Loading local weather data...")
    df = pd.read_csv(DATA_DIR / "weather_2023_2024.csv")
    df['time'] = pd.to_datetime(df['time'])
    
    weather_dict = {}
    for _, row in df.iterrows():
        dt = row['time']
        weather_dict[(dt.year, dt.month, dt.day, dt.hour)] = (row['precip'], row['wind'], row['temp'])
        
    avg_precip = df['precip'].mean()
    avg_wind = df['wind'].mean()
    avg_temp = df['temp'].mean()
    
    return weather_dict, (avg_precip, avg_wind, avg_temp)

def add_weather_features(df: pd.DataFrame, weather_dict: dict, weather_avgs: tuple) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    keys = list(zip(ts.dt.year, ts.dt.month, ts.dt.day, ts.dt.hour))
    
    weather_vals =[weather_dict.get(k, weather_avgs) for k in keys]
    df["precip"] = [w[0] for w in weather_vals]
    df["wind"] = [w[1] for w in weather_vals]
    df["temp"] = [w[2] for w in weather_vals]
    
    df["precip"] = df["precip"].astype("float32")
    df["wind"] = df["wind"].astype("float32")
    df["temp"] = df["temp"].astype("float32")
    return df

def get_zone_centroids() -> dict:
    print("Loading local NYC Taxi Zones shapefile...")
    shp_files = list((DATA_DIR / "taxi_zones").rglob("*.shp"))
    gdf = gpd.read_file(shp_files[0]).to_crs(epsg=2263)
    centroids = gdf.geometry.centroid.to_crs(epsg=4326)
    
    zone_centers = {row.LocationID: (centroid.y, centroid.x) for row, centroid in zip(gdf.itertuples(), centroids)}
    zone_centers[0] = zone_centers[264] = zone_centers[265] = (40.7128, -74.0060)
    return zone_centers

def get_zone_clusters(zone_centers: dict) -> dict:
    print("Clustering zones into neighborhoods...")
    coords = np.array([zone_centers[i] for i in range(1, 266) if i in zone_centers])
    kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, n_init=3)
    kmeans.fit(coords)
    zone_clusters = {z: kmeans.predict([c])[0] for z, c in zone_centers.items()}
    return zone_clusters

def add_spatial_features(df: pd.DataFrame, zone_centers: dict, zone_clusters: dict) -> pd.DataFrame:
    df["p_lat"] = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).astype("float32")
    df["p_lon"] = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).astype("float32")
    df["d_lat"] = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).astype("float32")
    df["d_lon"] = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).astype("float32")
    
    df["pickup_cluster"] = df["pickup_zone"].map(lambda x: zone_clusters.get(x, 0)).astype("int32")
    df["dropoff_cluster"] = df["dropoff_zone"].map(lambda x: zone_clusters.get(x, 0)).astype("int32")
    
    p_lat_r, p_lon_r, d_lat_r, d_lon_r = map(np.radians,[df["p_lat"], df["p_lon"], df["d_lat"], df["d_lon"]])
    
    dlon = d_lon_r - p_lon_r
    dlat = d_lat_r - p_lat_r
    a = np.sin(dlat/2.0)**2 + np.cos(p_lat_r) * np.cos(d_lat_r) * np.sin(dlon/2.0)**2
    df["haversine_dist"] = (6371 * 2 * np.arcsin(np.sqrt(a))).astype("float32")
    
    a_lat = np.sin(dlat/2.0)**2
    a_lon = np.cos(p_lat_r) * np.cos(d_lat_r) * np.sin(dlon/2.0)**2
    df["manhattan_dist"] = (6371 * (2 * np.arcsin(np.sqrt(a_lat)) + 2 * np.arcsin(np.sqrt(a_lon)))).astype("float32")
    
    y = np.sin(dlon) * np.cos(d_lat_r)
    x = np.cos(p_lat_r) * np.sin(d_lat_r) - np.sin(p_lat_r) * np.cos(d_lat_r) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y, x)).astype("float32")
    
    return df

def build_target_encodings(train_df: pd.DataFrame, smoothing: int = 20):
    global_mean = train_df["duration_seconds"].mean()
    
    train_df["route_key"] = train_df["pickup_zone"].astype(str) + "_" + train_df["dropoff_zone"].astype(str)
    route_agg = train_df.groupby("route_key")["duration_seconds"].agg(["sum", "count"])
    route_stats = {k: ((row["sum"] + global_mean * smoothing) / (row["count"] + smoothing), row["count"]) for k, row in route_agg.iterrows()}
    
    train_df["cluster_route_key"] = train_df["pickup_cluster"].astype(str) + "_" + train_df["dropoff_cluster"].astype(str)
    c_route_agg = train_df.groupby("cluster_route_key")["duration_seconds"].agg(["sum", "count"])
    c_route_stats = {k: (row["sum"] + global_mean * smoothing) / (row["count"] + smoothing) for k, row in c_route_agg.iterrows()}
    
    return route_stats, c_route_stats, global_mean

def apply_target_encodings(df: pd.DataFrame, route_stats: dict, c_route_stats: dict, global_mean: float) -> pd.DataFrame:
    route_keys = df["pickup_zone"].astype(str) + "_" + df["dropoff_zone"].astype(str)
    r_stats = route_keys.map(lambda k: route_stats.get(k, (global_mean, 0)))
    df["route_mean"] = r_stats.apply(lambda x: x[0]).astype("float32")
    df["route_count"] = r_stats.apply(lambda x: x[1]).astype("int32")
    
    c_route_keys = df["pickup_cluster"].astype(str) + "_" + df["dropoff_cluster"].astype(str)
    df["cluster_route_mean"] = c_keys = c_route_keys.map(lambda k: c_route_stats.get(k, global_mean)).astype("float32")
    
    return df

def main():
    weather_dict, weather_avgs = get_local_weather()
    zone_centers = get_zone_centroids()
    zone_clusters = get_zone_clusters(zone_centers)
    
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    
    print("Cleaning training data...")
    valid_duration = (train["duration_seconds"] >= 60) & (train["duration_seconds"] <= 10800)
    train = train[valid_duration].copy()
    
    if len(train) > 2_000_000:
        train = train.sample(n=2_000_000, random_state=42)
    
    print("Engineering features...")
    for df in[train, dev]:
        ts = pd.to_datetime(df["requested_at"])
        df["hour"] = ts.dt.hour.astype("int8")
        df["minute"] = ts.dt.minute.astype("int8")
        df["dow"] = ts.dt.dayofweek.astype("int8")
        df["month"] = ts.dt.month.astype("int8")
        df["is_weekend"] = (df["dow"] >= 5).astype("int8")
        df["passenger_count"] = df["passenger_count"].astype("float32")
        
    train = add_weather_features(train, weather_dict, weather_avgs)
    dev = add_weather_features(dev, weather_dict, weather_avgs)
    
    train = add_spatial_features(train, zone_centers, zone_clusters)
    dev = add_spatial_features(dev, zone_centers, zone_clusters)
    
    print("Building target encodings...")
    route_stats, c_route_stats, global_mean = build_target_encodings(train)
    train = apply_target_encodings(train, route_stats, c_route_stats, global_mean)
    dev = apply_target_encodings(dev, route_stats, c_route_stats, global_mean)
    
    X_train = train[FEATURES]
    y_train = np.log1p(train["duration_seconds"].to_numpy())
    
    X_dev = dev[FEATURES]
    y_dev_raw = dev["duration_seconds"].to_numpy()

    print("\nHyperparameter Tuning Decision Tree...")
    param_dist = {
        "max_depth":[10, 15, 20, 25],
        "min_samples_leaf":[50, 100, 200, 500],
        "min_samples_split":[100, 200, 500, 1000]
    }
    
    tune_idx = np.random.choice(len(X_train), min(200_000, len(X_train)), replace=False)
    X_tune, y_tune = X_train.iloc[tune_idx], y_train[tune_idx]
    
    dt = DecisionTreeRegressor(random_state=42)
    search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=10, 
                                scoring='neg_mean_absolute_error', cv=3, verbose=1, random_state=42, n_jobs=-1)
    search.fit(X_tune, y_tune)
    best_params = search.best_params_
    print(f"Best params found: {best_params}")

    print("\nTraining Final Decision Tree Model...")
    final_model = DecisionTreeRegressor(**best_params, random_state=42)
    
    t0 = time.time()
    final_model.fit(X_train, y_train)
    print(f"  trained in {time.time() - t0:.0f}s")

    preds_log = final_model.predict(X_dev)
    preds_raw = np.expm1(preds_log)
    
    mae = float(np.mean(np.abs(preds_raw - y_dev_raw)))
    print(f"\nFinal Dev MAE: {mae:.1f} seconds")

    artifact = {
        "model": final_model,
        "zone_centers": zone_centers,
        "zone_clusters": zone_clusters,
        "weather_dict": weather_dict,
        "weather_avgs": weather_avgs,
        "route_stats": route_stats,
        "c_route_stats": c_route_stats,
        "global_mean": global_mean,
        "features": FEATURES,
        "model_type": "dt_1"
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model artifact to {MODEL_PATH}")

if __name__ == "__main__":
    main()