#!/usr/bin/env python
"""
Deep Cross Network (DCN) Training Script.
Features:
- Geospatial features (Haversine, Manhattan, Bearing) via NYC Shapefiles.
- Cyclical time encoding (sin/cos).
- Log-transformed target for stable gradient descent.
- PyTorch DCN architecture with Early Stopping.

Run: python train/dcn_1.py
"""

import argparse
import io
import pickle
import time
import zipfile
from pathlib import Path

import requests

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "dcn_1.pkl"
MODELS_DIR.mkdir(exist_ok=True)

def get_zone_centroids() -> dict:
    """Downloads NYC Taxi Zones shapefile and computes lat/lon centroids."""
    print("Downloading and processing NYC Taxi Zones shapefile...")
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    r = requests.get(url)
    
    extract_dir = DATA_DIR / "taxi_zones"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(extract_dir)
    
    # Dynamically find the .shp file (handles nested folders in the zip)
    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"Could not find any .shp file in {extract_dir}")
    shp_path = shp_files[0]
    print(f"Found shapefile at: {shp_path}")
    
    # Read shapefile and convert to standard lat/lon (EPSG:4326)
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(epsg=4326)
    
    # Compute centroids
    centroids = gdf.geometry.centroid
    zone_centers = {
        row.LocationID: (centroid.y, centroid.x) # (lat, lon)
        for row, centroid in zip(gdf.itertuples(), centroids)
    }
    # Fallback for unknown zones
    zone_centers[0] = (40.7128, -74.0060) # NYC Center
    zone_centers[264] = (40.7128, -74.0060) # Unknown
    zone_centers[265] = (40.7128, -74.0060) # Unknown
    return zone_centers

def add_spatial_features(df: pd.DataFrame, zone_centers: dict) -> pd.DataFrame:
    """Calculates Haversine, Manhattan, and Bearing."""
    # Map coordinates
    p_lat = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).values
    p_lon = df["pickup_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).values
    d_lat = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[0]).values
    d_lon = df["dropoff_zone"].map(lambda x: zone_centers.get(x, zone_centers[0])[1]).values
    
    # Convert to radians
    p_lat_r, p_lon_r, d_lat_r, d_lon_r = map(np.radians, [p_lat, p_lon, d_lat, d_lon])
    
    # Haversine
    dlon = d_lon_r - p_lon_r
    dlat = d_lat_r - p_lat_r
    a = np.sin(dlat/2.0)**2 + np.cos(p_lat_r) * np.cos(d_lat_r) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine_dist"] = 6371 * c # km
    
    # Manhattan (approximate)
    a_lat = np.sin(dlat/2.0)**2
    c_lat = 2 * np.arcsin(np.sqrt(a_lat))
    a_lon = np.cos(p_lat_r) * np.cos(d_lat_r) * np.sin(dlon/2.0)**2
    c_lon = 2 * np.arcsin(np.sqrt(a_lon))
    df["manhattan_dist"] = 6371 * (c_lat + c_lon)
    
    # Bearing
    y = np.sin(dlon) * np.cos(d_lat_r)
    x = np.cos(p_lat_r) * np.sin(d_lat_r) - np.sin(p_lat_r) * np.cos(d_lat_r) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y, x))
    
    return df

# --- 2. FEATURE ENGINEERING ---
def extract_features(df: pd.DataFrame, zone_centers: dict) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    
    # Cyclical Time
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["minute_sin"] = np.sin(2 * np.pi * ts.dt.minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * ts.dt.minute / 60)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    
    df["passenger_count"] = df["passenger_count"].astype("float32")
    
    # Spatial
    df = add_spatial_features(df, zone_centers)
    return df

# --- 3. PYTORCH DCN MODEL ---
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
        
    def forward(self, x0):
        xl = x0
        for i in range(self.num_layers):
            # x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l
            xl_w = torch.sum(xl * self.weights[i], dim=1, keepdim=True)
            xl = x0 * xl_w + self.biases[i] + xl
        return xl

class DCN(nn.Module):
    def __init__(self, cat_dims, dense_dim, embed_dim=16, cross_layers=3, deep_layers=[128, 64, 32]):
        super().__init__()
        # Embeddings for categorical features (pickup, dropoff)
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims])
        
        total_input_dim = (len(cat_dims) * embed_dim) + dense_dim
        
        self.cross_net = CrossNetwork(total_input_dim, cross_layers)
        
        layers = []
        in_dim = total_input_dim
        for out_dim in deep_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = out_dim
        self.deep_net = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(total_input_dim + deep_layers[-1], 1)
        
    def forward(self, cat_x, dense_x):
        embeds = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embeds + [dense_x], dim=1)
        
        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)
        
        out = torch.cat([cross_out, deep_out], dim=1)
        return self.output_layer(out)

def main():
    parser = argparse.ArgumentParser(description="Train DCN ETA model (dcn_1)")
    parser.add_argument(
        "--train-size",
        type=int,
        default=3_000_000,
        help="Max training rows after filtering (default 3_000_000; 0 = use all)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    zone_centers = get_zone_centroids()
    
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    
    # Filter anomalies
    valid_duration = (train["duration_seconds"] >= 60) & (train["duration_seconds"] <= 7200)
    train = train[valid_duration].copy()
    
    cap = args.train_size
    if cap > 0 and len(train) > cap:
        train = train.sample(n=cap, random_state=42)
        
    print("Extracting features...")
    train = extract_features(train, zone_centers)
    dev = extract_features(dev, zone_centers)
    
    cat_cols = ["pickup_zone", "dropoff_zone"]
    dense_cols = [
        "passenger_count", "haversine_dist", "manhattan_dist", "bearing",
        "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos"
    ]
    
    # Scale dense features
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(train[dense_cols].astype(np.float32))
    dev_dense = scaler.transform(dev[dense_cols].astype(np.float32))
    
    train_cat = train[cat_cols].astype(np.int64).values
    dev_cat = dev[cat_cols].astype(np.int64).values
    
    # Log transform target: log(1 + duration)
    train_y = np.log1p(train["duration_seconds"].values).astype(np.float32).reshape(-1, 1)
    dev_y = np.log1p(dev["duration_seconds"].values).astype(np.float32).reshape(-1, 1)
    
    # DataLoaders
    batch_size = 4096
    train_loader = DataLoader(TensorDataset(torch.tensor(train_cat), torch.tensor(train_dense), torch.tensor(train_y)), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(torch.tensor(dev_cat), torch.tensor(dev_dense), torch.tensor(dev_y)), 
                            batch_size=batch_size)
    
    # Initialize Model
    cat_dims = [266, 266] # Max zone ID is 265
    model = DCN(cat_dims=cat_dims, dense_dim=len(dense_cols)).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    print("\nTraining DCN...")
    epochs = 10
    best_mae = float('inf')
    patience = 2
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = 0
        for cat_x, dense_x, y in train_loader:
            cat_x, dense_x, y = cat_x.to(device), dense_x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(cat_x, dense_x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_truth = []
        with torch.no_grad():
            for cat_x, dense_x, y in dev_loader:
                cat_x, dense_x = cat_x.to(device), dense_x.to(device)
                preds = model(cat_x, dense_x)
                val_preds.append(preds.cpu().numpy())
                val_truth.append(y.numpy())
                
        val_preds = np.vstack(val_preds)
        val_truth = np.vstack(val_truth)
        
        # Inverse log transform to calculate actual MAE in seconds
        actual_preds = np.expm1(val_preds)
        actual_truth = np.expm1(val_truth)
        mae = np.mean(np.abs(actual_preds - actual_truth))
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {time.time()-t0:.1f}s | Train Loss: {train_loss/len(train_loader):.4f} | Dev MAE: {mae:.1f}s")
        
        if mae < best_mae:
            best_mae = mae
            patience_counter = 0
            # Save best weights
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    # Package Artifact
    artifact = {
        "model_state": best_state,
        "zone_centers": zone_centers,
        "scaler": scaler,
        "cat_dims": cat_dims,
        "dense_dim": len(dense_cols),
        "model_type": "dcn"
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved DCN artifact to {MODEL_PATH}")

if __name__ == "__main__":
    main()