"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

_PREDICT_FN: Callable[[dict], float] | None = None


def _resolved_model_path() -> Path:
    env = os.environ.get("ETA_MODEL_PATH")
    if env:
        p = Path(env)
        return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()
    return (REPO_ROOT / "model.pkl").resolve()


def _predict_xgb(model: Any, request: dict) -> float:
    ts = datetime.fromisoformat(request["requested_at"])
    x = np.array(
        [[
            int(request["pickup_zone"]),
            int(request["dropoff_zone"]),
            ts.hour,
            ts.weekday(),
            ts.month,
            int(request["passenger_count"]),
        ]],
        dtype=np.int32,
    )
    return float(model.predict(x)[0])


def _build_dcn_predict_fn(artifact: dict) -> Callable[[dict], float]:
    import torch
    import pandas as pd
    from train.dcn_1 import DCN, extract_features

    cat_dims = artifact["cat_dims"]
    dense_dim = artifact["dense_dim"]
    net = DCN(cat_dims=cat_dims, dense_dim=dense_dim)
    net.load_state_dict(artifact["model_state"])
    net.eval()

    zone_centers = artifact["zone_centers"]
    scaler = artifact["scaler"]
    cat_cols = ["pickup_zone", "dropoff_zone"]
    dense_cols = [
        "passenger_count", "haversine_dist", "manhattan_dist", "bearing",
        "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
    ]

    def _one(request: dict) -> float:
        df = pd.DataFrame([{
            "pickup_zone": int(request["pickup_zone"]),
            "dropoff_zone": int(request["dropoff_zone"]),
            "requested_at": request["requested_at"],
            "passenger_count": int(request["passenger_count"]),
        }])
        df = extract_features(df, zone_centers)
        dense = scaler.transform(df[dense_cols].astype(np.float32))
        cat = df[cat_cols].astype(np.int64).values
        with torch.no_grad():
            pred = net(
                torch.tensor(cat, dtype=torch.long),
                torch.tensor(dense, dtype=torch.float32),
            )
        return float(torch.expm1(pred).squeeze().detach().cpu().item())

    return _one


def _build_dt_1_predict_fn(artifact: dict) -> Callable[[dict], float]:
    import pandas as pd
    from train.dt_1 import add_spatial_features, add_weather_features, apply_target_encodings

    model = artifact["model"]
    zone_centers = artifact["zone_centers"]
    zone_clusters = artifact["zone_clusters"]
    weather_dict = artifact["weather_dict"]
    weather_avgs = artifact["weather_avgs"]
    route_stats = artifact["route_stats"]
    c_route_stats = artifact["c_route_stats"]
    global_mean = artifact["global_mean"]
    features = artifact["features"]

    def _one(request: dict) -> float:
        df = pd.DataFrame([{
            "pickup_zone": int(request["pickup_zone"]),
            "dropoff_zone": int(request["dropoff_zone"]),
            "requested_at": request["requested_at"],
            "passenger_count": int(request["passenger_count"]),
        }])
        ts = pd.to_datetime(df["requested_at"])
        df["hour"] = ts.dt.hour.astype("int8")
        df["minute"] = ts.dt.minute.astype("int8")
        df["dow"] = ts.dt.dayofweek.astype("int8")
        df["month"] = ts.dt.month.astype("int8")
        df["is_weekend"] = (df["dow"] >= 5).astype("int8")
        df["passenger_count"] = df["passenger_count"].astype("float32")
        df = add_weather_features(df, weather_dict, weather_avgs)
        df = add_spatial_features(df, zone_centers, zone_clusters)
        df = apply_target_encodings(df, route_stats, c_route_stats, global_mean)
        x = df[features]
        pred_log = model.predict(x)
        return float(np.expm1(np.asarray(pred_log, dtype=np.float64).ravel()[0]))

    return _one


def _build_dt_2_predict_fn(artifact: dict) -> Callable[[dict], float]:
    import pandas as pd
    from train.dt_2 import add_temporal_and_weather, add_spatial_features, apply_encodings

    model = artifact["model"]
    zone_centers = artifact["zone_centers"]
    zone_clusters = artifact["zone_clusters"]
    raw_traffic = artifact["traffic_agg"]
    traffic_agg = {(int(k[0]), int(k[1])): int(v) for k, v in raw_traffic.items()}
    weather_dict = artifact["weather_dict"]
    weather_avgs = artifact["weather_avgs"]
    hubs = artifact["hubs"]
    holidays_set = set(artifact["holidays"])
    airport_zones = set(artifact["airport_zones"])
    features = artifact["features"]
    full_stats = (
        artifact["route_stats"],
        artifact["c_route_stats"],
        artifact["pu_stats"],
        artifact["do_stats"],
        artifact["global_mean"],
        artifact["global_speed"],
    )

    def _one(request: dict) -> float:
        df = pd.DataFrame([{
            "pickup_zone": int(request["pickup_zone"]),
            "dropoff_zone": int(request["dropoff_zone"]),
            "requested_at": request["requested_at"],
            "passenger_count": int(request["passenger_count"]),
        }])
        df = add_temporal_and_weather(df, holidays_set, airport_zones, weather_dict, weather_avgs)
        df["traffic_density"] = [
            traffic_agg.get((int(d), int(h)), 0) for d, h in zip(df["dow"], df["hour"])
        ]
        df = add_spatial_features(df, zone_centers, hubs, zone_clusters)
        df["route_key"] = df["pickup_zone"].astype(str) + "_" + df["dropoff_zone"].astype(str)
        df["cluster_route_key"] = (
            df["pickup_cluster"].astype(str) + "_" + df["dropoff_cluster"].astype(str)
        )
        df = apply_encodings(df, full_stats)
        pred_log = model.predict(df[features])
        return float(np.expm1(np.asarray(pred_log, dtype=np.float64).ravel()[0]))

    return _one


def _build_xgb_1_predict_fn(artifact: dict) -> Callable[[dict], float]:
    import pandas as pd
    from train.xgb_1 import add_spatial_features, apply_target_encodings

    model = artifact["model"]
    zone_centers = artifact["zone_centers"]
    route_stats = artifact["route_stats"]
    pu_stats = artifact["pu_stats"]
    do_stats = artifact["do_stats"]
    global_mean = artifact["global_mean"]
    features = artifact["features"]

    def _one(request: dict) -> float:
        df = pd.DataFrame([{
            "pickup_zone": int(request["pickup_zone"]),
            "dropoff_zone": int(request["dropoff_zone"]),
            "requested_at": request["requested_at"],
            "passenger_count": int(request["passenger_count"]),
        }])
        ts = pd.to_datetime(df["requested_at"])
        df["hour"] = ts.dt.hour.astype("int8")
        df["minute"] = ts.dt.minute.astype("int8")
        df["dow"] = ts.dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dow"] >= 5).astype("int8")
        df["month"] = ts.dt.month.astype("int8")
        df["passenger_count"] = df["passenger_count"].astype("int8")
        df = add_spatial_features(df, zone_centers)
        df = apply_target_encodings(df, route_stats, pu_stats, do_stats, global_mean)
        pred_log = model.predict(df[features])
        return float(np.expm1(np.asarray(pred_log, dtype=np.float64).ravel()[0]))

    return _one


def _build_xgb_2_predict_fn(artifact: dict) -> Callable[[dict], float]:
    import pandas as pd
    from train import add_spatial_features, add_temporal_base, apply_encodings

    model = artifact["model"]
    zone_centers = artifact["zone_centers"]
    raw_traffic = artifact["traffic_agg"]
    traffic_agg = {(int(k[0]), int(k[1])): int(v) for k, v in raw_traffic.items()}
    hubs = artifact["hubs"]
    holidays_set = set(artifact["holidays"])
    airport_zones = set(artifact["airport_zones"])
    features = artifact["features"]
    full_stats = (
        artifact["route_stats"],
        artifact["rh_stats"],
        artifact["pu_stats"],
        artifact["do_stats"],
        artifact["global_mean"],
    )

    def _one(request: dict) -> float:
        df = pd.DataFrame([{
            "pickup_zone": int(request["pickup_zone"]),
            "dropoff_zone": int(request["dropoff_zone"]),
            "requested_at": request["requested_at"],
            "passenger_count": int(request["passenger_count"]),
        }])
        df = add_temporal_base(df, holidays_set, airport_zones)
        df["traffic_density"] = [
            traffic_agg.get((int(d), int(h)), 0) for d, h in zip(df["dow"], df["hour"])
        ]
        df["traffic_density"] = df["traffic_density"].astype("float32")
        df = add_spatial_features(df, zone_centers, hubs)
        df["route_key"] = df["pickup_zone"].astype(str) + "_" + df["dropoff_zone"].astype(str)
        df["route_hour_key"] = df["route_key"] + "_" + df["hour"].astype(str)
        df = apply_encodings(df, full_stats)
        pred_log = model.predict(df[features])
        return float(np.expm1(np.asarray(pred_log, dtype=np.float64).ravel()[0]))

    return _one


def _load_predict_fn() -> Callable[[dict], float]:
    path = _resolved_model_path()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and obj.get("model_type") == "dcn":
        return _build_dcn_predict_fn(obj)
    if isinstance(obj, dict) and obj.get("model_type") == "dt_1":
        return _build_dt_1_predict_fn(obj)
    if isinstance(obj, dict) and obj.get("model_type") == "dt_2":
        return _build_dt_2_predict_fn(obj)
    if isinstance(obj, dict) and obj.get("model_type") == "xgb_1":
        return _build_xgb_1_predict_fn(obj)
    if isinstance(obj, dict) and obj.get("model_type") == "xgb_2":
        return _build_xgb_2_predict_fn(obj)
    model = obj
    if hasattr(model, "get_booster"):
        model.get_booster().feature_names = None
    return lambda req: _predict_xgb(model, req)


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    global _PREDICT_FN
    if _PREDICT_FN is None:
        _PREDICT_FN = _load_predict_fn()
    return _PREDICT_FN(request)
