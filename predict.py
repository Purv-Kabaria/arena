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
    return (REPO_ROOT / "models" / "baseline.pkl").resolve()


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


def _load_predict_fn() -> Callable[[dict], float]:
    path = _resolved_model_path()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and obj.get("model_type") == "dcn":
        return _build_dcn_predict_fn(obj)
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
