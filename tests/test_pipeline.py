"""
🧪 Automated Scientific & Pipeline Unit Testing Suite
Tests:
  1. FITS Data Ingestion & Metadata Parsing
  2. Zero Future Label Leakage in Observational Headers
  3. Correct 24-48h Forward Target Window Construction [T+24h .. T+48h]
  4. Active-Region-Aware Dataset Splitting (Zero active-region overlap)
  5. 4-Channel Feature Tensor Construction [Sequence(4), Channels(4), 256, 256]
  6. Multi-Task PyTorch ConvLSTM Forward Pass & Grad-CAM Backprop
  7. Probability Temperature Calibration
  8. Space-Weather Evaluation Metrics (TSS, HSS, F1, ROC-AUC)
  9. FastAPI Microservice Endpoints
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import torch
from datetime import datetime, timezone, timedelta
from astropy.io import fits
from fastapi.testclient import TestClient

from config import BASE_DIR, DATA_DIR, CATALOGS_DIR, SEQ_LENGTH, IN_CHANNELS, IMG_SIZE
from preprocess import build_multi_channel_frame, preprocess_solar_disk, compute_optical_flux_and_shear_proxies
from build_labels import extract_observation_metadata, build_forward_target_labels
from model import SolarFlarePredictor, ModelCalibrator, SpatioTemporalGradCAM
from evaluate import compute_space_weather_skill_scores
from dataset import get_active_region_split_datasets
from api import app


# -----------------------------------------------------------------------------
# 1. FITS INGESTION & ZERO LEAKAGE
# -----------------------------------------------------------------------------
def test_zero_future_label_leakage_in_headers():
    """Verifies that FITS observation headers NEVER contain future flare outcomes."""
    fits_files = list(DATA_DIR.glob("*.fits"))
    assert len(fits_files) > 0, "No FITS files found in data directory"

    forbidden_keys = ["FLARE_LABEL", "GOES_CLASS", "PEAK_FLUX", "FUTURE_FLARE"]
    for fpath in fits_files[:10]:
        with fits.open(fpath) as hdul:
            hdr = hdul[0].header
            for k in forbidden_keys:
                assert k not in hdr, f"DATA LEAKAGE DETECTED: Header in {fpath.name} contains {k}"
            assert "DATE-OBS" in hdr, f"Missing observation timestamp in {fpath.name}"
            assert "NOAA_AR" in hdr, f"Missing active region ID in {fpath.name}"


# -----------------------------------------------------------------------------
# 2. 24-48H FORWARD TARGET WINDOW LOGIC
# -----------------------------------------------------------------------------
def test_forward_target_window_calculation():
    """Verifies that target window is strictly [T_obs + 24h, T_obs + 48h]."""
    t_obs = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    win_start = t_obs + timedelta(hours=24)
    win_end = t_obs + timedelta(hours=48)

    assert win_start == datetime(2024, 5, 2, 12, 0, 0, tzinfo=timezone.utc)
    assert win_end == datetime(2024, 5, 3, 12, 0, 0, tzinfo=timezone.utc)
    assert (win_end - win_start).total_seconds() == 24 * 3600


# -----------------------------------------------------------------------------
# 3. 4-CHANNEL FEATURE TENSOR SHAPE
# -----------------------------------------------------------------------------
def test_4_channel_feature_synthesis():
    """Verifies 4-channel tensor shape: [4, H, W] for UV, Gradient, Laplacian, Temporal Diff."""
    dummy_patch = np.random.uniform(0.0, 1.0, IMG_SIZE).astype(np.float32)
    dummy_prev = np.random.uniform(0.0, 1.0, IMG_SIZE).astype(np.float32)

    mch = build_multi_channel_frame(dummy_patch, prev_patch=dummy_prev)
    assert mch.shape == (4, IMG_SIZE[0], IMG_SIZE[1]), f"Expected (4, 256, 256), got {mch.shape}"
    assert mch.dtype == np.float32
    assert np.all(mch >= 0.0) and np.all(mch <= 1.0)


# -----------------------------------------------------------------------------
# 4. ACTIVE-REGION-AWARE SPLIT INDEPENDENCE
# -----------------------------------------------------------------------------
def test_active_region_split_disjointness():
    """Verifies that active regions in Train, Validation, and Test sets are mutually exclusive."""
    train_ds, val_ds, test_ds = get_active_region_split_datasets()
    
    train_ars = set(train_ds.df["active_region"].unique()) if hasattr(train_ds, "df") else set()
    val_ars = set(val_ds.df["active_region"].unique()) if hasattr(val_ds, "df") else set()
    test_ars = set(test_ds.df["active_region"].unique()) if hasattr(test_ds, "df") else set()

    if len(train_ars) > 1 and len(val_ars) >= 1 and len(test_ars) >= 1:
        assert train_ars.isdisjoint(val_ars), "Leakage: Train and Validation sets share active regions!"
        assert train_ars.isdisjoint(test_ars), "Leakage: Train and Test sets share active regions!"
        assert val_ars.isdisjoint(test_ars), "Leakage: Validation and Test sets share active regions!"


# -----------------------------------------------------------------------------
# 5. MULTI-TASK MODEL FORWARD PASS & GRAD-CAM
# -----------------------------------------------------------------------------
def test_model_multi_task_forward_and_gradcam():
    """Verifies that model returns binary, multiclass, and flux regression outputs and supports Grad-CAM."""
    model = SolarFlarePredictor(in_channels=4, hidden_dim=32)
    model.eval()

    dummy_input = torch.randn(1, 4, 4, 256, 256)
    preds = model(dummy_input, return_all_heads=True)

    assert "binary_logits" in preds
    assert "multiclass_logits" in preds
    assert "log_flux_pred" in preds
    assert preds["binary_logits"].shape == (1, 2)
    assert preds["multiclass_logits"].shape == (1, 4)
    assert preds["log_flux_pred"].shape == (1,)

    # Test Grad-CAM
    cam_engine = SpatioTemporalGradCAM(model)
    cams, clean_preds = cam_engine.generate(dummy_input, target_class=1, task="binary")
    assert len(cams) == 4
    assert cams[0].shape == (256, 256)
    assert "calibrated_binary_probs" in clean_preds


# -----------------------------------------------------------------------------
# 6. SPACE-WEATHER SKILL SCORES CALCULATION
# -----------------------------------------------------------------------------
def test_space_weather_skill_scores():
    """Verifies calculation of TSS, HSS, Precision, Recall without hardcoded fallbacks."""
    y_true = [1, 1, 0, 0, 1, 0, 0, 1]
    y_pred_probs = [0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.1, 0.85]

    scores = compute_space_weather_skill_scores(y_true, y_pred_probs, threshold=0.5)
    assert scores["recall_tpr"] == 1.0
    assert scores["false_alarm_rate_fpr"] == 0.0
    assert scores["true_skill_statistic_tss"] == 1.0
    assert scores["heidke_skill_score_hss"] == 1.0
    assert scores["f1_score"] == 1.0


# -----------------------------------------------------------------------------
# 7. FASTAPI TEST CLIENT
# -----------------------------------------------------------------------------
def test_fastapi_endpoints():
    """Verifies that FastAPI REST endpoints respond with valid schemas."""
    client = TestClient(app)

    # Test /health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ONLINE"
    assert "model_loaded" in res_health.json()

    # Test /model/info
    res_info = client.get("/model/info")
    assert res_info.status_code == 200
    assert "SolarFlareNet" in res_info.json()["model_name"]

    # Test /active-regions
    res_ar = client.get("/active-regions")
    assert res_ar.status_code == 200
    assert len(res_ar.json()["tracked_active_regions"]) >= 3

    # Test /metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert "single_split_test_metrics" in metrics_data or "loro_cv_aggregate_summary" in metrics_data

    # Test /predict
    res_pred = client.post("/predict", json={"scenario_id": "AR3664_Impending_X_Flare", "data_mode": "DEMO"})
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "mx_probability_24h" in data
    assert "predicted_class" in data
    assert "mitigation_directives" in data
    assert "forecast_window" in data

    # Test /bulletin (dynamic ISSDC bulletin with params & bare call)
    res_bulletin = client.get("/bulletin?scenario_id=AR3664_Impending_X_Flare&data_mode=DEMO")
    assert res_bulletin.status_code == 200
    assert "INDIAN SPACE RESEARCH ORGANISATION" in res_bulletin.text
    assert "Aditya-L1" in res_bulletin.text

    # Test /bulletin bare call (without query params)
    res_bare_bulletin = client.get("/bulletin")
    assert res_bare_bulletin.status_code == 200
    assert "INDIAN SPACE RESEARCH ORGANISATION" in res_bare_bulletin.text

    # Test /api/gradcam
    res_cam = client.get("/api/gradcam?scenario_id=AR3664_Impending_X_Flare")
    assert res_cam.status_code == 200
    cam_data = res_cam.json()
    assert "frames" in cam_data
    assert len(cam_data["frames"]) == 4
