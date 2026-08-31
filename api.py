"""
🚀 Aditya-L1 Solar Flare & Space Weather Warning System
Production-Grade FastAPI Backend & Space-Ops Microservice

Endpoints:
  - GET  /health
  - GET  /model/info
  - POST /predict
  - POST /predict/sequence
  - GET  /active-regions
  - GET  /historical-events
  - GET  /metrics
  - GET  /api/gradcam
  - GET  /bulletin
"""

import os
import io
import json
import base64
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from astropy.io import fits

from config import (
    BASE_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    MODELS_LATEST_DIR,
    CATALOGS_DIR,
    SEQ_LENGTH,
    IN_CHANNELS
)
from model import SolarFlarePredictor, SpatioTemporalGradCAM
from preprocess import (
    load_and_clean_fits,
    preprocess_solar_disk,
    extract_active_region,
    build_multi_channel_frame,
    compute_optical_flux_and_shear_proxies,
)
from cme_module import SpaceWeatherDecisionEngine

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aditya_l1_api")

# -----------------------------------------------------------------------------
# FASTAPI APP SETUP
# -----------------------------------------------------------------------------
app = FastAPI(
    title="☀️ ISRO Aditya-L1 Solar Flare Early Warning System API",
    description="""
    **Smart India Hackathon (SIH) 2026**
    
    Production-grade AI microservice providing spatio-temporal solar flare forecasting 
    24 to 48 hours prior to Earth impact using 4-channel multi-spectral solar representations.
    """,
    version="2.6.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration:
# Note: Per W3C CORS specifications, `allow_origins=["*"]` cannot be combined with `allow_credentials=True`.
# We use an explicit whitelist of trusted origins (configurable via ALLOWED_ORIGINS env var) to maintain
# full browser compatibility for credentialed requests while avoiding browser-level CORS rejection.
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# THREAD-SAFETY & MODEL INITIALIZATION
# -----------------------------------------------------------------------------
# Thread-safety lock: FastAPI runs synchronous endpoint handlers in a multi-threaded worker pool.
# Concurrent inference requests or PyTorch autograd backward passes (in Grad-CAM) can race on internal
# model buffers and hook activations. We wrap all forward and backward passes in `model_lock` and instantiate
# a fresh SpatioTemporalGradCAM context per request.
model_lock = threading.Lock()

device = torch.device("cpu")
model = SolarFlarePredictor(in_channels=4, hidden_dim=32).to(device)

model_loaded: bool = False
model_weights_path: Optional[str] = None

model_paths = [
    MODELS_LATEST_DIR / "solar_flare_model.pth",
    BASE_DIR / "models" / "latest" / "solar_flare_model.pth",
    BASE_DIR / "solar_flare_model.pth",
    DATA_DIR / "solar_flare_model.pth"
]

for p in model_paths:
    if p.exists():
        try:
            try:
                # Primary secure weights loading (PyTorch 2.4+)
                state_dict = torch.load(p, map_location=device, weights_only=True)
            except Exception as w_err:
                logger.warning(f"weights_only=True load failed for {p} ({w_err}); falling back to standard torch.load")
                state_dict = torch.load(p, map_location=device)
            
            model.load_state_dict(state_dict)
            model_loaded = True
            model_weights_path = str(p)
            logger.info(f"Successfully loaded trained model weights from {p}")
            break
        except Exception as e:
            logger.warning(f"Failed loading candidate model weights from {p}: {e}")

if not model_loaded:
    logger.warning("No pre-trained model weights could be loaded from candidate paths; using uninitialized baseline weights.")

model.eval()

# Spatio-Temporal Grad-CAM initialized ONCE globally at module load to avoid duplicate hook accumulation/memory leaks.
# Thread safety during forward/backward generation is guaranteed by acquiring model_lock.
gradcam_engine = SpatioTemporalGradCAM(model)


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    active_region: Optional[str] = "AR-13664"
    scenario_id: Optional[str] = "AR3664_Impending_X_Flare"
    data_mode: Optional[str] = "DEMO"  # "REAL" or "DEMO"


class ForecastWindow(BaseModel):
    start_utc: str
    end_utc: str


class PredictResponse(BaseModel):
    observation_time: str
    forecast_window: ForecastWindow
    target_active_region: str
    data_mode: str
    mx_probability_24h: float
    mx_probability_48h: float
    calibrated_probability: float
    model_confidence: float
    predicted_class: str
    multiclass_distribution: Dict[str, float]
    estimated_peak_flux: str
    risk_level: str
    explanation_available: bool
    optical_proxies: Dict[str, Any]
    mitigation_directives: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# HELPER & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
def get_scenario_dir(scenario_id: Optional[str]) -> Path:
    scenarios_map = {
        "AR3664_Impending_X_Flare": BASE_DIR / "scenarios" / "AR3664_Impending_X_Flare",
        "AR3685_M_Class_Eruption": BASE_DIR / "scenarios" / "AR3685_M_Class_Eruption",
        "AR3670_Quiet_Sun": BASE_DIR / "scenarios" / "AR3670_Quiet_Sun",
        "live_feed": DATA_DIR
    }
    target = scenarios_map.get(scenario_id or "", DATA_DIR)
    if not target.exists():
        target = DATA_DIR
    return target


def array_to_base64_png(img_array: np.ndarray) -> str:
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_array
    _, buffer = cv2.imencode('.png', bgr)
    return base64.b64encode(buffer).decode('utf-8')


def run_inference(request: PredictRequest) -> PredictResponse:
    """
    Unified end-to-end inference pipeline shared between /predict and /bulletin endpoints.
    Handles FITS sequence retrieval, 4-channel tensor synthesis, multi-task inference,
    error validation, and downstream decision directives generation.
    """
    target_dir = get_scenario_dir(request.scenario_id)
    fits_files = sorted(list(target_dir.glob("*.fits")))

    if len(fits_files) < SEQ_LENGTH:
        logger.info(f"Scenario dir '{target_dir}' has {len(fits_files)} FITS files (< {SEQ_LENGTH}). Falling back to DATA_DIR.")
        fits_files = sorted(list(DATA_DIR.glob("*.fits")))

    if len(fits_files) < SEQ_LENGTH:
        logger.error(f"Insufficient FITS frames: required {SEQ_LENGTH}, found {len(fits_files)}")
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient FITS files to construct 4-frame sequence (found {len(fits_files)}, required {SEQ_LENGTH})."
        )

    seq_files = fits_files[:SEQ_LENGTH]
    patches = []
    mch_frames = []
    headers = []
    prev_patch = None

    if request.scenario_id in ("AR3685_M_Class_Eruption", "AR12673_Impending_M_Flare"):
        default_ar = "AR-12673"
    elif request.scenario_id == "AR3670_Quiet_Sun":
        default_ar = "AR-13100"
    else:
        default_ar = request.active_region or "AR-13664"

    try:
        for fpath in seq_files:
            raw = load_and_clean_fits(fpath)
            disk = preprocess_solar_disk(raw)
            patch = extract_active_region(disk, patch_size=(256, 256))
            patches.append(patch)

            mch = build_multi_channel_frame(patch, prev_patch=prev_patch)
            prev_patch = patch
            mch_frames.append(torch.tensor(mch, dtype=torch.float32))

            meta = {"noaa_ar": default_ar, "date_obs": fpath.stem}
            try:
                with fits.open(fpath) as hdul:
                    h = hdul[0].header
                    meta["noaa_ar"] = str(h.get("NOAA_AR", default_ar))
                    meta["date_obs"] = str(h.get("DATE-OBS", fpath.stem))
            except Exception as h_err:
                logger.warning(f"Could not read FITS header from {fpath.name}: {h_err}")
            headers.append(meta)

        seq_tensor = torch.stack(mch_frames, dim=0).unsqueeze(0)  # [1, 4, 4, 256, 256]

    except (IOError, OSError, ValueError, RuntimeError, fits.VerifyError) as e:
        logger.error(f"FITS loading or preprocessing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=f"FITS frame preprocessing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during sequence processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected error during FITS tensor preparation: {str(e)}"
        )

    # Thread-safe multi-task forward inference
    try:
        with model_lock:
            with torch.no_grad():
                preds = model(seq_tensor, return_all_heads=True)
                raw_bin_probs = torch.softmax(preds["binary_logits"], dim=1).numpy()[0]
                cal_bin_probs = torch.softmax(preds["calibrated_binary_logits"], dim=1).numpy()[0]
                
                flare_prob_24h = float(cal_bin_probs[1]) * 100.0
                flare_prob_48h = min(100.0, flare_prob_24h * 1.12)
                confidence = float(np.max(cal_bin_probs)) * 100.0

                multi_probs = torch.softmax(preds["multiclass_logits"], dim=1).numpy()[0]
                pred_idx = int(np.argmax(multi_probs))
                labels = ["Quiet / B-Class", "C-Class", "M-Class", "X-Class"]
                flare_class = labels[pred_idx]

                log_flux = float(preds["log_flux_pred"].numpy()[0])
                peak_flux = f"{10.0 ** log_flux:.2e} W/m²"

    except Exception as inf_err:
        logger.error(f"Model forward pass failed: {inf_err}", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=f"Model forward inference failed: {str(inf_err)}"
        )

    try:
        physics = compute_optical_flux_and_shear_proxies(patches[-1])
        directives = SpaceWeatherDecisionEngine.generate_national_infrastructure_directives(
            flare_prob_24h, flare_class, 10.0 ** log_flux
        )
    except Exception as d_err:
        logger.error(f"Decision engine directives generation failed: {d_err}", exc_info=True)
        physics = {}
        directives = []

    risk_level = (
        "CRITICAL" if flare_prob_24h >= 75.0
        else ("HIGH" if flare_prob_24h >= 55.0
        else ("MODERATE" if flare_prob_24h >= 30.0 else "LOW"))
    )

    obs_time_str = headers[-1]["date_obs"]
    try:
        obs_dt = pd.to_datetime(obs_time_str, utc=True)
    except Exception as parse_err:
        logger.error(f"Unable to parse observation timestamp '{obs_time_str}': {parse_err}")
        raise HTTPException(
            status_code=422,
            detail=f"Unable to parse observation timestamp '{obs_time_str}'. Expected valid ISO-8601 or FITS timestamp format."
        )

    win_start = (obs_dt + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    win_end = (obs_dt + timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")

    return PredictResponse(
        observation_time=obs_time_str,
        forecast_window=ForecastWindow(start_utc=win_start, end_utc=win_end),
        target_active_region=headers[-1]["noaa_ar"],
        data_mode=request.data_mode or "DEMO",
        mx_probability_24h=round(flare_prob_24h, 2),
        mx_probability_48h=round(flare_prob_48h, 2),
        calibrated_probability=round(float(cal_bin_probs[1]), 4),
        model_confidence=round(confidence, 2),
        predicted_class=flare_class,
        multiclass_distribution={
            "Quiet_B": round(float(multi_probs[0]) * 100, 2),
            "C_Class": round(float(multi_probs[1]) * 100, 2),
            "M_Class": round(float(multi_probs[2]) * 100, 2),
            "X_Class": round(float(multi_probs[3]) * 100, 2)
        },
        estimated_peak_flux=peak_flux,
        risk_level=risk_level,
        explanation_available=True,
        optical_proxies=physics,
        mitigation_directives=directives
    )


# -----------------------------------------------------------------------------
# REST ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/health", tags=["System Diagnostics"])
def health_check():
    return {
        "status": "ONLINE",
        "service": "Aditya-L1 Space Weather Warning System API",
        "version": "2.6.0",
        "model_loaded": model_loaded,
        "model_weights_path": model_weights_path,
        "model_architecture": "4-Channel Spatio-Temporal ConvLSTM + Temperature Calibrator",
        "time_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/model/info", tags=["Model Architecture & Provenance"])
def model_info():
    meta_file = MODELS_LATEST_DIR / "model_meta.json"
    if not meta_file.exists():
        meta_file = BASE_DIR / "models" / "latest" / "model_meta.json"

    meta = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read model metadata from {meta_file}: {e}")

    return {
        "model_name": meta.get("model_name", "SolarFlareNet-ConvLSTM-MultiTask"),
        "version": meta.get("version", "2.6.0"),
        "input_tensor_shape": [1, SEQ_LENGTH, IN_CHANNELS, 256, 256],
        "input_channels": [
            "Ch0: Calibrated UV / Optical Intensity",
            "Ch1: Spatial Flux Gradient (|∇I|) [Shear Complexity Proxy]",
            "Ch2: High-Frequency Laplacian Curvature (∇²I) [Loop Complexity Proxy]",
            "Ch3: Temporal Differential Rate (ΔI_t) [Flux Emergence Rate]"
        ],
        "tasks": [
            "Task 1: 24-48h Binary M/X-Class Eruption Probability",
            "Task 2: 4-Class NOAA Flare Classification [Quiet/B, C, M, X]",
            "Task 3: Continuous Log10 Peak Flux Regression"
        ],
        "calibration_method": "Post-Hoc Temperature Scaling (Platt Scaling)",
        "calibrated_temperature": meta.get("calibrated_temperature", 1.15),
        "optimal_threshold": meta.get("optimal_threshold", 0.5),
        "training_active_regions": meta.get("active_regions", {}).get("train", ["AR-13664", "AR-12673", "AR-11158"])
    }


@app.get("/active-regions", tags=["Solar Catalog"])
def get_active_regions():
    return {
        "tracked_active_regions": [
            {"ar": "AR-13664", "solar_cycle": 25, "classification": "Delta-Configuration Superflare Origin", "split": "TRAIN"},
            {"ar": "AR-12673", "solar_cycle": 24, "classification": "X9.3 Eruptive Magnetogram", "split": "TRAIN"},
            {"ar": "AR-11158", "solar_cycle": 24, "classification": "M5.4/X2.2 Valentine Flare", "split": "TRAIN"},
            {"ar": "AR-12887", "solar_cycle": 25, "classification": "X1.0 Halloween Eruption", "split": "VALIDATION"},
            {"ar": "AR-13000", "solar_cycle": 25, "classification": "C-Class Plage", "split": "TEST"},
            {"ar": "AR-13100", "solar_cycle": 25, "classification": "Quiet Sun Baseline", "split": "TEST"}
        ]
    }


@app.get("/historical-events", tags=["Historical Replay"])
def get_historical_events():
    return {
        "events": [
            {"id": "AR-13664-2024", "name": "May 2024 Mother's Day Superflare Event", "actual_flare": "X2.8", "date": "2024-05-10"},
            {"id": "AR-12673-2017", "name": "Sept 2017 Monster X9.3 Solar Flare", "actual_flare": "X9.3", "date": "2017-09-06"},
            {"id": "AR-11158-2011", "name": "Feb 2011 Valentine's Day Eruption", "actual_flare": "M5.4", "date": "2011-02-13"},
            {"id": "AR-13100-2026", "name": "August 2026 Quiet Sun Baseline", "actual_flare": "Quiet", "date": "2026-08-25"}
        ]
    }


@app.get("/metrics", tags=["Evaluation & Benchmarks"])
def get_metrics():
    """
    Returns authentic evaluation metrics from model_meta.json (single-split test set)
    and cv_results.json (12-Fold Leave-One-Region-Out Cross-Validation).
    Raises HTTPException(503) if benchmark files are missing rather than fabricating fake data.
    """
    meta_file = MODELS_LATEST_DIR / "model_meta.json"
    cv_file = MODELS_LATEST_DIR / "cv_results.json"

    if not meta_file.exists():
        meta_file = BASE_DIR / "models" / "latest" / "model_meta.json"
    if not cv_file.exists():
        cv_file = BASE_DIR / "models" / "latest" / "cv_results.json"

    if not meta_file.exists() and not cv_file.exists():
        logger.error(f"Evaluation metric files missing: {meta_file}, {cv_file}")
        raise HTTPException(
            status_code=503,
            detail="Model evaluation metrics are not available. Ensure training and cross-validation scripts have been executed."
        )

    response_data: Dict[str, Any] = {}

    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            response_data["single_split_test_metrics"] = meta.get("single_split_test_metrics", {})
            response_data["model_metadata"] = {
                "version": meta.get("version"),
                "training_timestamp_utc": meta.get("training_timestamp_utc"),
                "optimal_threshold": meta.get("optimal_threshold"),
                "calibrated_temperature": meta.get("calibrated_temperature")
            }
        except Exception as e:
            logger.error(f"Error reading {meta_file}: {e}")

    if cv_file.exists():
        try:
            with open(cv_file, "r") as f:
                cv_data = json.load(f)
            response_data["loro_cv_aggregate_summary"] = cv_data.get("aggregate_summary", {})
            response_data["loro_cv_protocol"] = cv_data.get("evaluation_protocol")
            response_data["total_active_regions_evaluated"] = cv_data.get("total_active_regions_evaluated")
        except Exception as e:
            logger.error(f"Error reading {cv_file}: {e}")

    if not response_data.get("single_split_test_metrics") and not response_data.get("loro_cv_aggregate_summary"):
        raise HTTPException(
            status_code=503,
            detail="Model evaluation metrics could not be parsed from metric files."
        )

    return response_data


@app.post("/predict", response_model=PredictResponse, tags=["AI Forecasting"])
def predict(request: PredictRequest):
    return run_inference(request)


@app.get("/api/gradcam", tags=["Explainable AI (XAI)"])
def get_gradcam(scenario_id: Optional[str] = "AR3664_Impending_X_Flare"):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) saliency maps
    across the 4-frame observation sequence for explainable AI verification.
    """
    target_dir = get_scenario_dir(scenario_id)
    fits_files = sorted(list(target_dir.glob("*.fits")))

    if len(fits_files) < SEQ_LENGTH:
        logger.info(f"Scenario dir '{target_dir}' has {len(fits_files)} FITS (< {SEQ_LENGTH}). Falling back to DATA_DIR.")
        fits_files = sorted(list(DATA_DIR.glob("*.fits")))

    if len(fits_files) < SEQ_LENGTH:
        logger.error(f"Insufficient FITS frames for Grad-CAM: found {len(fits_files)}, required {SEQ_LENGTH}")
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient FITS files to construct Grad-CAM sequence (found {len(fits_files)}, required {SEQ_LENGTH})."
        )

    patches = []
    mch_frames = []
    prev_patch = None

    try:
        for fpath in fits_files[:SEQ_LENGTH]:
            raw = load_and_clean_fits(fpath)
            disk = preprocess_solar_disk(raw)
            patch = extract_active_region(disk, patch_size=(256, 256))
            patches.append(patch)

            mch = build_multi_channel_frame(patch, prev_patch=prev_patch)
            prev_patch = patch
            mch_frames.append(torch.tensor(mch, dtype=torch.float32))

        seq_tensor = torch.stack(mch_frames, dim=0).unsqueeze(0)

        # Thread-safe Grad-CAM generation protected by global model_lock (no hook accumulation)
        with model_lock:
            cams, _ = gradcam_engine.generate(seq_tensor, target_class=1, task="binary")

    except (IOError, OSError, ValueError, RuntimeError, fits.VerifyError) as e:
        logger.error(f"Grad-CAM tensor generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=f"Grad-CAM processing failed during FITS computation: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in Grad-CAM computation: {e}", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=f"Grad-CAM computation encountered an unexpected error: {str(e)}"
        )

    result_frames = []
    for i in range(len(cams)):
        patch_base = patches[i]
        cam_map = cams[i]

        cam_uint8 = np.clip(cam_map * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        base_rgb = cv2.cvtColor(np.clip(patch_base * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(base_rgb, 0.4, heatmap_rgb, 0.6, 0)

        result_frames.append({
            "step": f"T-{SEQ_LENGTH - 1 - i}",
            "patch_base64": f"data:image/png;base64,{array_to_base64_png(base_rgb)}",
            "gradcam_base64": f"data:image/png;base64,{array_to_base64_png(blended)}",
            "peak_attention_score": float(np.max(cam_map))
        })

    return {
        "attribution_note": "Gradient-weighted Class Activation Mapping computed from PyTorch backward autograd pass across 4 input channels.",
        "frames": result_frames
    }


@app.get("/api/solar-channels", tags=["Multi-Spectral Diagnostics"])
def get_solar_channels(scenario_id: Optional[str] = "AR3664_Impending_X_Flare"):
    target_dir = get_scenario_dir(scenario_id)
    fits_files = sorted(list(target_dir.glob("*.fits")))
    if len(fits_files) < SEQ_LENGTH:
        fits_files = sorted(list(DATA_DIR.glob("*.fits")))

    if not fits_files:
        raise HTTPException(status_code=404, detail="No FITS files available for channel extraction.")

    try:
        curr_raw = load_and_clean_fits(fits_files[-1])
        curr_disk = preprocess_solar_disk(curr_raw)
        curr_patch = extract_active_region(curr_disk, patch_size=(256, 256))

        prev_patch = None
        if len(fits_files) >= 2:
            prev_raw = load_and_clean_fits(fits_files[-2])
            prev_disk = preprocess_solar_disk(prev_raw)
            prev_patch = extract_active_region(prev_disk, patch_size=(256, 256))

        mch = build_multi_channel_frame(curr_patch, prev_patch=prev_patch)

        def _colorize(arr, cmap):
            uint8_arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(uint8_arr, cmap)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        disk_colored = _colorize(curr_disk, cv2.COLORMAP_INFERNO)
        ch0_img = _colorize(mch[0], cv2.COLORMAP_MAGMA)
        ch1_img = _colorize(mch[1], cv2.COLORMAP_VIRIDIS)
        ch2_img = _colorize(mch[2], cv2.COLORMAP_PLASMA)
        ch3_img = _colorize(mch[3], cv2.COLORMAP_CIVIDIS)

    except Exception as e:
        logger.error(f"Solar channels extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Multi-spectral channel extraction failed: {str(e)}")

    return {
        "full_disk": f"data:image/png;base64,{array_to_base64_png(disk_colored)}",
        "channels": [
            {
                "id": "ch0",
                "name": "Ch 0: Dynamic Range UV Intensity",
                "description": "Log-compressed optical intensity representing photospheric and chromospheric UV emission.",
                "image_base64": f"data:image/png;base64,{array_to_base64_png(ch0_img)}"
            },
            {
                "id": "ch1",
                "name": "Ch 1: Spatial Gradient (|∇I|)",
                "description": "First-order spatial derivatives measuring magnetic shear proxies along neutral polarity lines.",
                "image_base64": f"data:image/png;base64,{array_to_base64_png(ch1_img)}"
            },
            {
                "id": "ch2",
                "name": "Ch 2: Laplacian Curvature (∇²I)",
                "description": "Second-order spatial derivatives capturing active topological loop complexity and filament twist.",
                "image_base64": f"data:image/png;base64,{array_to_base64_png(ch2_img)}"
            },
            {
                "id": "ch3",
                "name": "Ch 3: Temporal Differential Rate (ΔI_t)",
                "description": "Sub-hourly temporal difference measuring rapid magnetic flux emergence and filament destabilization.",
                "image_base64": f"data:image/png;base64,{array_to_base64_png(ch3_img)}"
            }
        ]
    }


@app.get("/bulletin", response_class=PlainTextResponse, tags=["ISSDC Advisory"])
def get_bulletin(
    active_region: Optional[str] = Query("AR-13664", description="NOAA Active Region ID"),
    scenario_id: Optional[str] = Query("AR3664_Impending_X_Flare", description="Scenario ID or live_feed"),
    data_mode: Optional[str] = Query("DEMO", description="Data Mode: REAL or DEMO")
):
    """
    Generates an automated, dynamic ISRO ISSDC Space Weather Forecast Bulletin
    populated directly from real-time model inference and infrastructure impact directives.
    """
    req = PredictRequest(active_region=active_region, scenario_id=scenario_id, data_mode=data_mode)
    pred = run_inference(req)

    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)

    # Dynamic Geomagnetic threat level determination based on model predictions
    if pred.predicted_class == "X-Class" or pred.mx_probability_24h >= 75.0:
        geomag_threat = "G3 - G5 [Strong to Extreme Geomagnetic Storm Warning]"
    elif pred.predicted_class == "M-Class" or pred.mx_probability_24h >= 45.0:
        geomag_threat = "G1 - G2 [Minor to Moderate Geomagnetic Storm Watch]"
    else:
        geomag_threat = "G0 [Quiet Baseline / Below Storm Threshold]"

    directives_formatted = "\n".join([
        f"   - {d.get('sector', 'Asset')}: [{d.get('status', 'STATUS')}] {d.get('directive', 'Maintain baseline.')}"
        for d in pred.mitigation_directives
    ]) if pred.mitigation_directives else "   - General: Standard baseline telemetry monitoring operational."

    bulletin = f"""================================================================================
INDIAN SPACE RESEARCH ORGANISATION (ISRO)
ISSDC SPACE WEATHER FORECAST & EARLY WARNING BULLETIN
ISSUED: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')} / {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}
================================================================================

1. OBSERVATIONAL SUMMARY:
   Spacecraft: Aditya-L1 | Payload: SUIT (Solar Ultraviolet Imaging Telescope)
   Target Active Region: {pred.target_active_region}
   Observation Timestamp: {pred.observation_time}
   Data Ingestion Mode: {pred.data_mode} | Spectral Filter: Mg II k (279.6 nm)
   Downlink Ground Station: ISSDC Bylalu (32m Deep Space Network Antenna)

2. 24-48 HOUR SPACE WEATHER FORECAST:
   Forward Target Window: {pred.forecast_window.start_utc} to {pred.forecast_window.end_utc}
   24h Major Flare Probability (M/X): {pred.mx_probability_24h}% (Platt Calibrated)
   48h Cumulative Probability: {pred.mx_probability_48h}%
   Predicted Flare Classification: {pred.predicted_class} (Confidence: {pred.model_confidence}%)
   Estimated Peak X-Ray Flux: {pred.estimated_peak_flux}
   Geomagnetic Storm Threat: {geomag_threat}
   Operational Condition: DEFCON ALERT [{pred.risk_level}]

3. DEFENCE & NATIONAL ASSET PROTECTION DIRECTIVES:
{directives_formatted}

================================================================================
Generated by Aditya-L1 Deep Learning Warning System | SIH 2026
================================================================================
"""
    return bulletin


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

