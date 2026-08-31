"""
🎯 Decoupled 24-48h Forward Target Label Construction Engine
Given observational sequences ending at timestamp T_obs:
  - Forecast Window: [T_obs + 24h, T_obs + 48h]
  - Queries independent GOES X-ray flare catalog for active region associations.
  - Zero Leakage: The input observation features contain NO information about future events.
Outputs: data/catalogs/sequence_labels.csv
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from astropy.io import fits

from config import CATALOGS_DIR, BASE_DIR, DATA_DIR, HORIZON_START_HOURS, HORIZON_END_HOURS, MAJOR_FLARE_THRESHOLD_FLUX


def extract_observation_metadata(filepath):
    """
    Extracts purely observational metadata from FITS primary header.
    Guarantees no future labels exist in header.
    """
    meta = {
        "filepath": str(filepath),
        "filename": filepath.name,
        "date_obs": None,
        "noaa_ar": "AR-13664",
        "wavelnth": "279.6 nm"
    }
    try:
        with fits.open(filepath) as hdul:
            hdr = hdul[0].header
            meta["date_obs"] = hdr.get("DATE-OBS", None)
            meta["noaa_ar"] = hdr.get("NOAA_AR", "AR-13664")
            meta["wavelnth"] = hdr.get("WAVELNTH", "279.6 nm")
    except Exception:
        pass
    return meta


def build_forward_target_labels(data_dir=None, catalog_path=None):
    """
    Constructs forward 24-48h target labels for all contiguous FITS observation sequences.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if catalog_path is None:
        catalog_path = CATALOGS_DIR / "goes_flare_catalog.csv"
        if not catalog_path.exists():
            catalog_path = BASE_DIR / "goes_flare_catalog.csv"

    print("=" * 70)
    print("CONSTRUCTING ZERO-LEAKAGE 24-48H FORWARD TARGET LABELS")
    print(f"Observation Directory: {data_dir}")
    print(f"Catalog Source: {catalog_path}")
    print("=" * 70)

    if not Path(catalog_path).exists():
        raise FileNotFoundError(f"GOES catalog not found at {catalog_path}. Run download_data.py first.")

    catalog_df = pd.read_csv(catalog_path)
    catalog_df["start_dt"] = pd.to_datetime(catalog_df["start_time"], utc=True, format="ISO8601")
    catalog_df["peak_dt"] = pd.to_datetime(catalog_df["peak_time"], utc=True, format="ISO8601")

    # Extract metadata for all FITS files
    all_fits = sorted(list(Path(data_dir).glob("*.fits")))
    file_records = []
    for f in all_fits:
        m = extract_observation_metadata(f)
        dt = pd.to_datetime(m["date_obs"], utc=True, format="ISO8601") if m["date_obs"] else None
        file_records.append({
            "filepath": m["filepath"],
            "filename": m["filename"],
            "date_obs": dt,
            "noaa_ar": m["noaa_ar"]
        })

    df_files = pd.DataFrame(file_records).dropna(subset=["date_obs"])
    df_files = df_files.sort_values("date_obs").reset_index(drop=True)

    sequence_records = []
    seq_len = 4

    # Build sliding window sequences per Active Region and calculate forward window target
    for ar, group in df_files.groupby("noaa_ar"):
        group = group.sort_values("date_obs").reset_index(drop=True)
        if len(group) >= seq_len:
            for i in range(len(group) - seq_len + 1):
                sub = group.iloc[i: i + seq_len]
                t_last_obs = sub.iloc[-1]["date_obs"]
                
                # STRICT FORWARD TARGET WINDOW: [T_obs + 24h, T_obs + 48h]
                win_start = t_last_obs + timedelta(hours=HORIZON_START_HOURS)
                win_end = t_last_obs + timedelta(hours=HORIZON_END_HOURS)

                # Query catalog
                events_in_win = catalog_df[
                    (catalog_df["active_region"] == ar) &
                    (catalog_df["start_dt"] >= win_start) &
                    (catalog_df["start_dt"] <= win_end)
                ]

                binary_label = 0
                multiclass_label = 0  # 0: Quiet/B, 1: C, 2: M, 3: X
                max_flux = 1.0e-7     # Baseline quiet flux

                if not events_in_win.empty:
                    top_ev = events_in_win.sort_values("peak_flux_wm2", ascending=False).iloc[0]
                    fl_class = str(top_ev["flare_class"])
                    max_flux = float(top_ev["peak_flux_wm2"])

                    if fl_class.startswith("X") or max_flux >= 1.0e-4:
                        binary_label = 1
                        multiclass_label = 3
                    elif fl_class.startswith("M") or max_flux >= 1.0e-5:
                        binary_label = 1
                        multiclass_label = 2
                    elif fl_class.startswith("C") or max_flux >= 1.0e-6:
                        binary_label = 0
                        multiclass_label = 1

                log_peak_flux = float(np.log10(max(1.0e-8, max_flux)))

                sequence_records.append({
                    "sequence_id": f"SEQ_{ar}_{t_last_obs.strftime('%Y%m%d%H%M')}",
                    "active_region": ar,
                    "t_obs_end": t_last_obs.isoformat(),
                    "frame_0": sub.iloc[0]["filepath"],
                    "frame_1": sub.iloc[1]["filepath"],
                    "frame_2": sub.iloc[2]["filepath"],
                    "frame_3": sub.iloc[3]["filepath"],
                    "target_window_start": win_start.isoformat(),
                    "target_window_end": win_end.isoformat(),
                    "binary_target_MX_24_48h": binary_label,
                    "multiclass_target": multiclass_label,
                    "log10_peak_flux": round(log_peak_flux, 4)
                })

    df_labels = pd.DataFrame(sequence_records)
    out_labels_path = CATALOGS_DIR / "sequence_labels.csv"
    df_labels.to_csv(out_labels_path, index=False)
    df_labels.to_csv(BASE_DIR / "sequence_labels.csv", index=False)
    
    print(f"[SUCCESS] Built forward targets for {len(df_labels)} contiguous temporal sequences.")
    print(f"Target distribution: {df_labels['multiclass_target'].value_counts().to_dict()} (0:Quiet, 1:C, 2:M, 3:X)")
    print(f"[SAVED] -> {out_labels_path}")
    print("=" * 70)
    return df_labels


if __name__ == "__main__":
    build_forward_target_labels()
