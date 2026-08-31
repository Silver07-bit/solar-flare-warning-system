"""
🛠️ Dataset Preparation & Preprocessing Pipeline
Processes observational sequences into 4-channel tensor representations:
  - Channel 0: Calibrated UV / Intensity Patch
  - Channel 1: Intensity-derived Spatial Flux Gradient (|∇I|) [Shear Complexity Proxy]
  - Channel 2: High-Frequency Laplacian Curvature (∇²I) [Loop Curvature Proxy]
  - Channel 3: Temporal Differential Rate (ΔI_t) [Flux Emergence Proxy]
Splits data strictly by NOAA Active Region:
  - Train: AR-13664, AR-12673, AR-11158
  - Validation: AR-12887
  - Test: AR-13000, AR-13100
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from config import (
    PROCESSED_DATA_DIR,
    CATALOGS_DIR,
    BASE_DIR,
    DATA_DIR,
    TRAIN_ACTIVE_REGIONS,
    VAL_ACTIVE_REGIONS,
    TEST_ACTIVE_REGIONS,
    IMG_SIZE,
    SEQ_LENGTH
)
from preprocess import (
    load_and_clean_fits,
    preprocess_solar_disk,
    extract_active_region,
    build_multi_channel_frame
)
from build_labels import build_forward_target_labels


def prepare_spatiotemporal_dataset():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PREPARING 4-CHANNEL SPATIO-TEMPORAL DATASET")
    print("=" * 70)

    # 1. Ensure target labels exist
    labels_csv = CATALOGS_DIR / "sequence_labels.csv"
    if not labels_csv.exists():
        labels_df = build_forward_target_labels()
    else:
        labels_df = pd.read_csv(labels_csv)

    print(f"Loaded {len(labels_df)} labeled sequences.")

    # 2. Partition by Active Region
    train_df = labels_df[labels_df["active_region"].isin(TRAIN_ACTIVE_REGIONS)].copy()
    val_df = labels_df[labels_df["active_region"].isin(VAL_ACTIVE_REGIONS)].copy()
    test_df = labels_df[labels_df["active_region"].isin(TEST_ACTIVE_REGIONS)].copy()

    # Fallback if hardcoded config regions do not match (e.g. real SDOBenchmark dataset)
    if len(train_df) == 0:
        unique_ars = sorted(list(labels_df["active_region"].unique()))
        n_ar = len(unique_ars)
        n_train = int(n_ar * 0.70)
        n_val = int(n_ar * 0.15)

        train_ars = set(unique_ars[:n_train])
        val_ars = set(unique_ars[n_train: n_train + n_val])
        test_ars = set(unique_ars[n_train + n_val:])

        train_df = labels_df[labels_df["active_region"].isin(train_ars)].copy()
        val_df = labels_df[labels_df["active_region"].isin(val_ars)].copy()
        test_df = labels_df[labels_df["active_region"].isin(test_ars)].copy()

    print(f"Active-Region-Aware Splits:")
    print(f"  • Train Set:      {len(train_df)} sequences | ARs: {list(train_df['active_region'].unique())}")
    print(f"  • Validation Set: {len(val_df)} sequences | ARs: {list(val_df['active_region'].unique())}")
    print(f"  • Test Set:       {len(test_df)} sequences | ARs: {list(test_df['active_region'].unique())}")

    train_df.to_csv(PROCESSED_DATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_split.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_split.csv", index=False)

    print(f"[SUCCESS] Dataset splits written to {PROCESSED_DATA_DIR}")
    print("=" * 70)
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_spatiotemporal_dataset()
