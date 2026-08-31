"""
☀️ Aditya-L1 Spatio-Temporal Dataset & Active-Region-Aware DataLoader
Features:
  1. Zero Label Leakage: Observational FITS headers contain only past metadata.
  2. Decoupled 24-48h Forward Target Window: Targets queried strictly from GOES catalogs.
  3. 4-Channel Multi-Spectral & Topological Tensor: [Sequence (4), Channels (4), Height (256), Width (256)]
  4. Active-Region-Aware Chronological Splitting (Zero active-region contamination across train/val/test).
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits

from config import (
    BASE_DIR,
    DATA_DIR,
    CATALOGS_DIR,
    PROCESSED_DATA_DIR,
    IMG_SIZE,
    SEQ_LENGTH,
    TRAIN_ACTIVE_REGIONS,
    VAL_ACTIVE_REGIONS,
    TEST_ACTIVE_REGIONS
)
from preprocess import (
    load_and_clean_fits,
    preprocess_solar_disk,
    extract_active_region,
    build_multi_channel_frame
)
from build_labels import build_forward_target_labels, extract_observation_metadata


_PATCH_CACHE = {}
_SEQ_CACHE = {}


def get_cached_patch(fpath):
    fpath_str = str(fpath)
    if fpath_str in _PATCH_CACHE:
        return _PATCH_CACHE[fpath_str]
    if Path(fpath).exists():
        raw = load_and_clean_fits(fpath)
        disk = preprocess_solar_disk(raw)
        patch = extract_active_region(disk, patch_size=IMG_SIZE)
    else:
        patch = np.zeros(IMG_SIZE, dtype=np.float32)
    _PATCH_CACHE[fpath_str] = patch
    return patch


CACHE_FILE = PROCESSED_DATA_DIR / "cached_sequences.pt"


def prewarm_global_sequence_cache(df=None):
    """Pre-extracts and caches all sequence tensors in memory and on disk for ultra-fast training."""
    global _SEQ_CACHE
    if len(_SEQ_CACHE) > 0:
        return

    if CACHE_FILE.exists():
        try:
            _SEQ_CACHE = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
            return
        except Exception:
            _SEQ_CACHE = {}

    if df is None:
        labels_csv = CATALOGS_DIR / "sequence_labels.csv"
        if not labels_csv.exists():
            labels_csv = BASE_DIR / "sequence_labels.csv"
        if labels_csv.exists():
            df = pd.read_csv(labels_csv)

    if df is not None:
        for idx in range(len(df)):
            row = df.iloc[idx]
            frame_paths = [
                Path(row.get("frame_0", row.get("filepath", ""))),
                Path(row.get("frame_1", "")),
                Path(row.get("frame_2", "")),
                Path(row.get("frame_3", ""))
            ]
            if not frame_paths[1].exists() and frame_paths[0].exists():
                frame_paths = [frame_paths[0]] * SEQ_LENGTH
            seq_key = tuple(str(p) for p in frame_paths)
            if seq_key not in _SEQ_CACHE:
                channel_frames = []
                prev_patch = None
                for fpath in frame_paths:
                    patch = get_cached_patch(fpath)
                    mch = build_multi_channel_frame(patch, prev_patch=prev_patch)
                    channel_frames.append(torch.tensor(mch, dtype=torch.float32))
                    prev_patch = patch
                _SEQ_CACHE[seq_key] = torch.stack(channel_frames, dim=0)

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(_SEQ_CACHE, CACHE_FILE)
        except Exception:
            pass


class SolarSequenceDataset(Dataset):
    """
    Spatio-Temporal Sequence Dataset for Solar Flare Forecasting.
    """

    def __init__(self, split_df=None, data_dir=None, is_demo_mode=False, auto_prewarm=True):
        self.is_demo_mode = is_demo_mode
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        
        if split_df is not None:
            self.df = split_df.reset_index(drop=True)
        else:
            # Build or load labels
            labels_csv = CATALOGS_DIR / "sequence_labels.csv"
            if not labels_csv.exists():
                labels_csv = BASE_DIR / "sequence_labels.csv"
            
            if labels_csv.exists():
                self.df = pd.read_csv(labels_csv)
            else:
                self.df = build_forward_target_labels(data_dir=self.data_dir)

        if auto_prewarm:
            prewarm_global_sequence_cache(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load the 4 sequence frames
        frame_paths = [
            Path(row.get("frame_0", row.get("filepath", ""))),
            Path(row.get("frame_1", "")),
            Path(row.get("frame_2", "")),
            Path(row.get("frame_3", ""))
        ]
        
        # Handle single-row fallback
        if not frame_paths[1].exists() and frame_paths[0].exists():
            frame_paths = [frame_paths[0]] * SEQ_LENGTH

        seq_key = tuple(str(p) for p in frame_paths)
        if seq_key in _SEQ_CACHE:
            seq_tensor = _SEQ_CACHE[seq_key]
        else:
            channel_frames = []
            prev_patch = None

            for fpath in frame_paths:
                patch = get_cached_patch(fpath)
                mch = build_multi_channel_frame(patch, prev_patch=prev_patch)
                channel_frames.append(torch.tensor(mch, dtype=torch.float32))
                prev_patch = patch

            # Tensor shape: [T (4), C (4), H (256), W (256)]
            seq_tensor = torch.stack(channel_frames, dim=0)
            _SEQ_CACHE[seq_key] = seq_tensor

        targets = {
            "binary_label": torch.tensor(int(row.get("binary_target_MX_24_48h", 0)), dtype=torch.long),
            "multiclass_label": torch.tensor(int(row.get("multiclass_target", 0)), dtype=torch.long),
            "log_flux": torch.tensor(float(row.get("log10_peak_flux", -7.5)), dtype=torch.float32),
            "active_region": str(row.get("active_region", "AR-13664")),
            "t_obs_end": str(row.get("t_obs_end", ""))
        }

        return seq_tensor, targets


def get_active_region_split_datasets():
    """
    Returns train, validation, and test datasets partitioned strictly by NOAA active regions.
    """
    train_split_csv = PROCESSED_DATA_DIR / "train_split.csv"
    val_split_csv = PROCESSED_DATA_DIR / "val_split.csv"
    test_split_csv = PROCESSED_DATA_DIR / "test_split.csv"

    if train_split_csv.exists() and val_split_csv.exists() and test_split_csv.exists():
        train_df = pd.read_csv(train_split_csv)
        val_df = pd.read_csv(val_split_csv)
        test_df = pd.read_csv(test_split_csv)
    else:
        labels_csv = CATALOGS_DIR / "sequence_labels.csv"
        if not labels_csv.exists():
            labels_csv = BASE_DIR / "sequence_labels.csv"

        if labels_csv.exists():
            labels_df = pd.read_csv(labels_csv)
        else:
            labels_df = build_forward_target_labels()

        train_df = labels_df[labels_df["active_region"].isin(TRAIN_ACTIVE_REGIONS)].copy()
        val_df = labels_df[labels_df["active_region"].isin(VAL_ACTIVE_REGIONS)].copy()
        test_df = labels_df[labels_df["active_region"].isin(TEST_ACTIVE_REGIONS)].copy()

        # Fallback if regions don't match hardcoded list: disjoint partition of unique ARs
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

    train_ds = SolarSequenceDataset(split_df=train_df)
    val_ds = SolarSequenceDataset(split_df=val_df)
    test_ds = SolarSequenceDataset(split_df=test_df)

    return train_ds, val_ds, test_ds