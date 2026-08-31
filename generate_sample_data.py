"""
☀️ Aditya-L1 Historical Solar Observation & GOES X-Ray Flare Catalog Generator
Simulates a multi-region chronological solar dataset with per-instance randomization,
near-miss active regions, and zero label leakage in FITS headers.
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from astropy.io import fits
from config import BASE_DIR, DATA_DIR


def generate_solar_physics_disk(size=512, time_step=0, region_type="X_FLARE", seed=42):
    """
    Simulates solar disk physics (limb darkening, coronal background, magnetic active regions)
    with realistic per-instance randomization (spatial jitter, intensity/spread variance, granulation).
    """
    rng = np.random.RandomState(seed + (time_step * 37) + int(time_step ** 2))
    
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2
    radius = size * 0.42

    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    disk_mask = dist <= radius

    # Limb darkening: I(mu) = I_0 * (0.3 + 0.7 * sqrt(1 - (r/R)^2))
    mu = np.sqrt(np.maximum(0.0, 1.0 - (dist / radius) ** 2))
    solar_disk = (0.3 + 0.7 * mu) * disk_mask * 1000.0

    # Solar background granulation & coronal noise
    granulation = rng.normal(0.0, 12.0, (size, size))
    corona_noise = rng.normal(5.0, 1.8, (size, size))
    image = np.where(disk_mask, solar_disk + granulation, corona_noise)

    # Per-instance randomization parameters (jitter & intensity scaling)
    jitter_x = rng.uniform(-5.0, 5.0)
    jitter_y = rng.uniform(-5.0, 5.0)
    int_scale = rng.uniform(0.85, 1.15)
    spread_scale = rng.uniform(0.88, 1.12)

    if region_type == "X_FLARE":
        # Complex delta-configuration with rapid temporal flux emergence
        ar_cy = center_y - 45 + int(time_step * 2.2) + jitter_y
        ar_cx = center_x + 35 + int(time_step * 3.1) + jitter_x
        intensity = (3000.0 + (time_step * 1100.0)) * int_scale
        spread = (22 + (time_step * 1.8)) * spread_scale

        spot1 = np.exp(-((x - ar_cx) ** 2 + (y - ar_cy) ** 2) / (2 * spread ** 2))
        spot2 = np.exp(-((x - (ar_cx - 16)) ** 2 + (y - (ar_cy + 16)) ** 2) / (2 * (spread - 4) ** 2))
        image += (spot1 * intensity) + (spot2 * intensity * 0.8)

    elif region_type == "M_FLARE":
        # Moderate active region with moderate shear and emergence
        ar_cy = center_y + 35 + int(time_step * 1.2) + jitter_y
        ar_cx = center_x - 45 + int(time_step * 1.9) + jitter_x
        intensity = (1800.0 + (time_step * 420.0)) * int_scale
        spread = 16.0 * spread_scale
        spot = np.exp(-((x - ar_cx) ** 2 + (y - ar_cy) ** 2) / (2 * spread ** 2))
        image += spot * intensity

    elif region_type == "NEAR_MISS_COMPLEX":
        # Visually complex multi-polar plage (multiple sunspots, high spatial shear proxy)
        # but temporally stable / quiescent (NO explosive emergence, NO M/X flare)
        ar_cy = center_y - 30 + int(time_step * 0.3) + jitter_y
        ar_cx = center_x + 40 + int(time_step * 0.4) + jitter_x
        base_int = 2100.0 * int_scale
        
        s1 = np.exp(-((x - ar_cx) ** 2 + (y - ar_cy) ** 2) / (2 * (15.0 * spread_scale) ** 2))
        s2 = np.exp(-((x - (ar_cx + 18)) ** 2 + (y - (ar_cy - 14)) ** 2) / (2 * (12.0 * spread_scale) ** 2))
        s3 = np.exp(-((x - (ar_cx - 16)) ** 2 + (y - (ar_cy + 16)) ** 2) / (2 * (11.0 * spread_scale) ** 2))
        image += (s1 * base_int) + (s2 * base_int * 0.75) + (s3 * base_int * 0.65)

    elif region_type == "C_FLARE":
        # Mild active plage
        ar_cy = center_y - 20 + int(time_step * 0.8) + jitter_y
        ar_cx = center_x - 30 + int(time_step * 0.8) + jitter_x
        spread = 14.0 * spread_scale
        spot = np.exp(-((x - ar_cx) ** 2 + (y - ar_cy) ** 2) / (2 * spread ** 2))
        image += spot * (900.0 * int_scale)

    else:  # QUIET_SUN
        # Stable dipole or quiet unipolar sunspot
        ar_cy = center_y + 60 + jitter_y
        ar_cx = center_x - 70 + jitter_x
        spread = 10.0 * spread_scale
        spot = np.exp(-((x - ar_cx) ** 2 + (y - ar_cy) ** 2) / (2 * spread ** 2))
        image += spot * (400.0 * int_scale)

    return np.clip(image, 0.0, None).astype(np.float32)


def save_pure_fits_observation(filepath, data, obs_time_str, noaa_ar):
    """
    Saves a pure FITS observation HDU.
    CRITICAL: Contains ONLY past/present observational metadata. NO label leakage!
    """
    hdu = fits.PrimaryHDU(data)
    hdr = hdu.header
    hdr["TELESCOP"] = "Aditya-L1"
    hdr["INSTRUME"] = "SUIT"
    hdr["WAVELNTH"] = "279.6 nm (Mg II k)"
    hdr["DATE-OBS"] = obs_time_str
    hdr["NOAA_AR"] = noaa_ar
    hdr["EXPTIME"] = 0.250
    hdr["OBS_GEO"] = "Lagrange Point L1"
    hdu.writeto(filepath, overwrite=True)


def build_historical_dataset():
    print("Building multi-region randomized solar FITS dataset & independent GOES flare catalog...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scenarios_dir = BASE_DIR / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Active Region tracking list (including Near-Miss Hard Negative & Pure C-Class Regions)
    active_regions_profile = [
        {"ar": "AR-13664", "type": "X_FLARE", "flare_class": "X2.8", "peak_flux": 2.8e-4, "flare_offset_hours": 42, "frames": 24, "seed": 100},
        {"ar": "AR-12673", "type": "X_FLARE", "flare_class": "X1.4", "peak_flux": 1.4e-4, "flare_offset_hours": 40, "frames": 20, "seed": 200},
        {"ar": "AR-11158", "type": "M_FLARE", "flare_class": "M5.4", "peak_flux": 5.4e-5, "flare_offset_hours": 42, "frames": 20, "seed": 300},
        {"ar": "AR-12887", "type": "M_FLARE", "flare_class": "M2.1", "peak_flux": 2.1e-5, "flare_offset_hours": 38, "frames": 20, "seed": 400},
        {"ar": "AR-13000", "type": "C_FLARE", "flare_class": "C4.5", "peak_flux": 4.5e-6, "flare_offset_hours": 36, "frames": 16, "seed": 500},
        {"ar": "AR-13200", "type": "C_FLARE", "flare_class": "C2.8", "peak_flux": 2.8e-6, "flare_offset_hours": 36, "frames": 20, "seed": 650},
        {"ar": "AR-13300", "type": "C_FLARE", "flare_class": "C1.5", "peak_flux": 1.5e-6, "flare_offset_hours": 38, "frames": 20, "seed": 700},
        {"ar": "AR-13450", "type": "NEAR_MISS_COMPLEX", "flare_class": "Quiet", "peak_flux": 3.2e-7, "flare_offset_hours": 0, "frames": 20, "seed": 550},
        {"ar": "AR-13500", "type": "NEAR_MISS_COMPLEX", "flare_class": "Quiet", "peak_flux": 2.8e-7, "flare_offset_hours": 0, "frames": 20, "seed": 580},
        {"ar": "AR-13700", "type": "NEAR_MISS_COMPLEX", "flare_class": "Quiet", "peak_flux": 3.0e-7, "flare_offset_hours": 0, "frames": 20, "seed": 800},
        {"ar": "AR-13100", "type": "QUIET_SUN", "flare_class": "Quiet", "peak_flux": 1.2e-7, "flare_offset_hours": 0, "frames": 20, "seed": 600},
        {"ar": "AR-13600", "type": "QUIET_SUN", "flare_class": "Quiet", "peak_flux": 1.5e-7, "flare_offset_hours": 0, "frames": 20, "seed": 750},
    ]

    goes_events = []
    current_time = base_time

    # 1. Populate main dataset with chronological FITS frames
    for profile in active_regions_profile:
        ar_name = profile["ar"]
        r_type = profile["type"]
        n_frames = profile["frames"]
        flare_class = profile["flare_class"]
        peak_flux = profile["peak_flux"]
        offset_hrs = profile["flare_offset_hours"]
        seed = profile["seed"]

        ar_start_time = current_time

        # Generate FITS frames at 3-hour cadences
        for t in range(n_frames):
            frame_time = ar_start_time + timedelta(hours=t * 3)
            time_str = frame_time.strftime("%Y-%m-%dT%H:%M:%S.000")
            filename = f"suit_{ar_name}_T{t:03d}_{frame_time.strftime('%Y%m%d%H%M')}.fits"
            filepath = DATA_DIR / filename

            disk_data = generate_solar_physics_disk(time_step=t, region_type=r_type, seed=seed)
            save_pure_fits_observation(filepath, disk_data, time_str, ar_name)

        # Register corresponding GOES flare event in the forward window (if not Quiet)
        if flare_class != "Quiet":
            flare_start = ar_start_time + timedelta(hours=offset_hrs)
            flare_peak = flare_start + timedelta(minutes=18)
            flare_end = flare_start + timedelta(minutes=45)
            goes_events.append({
                "flare_id": f"GOES_{ar_name}_{flare_start.strftime('%Y%m%d%H%M')}",
                "active_region": ar_name,
                "start_time": flare_start.strftime("%Y-%m-%dT%H:%M:%S.000"),
                "peak_time": flare_peak.strftime("%Y-%m-%dT%H:%M:%S.000"),
                "end_time": flare_end.strftime("%Y-%m-%dT%H:%M:%S.000"),
                "flare_class": flare_class,
                "peak_flux_wm2": peak_flux,
                "integrated_flux_jm2": peak_flux * 1800.0
            })

        current_time += timedelta(hours=(n_frames * 3) + 48)

    # 2. Build Named Demo Scenarios for UI & API Inspection
    scenarios_config = {
        "AR3664_Impending_X_Flare": {"type": "X_FLARE", "ar": "AR-13664", "class": "X2.8", "flux": 2.8e-4, "offset": 30},
        "AR3685_M_Class_Eruption": {"type": "M_FLARE", "ar": "AR-11158", "class": "M5.4", "flux": 5.4e-5, "offset": 34},
        "AR3670_Quiet_Sun": {"type": "QUIET_SUN", "ar": "AR-13100", "class": "Quiet", "flux": 1.2e-7, "offset": 0}
    }

    for scn_name, scn_data in scenarios_config.items():
        scn_path = scenarios_dir / scn_name
        scn_path.mkdir(parents=True, exist_ok=True)
        scn_start = datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc)

        for step in range(4):
            f_time = scn_start + timedelta(hours=step * 3)
            time_str = f_time.strftime("%Y-%m-%dT%H:%M:%S.000")
            fname = f"{scn_name}_step_{step}.fits"
            f_out = scn_path / fname

            d_data = generate_solar_physics_disk(time_step=step, region_type=scn_data["type"], seed=999)
            save_pure_fits_observation(f_out, d_data, time_str, scn_data["ar"])

    # 3. Save Independent GOES Flare Catalog
    catalog_df = pd.DataFrame(goes_events)
    catalog_path = BASE_DIR / "goes_flare_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    
    catalogs_sub_dir = BASE_DIR / "catalogs"
    catalogs_sub_dir.mkdir(parents=True, exist_ok=True)
    catalog_df.to_csv(catalogs_sub_dir / "goes_flare_catalog.csv", index=False)

    print(f"Generated FITS files in {DATA_DIR}")
    print(f"Saved {len(catalog_df)} independent GOES flare events to {catalog_path}")


if __name__ == "__main__":
    build_historical_dataset()
