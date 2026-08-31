"""
☀️ Aditya-L1 Image Preprocessing & Multi-Channel Feature Engineering Pipeline
Handles:
  1. FITS Data Ingestion & Atmospheric/Sky Noise Filtering
  2. Multi-Spectral False-Color Rendering (SUIT UV 279nm, AIA 171/193, Magnetogram, Inferno)
  3. 4-Channel Feature Synthesis:
     - Channel 0: Calibrated Intensity Patch (Normalized Dynamic Range)
     - Channel 1: Spatial Gradient Magnitude (|∇I|) [Spatial Shear Proxy]
     - Channel 2: Spatial Laplacian Curvature (∇²I) [High-Frequency Magnetic Loop Proxy]
     - Channel 3: Temporal Differential Rate (ΔI_t) [Flux Emergence Rate]
  4. Scientifically Honest Optical & Topological Proxies
"""

import cv2
import numpy as np
from astropy.io import fits


def load_and_clean_fits(fits_path):
    """Loads a FITS file and removes sky background noise/NaN values."""
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[0].data.astype(np.float32) if hdul[0].data is not None else hdul[1].data.astype(np.float32)

    data = np.nan_to_num(data, nan=0.0)
    data[data < 0] = 0.0
    return data


def preprocess_solar_disk(image, target_size=(512, 512)):
    """Applies logarithmic scaling and min-max normalization."""
    log_img = np.log1p(image)
    denom = log_img.max() - log_img.min() + 1e-8
    norm_img = (log_img - log_img.min()) / denom
    return cv2.resize(norm_img, target_size, interpolation=cv2.INTER_AREA)


def extract_active_region(image, bbox=None, patch_size=(256, 256)):
    """
    Crops the brightest Active Region dynamically or extracts by bounding box.
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
    else:
        h, w = image.shape
        cy, cx = np.unravel_index(np.argmax(image), image.shape)
        half_h, half_w = patch_size[0] // 2, patch_size[1] // 2
        y1, y2 = max(0, cy - half_h), min(h, cy + half_h)
        x1, x2 = max(0, cx - half_w), min(w, cx + half_w)
        crop = image[y1:y2, x1:x2]

    return cv2.resize(crop, patch_size)


def build_multi_channel_frame(current_patch, prev_patch=None):
    """
    Constructs a 4-channel multi-spectral/topological feature tensor of shape [4, H, W]:
      - Channel 0: Calibrated Optical/UV Intensity Patch [H, W]
      - Channel 1: Spatial Gradient Magnitude (|∇I|) [H, W]
      - Channel 2: Spatial Laplacian Curvature (∇²I) [H, W]
      - Channel 3: Temporal Differential Rate (ΔI_t) [H, W]
    """
    h, w = current_patch.shape

    # Ch 0: Intensity
    ch0 = current_patch.astype(np.float32)

    # Ch 1: Spatial Gradient (|∇I|)
    gx = cv2.Sobel(current_patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(current_patch, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    ch1 = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

    # Ch 2: Laplacian Curvature (∇²I)
    lap = cv2.Laplacian(current_patch, cv2.CV_32F, ksize=3)
    ch2 = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)

    # Ch 3: Temporal Differential (ΔI_t)
    if prev_patch is not None:
        diff = np.abs(current_patch - prev_patch)
        ch3 = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    else:
        ch3 = np.zeros((h, w), dtype=np.float32)

    # Stack into [4, H, W]
    return np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)


def apply_spectral_colormap(gray_image, colormap_name="SUIT_UV_279"):
    """
    Renders false-color multi-spectral representations commonly used in solar physics:
      - SUIT_UV_279: Aditya-L1 SUIT Mg II k UV filter representation
      - AIA_171_GOLD: SDO/AIA 171 Å Quiet Corona & Magnetic Loop Golden palette
      - AIA_193_BRONZE: SDO/AIA 193 Å Active Region Bronze palette
      - MAGNETOGRAM: SOHO/MDI line-of-sight dipole contrast representation
      - PLASMA_INFERNO: High-energy thermal gradient colormap
    """
    img_uint8 = np.clip(gray_image * 255.0, 0, 255).astype(np.uint8)

    if colormap_name == "SUIT_UV_279":
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = int(np.clip((i - 100) * 1.6, 0, 255))
            g = int(np.clip(i * 1.1, 0, 255))
            b = int(np.clip(120 + i * 0.6, 0, 255))
            lut[i, 0] = [b, g, r]
        colored = cv2.LUT(cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR), lut)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    elif colormap_name == "AIA_171_GOLD":
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_AUTUMN)
    elif colormap_name == "AIA_193_BRONZE":
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_COPPER)
    elif colormap_name == "MAGNETOGRAM":
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_TWILIGHT_SHIFTED)
    elif colormap_name == "PLASMA_INFERNO":
        return cv2.applyColorMap(img_uint8, cv2.COLORMAP_INFERNO)
    else:
        return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)


def compute_magnetic_flux_gradient(gray_image):
    """
    Computes spatial flux gradients (|∇I|) representing optical shear complexity.
    """
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    denom = grad_mag.max() - grad_mag.min() + 1e-8
    grad_norm = (grad_mag - grad_mag.min()) / denom

    threshold_val = np.percentile(grad_norm, 75)
    binary_mask = (grad_norm > threshold_val).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return grad_norm, contours


def compute_high_frequency_curvature(gray_image):
    """
    Computes spatial Laplacian second derivatives (∇²I) representing
    high-frequency curvature and magnetic loop complexity.
    """
    lap = cv2.Laplacian(gray_image, cv2.CV_32F, ksize=3)
    denom = lap.max() - lap.min() + 1e-8
    return (lap - lap.min()) / denom


def compute_optical_flux_and_shear_proxies(patch):
    """
    Computes honest optical and topological image proxies from the active region patch:
      - Total Optical Intensity Flux Proxy (Φ_opt)
      - Peak Intensity Gradient Magnitude (|∇I|)
      - Topological Active Region Complexity Index
    """
    phi_proxy = float(np.sum(patch) / 1000.0)
    grad_norm, contours = compute_magnetic_flux_gradient(patch)
    max_gradient = float(np.max(grad_norm))
    shear_complexity = float(len(contours) * 1.5 + (np.mean(grad_norm) * 100.0))

    return {
        "unsigned_flux_proxy": phi_proxy,
        "max_flux_gradient": max_gradient,
        "shear_complexity_index": min(100.0, shear_complexity),
        "total_contour_loops": len(contours)
    }


# Backwards compatibility alias
compute_solar_physical_metrics = compute_optical_flux_and_shear_proxies