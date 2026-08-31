"""
☀️ Aditya-L1 Spatio-Temporal Multi-Task Deep Learning Forecasting Architecture
Components:
  1. 4-Channel Multi-Spectral & Topological Spatial Feature Encoder
  2. Recurrent ConvLSTM Spatio-Temporal Sequence Cell
  3. Multi-Task Heads:
     - Head A: 24-48h Binary M/X-Class Eruption Probability (Cross-Entropy)
     - Head B: 4-Class NOAA Flare Classification [Quiet/B, C-Class, M-Class, X-Class] (Cross-Entropy)
     - Head C: Continuous Log10 Peak X-Ray Flux Estimation (MSE Regression)
  4. Temperature Scaling Model Calibrator for Post-Hoc Probability Calibration
  5. Authentic PyTorch Autograd Spatio-Temporal Grad-CAM for Model Attribution Explainability
"""

import torch
import torch.nn as nn
import cv2
import numpy as np


class ConvLSTMCell(nn.Module):
    """Custom ConvLSTM block for extracting spatio-temporal feature evolutions."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_channels, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        new_c = (forgetgate * c) + (ingate * cellgate)
        new_h = outgate * torch.tanh(new_c)
        return new_h, new_c


class ModelCalibrator(nn.Module):
    """
    Temperature Scaling Layer for Post-Hoc Probability Calibration (Platt Scaling).
    Learns single temperature parameter T > 0 on validation set such that:
      P_calibrated = softmax(Logits / T)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, logits):
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / temp

    def fit(self, val_logits, val_labels, lr=0.05, max_iter=20):
        optimizer = torch.optim.Adam([self.temperature], lr=lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            optimizer.step()


class SolarFlarePredictor(nn.Module):
    """
    4-Channel Spatio-Temporal Multi-Task Forecasting Network.
    Input Shape: [Batch, Time (4), Channels (4), Height (256), Width (256)]
    Channels:
      - Ch 0: Calibrated UV / Optical Intensity
      - Ch 1: Intensity-derived Spatial Flux Gradient (|∇I|) [Shear Complexity Proxy]
      - Ch 2: High-Frequency Laplacian Curvature (∇²I) [Loop Complexity Proxy]
      - Ch 3: Temporal Differential Rate (ΔI_t) [Flux Emergence Rate]
    """

    def __init__(self, in_channels=4, hidden_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 2D Spatial Feature Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),            # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                             # [B, 32, 32, 32]
        )

        # Spatio-Temporal Recurrent Sequence Cell
        self.conv_lstm = ConvLSTMCell(in_channels=32, hidden_channels=hidden_dim)

        # Adaptive Pooling Layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-Task Prediction Heads
        # Head A: 24-48h Binary M/X-Class Eruption (0: Low/No Flare, 1: M/X Flare)
        self.binary_head = nn.Linear(hidden_dim, 2)

        # Head B: NOAA Multi-Class Flare Classification [0: Quiet/B, 1: C-Class, 2: M-Class, 3: X-Class]
        self.multiclass_head = nn.Linear(hidden_dim, 4)

        # Head C: Continuous Log10 Peak X-Ray Flux (e.g. -7.5 W/m² to -3.5 W/m²)
        self.flux_regression_head = nn.Linear(hidden_dim, 1)

        # Probability Calibrator
        self.calibrator = ModelCalibrator()

    def forward(self, x, return_all_heads=True):
        b, t, c, h, w = x.shape

        # Support single-channel inputs by expanding
        if c == 1 and self.in_channels == 4:
            x = x.repeat(1, 1, 4, 1, 1)
            c = 4

        h_feat = h // 8
        w_feat = w // 8
        h_state = torch.zeros(b, self.hidden_dim, h_feat, w_feat, device=x.device)
        c_state = torch.zeros(b, self.hidden_dim, h_feat, w_feat, device=x.device)

        # Recurrent sequence processing
        for i in range(t):
            frame_input = x[:, i, :, :, :]
            spatial_features = self.encoder(frame_input)
            h_state, c_state = self.conv_lstm(spatial_features, h_state, c_state)

        # Feature bottleneck
        features = self.pool(h_state).flatten(start_dim=1)

        binary_logits = self.binary_head(features)
        calibrated_binary_logits = self.calibrator(binary_logits)
        multiclass_logits = self.multiclass_head(features)
        # Bounded between -8.0 (Quiet baseline 10^-8 W/m²) and -3.0 (X10 Superflare 10^-3 W/m²)
        log_flux_pred = -8.0 + 5.0 * torch.sigmoid(self.flux_regression_head(features).squeeze(-1))

        if not return_all_heads:
            return binary_logits

        return {
            "binary_logits": binary_logits,
            "calibrated_binary_logits": calibrated_binary_logits,
            "multiclass_logits": multiclass_logits,
            "log_flux_pred": log_flux_pred
        }


class SpatioTemporalGradCAM:
    """
    Computes genuine Gradient-weighted Class Activation Mapping (Grad-CAM)
    across the temporal sequence of the CNN-ConvLSTM model using PyTorch autograd hooks.
    
    Formula:
      α_k^(t) = (1/Z) Σ_{i,j} (∂y^c / ∂A_{k,i,j}^(t))
      L_{Grad-CAM}^(t) = ReLU(Σ_k α_k^(t) A_k^(t))
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.encoder[3]
        self.activations = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.insert(0, grad_out[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, seq_tensor, target_class=1, task="binary"):
        """
        Generates frame-by-frame Grad-CAM heatmaps for the given input sequence tensor.
        seq_tensor: [1, T, C, H, W]
        Returns: (list of 2D numpy arrays of shape [H, W], predictions_dict)
        """
        self.activations = []
        self.gradients = []
        self.model.eval()

        input_var = seq_tensor.clone().detach().requires_grad_(True)
        preds = self.model(input_var, return_all_heads=True)

        self.model.zero_grad()
        if task == "binary":
            score = preds["binary_logits"][0, target_class]
        else:
            score = preds["multiclass_logits"][0, target_class]

        score.backward()

        cams = []
        t = seq_tensor.shape[1]
        h_orig, w_orig = seq_tensor.shape[3], seq_tensor.shape[4]

        for i in range(min(t, len(self.activations), len(self.gradients))):
            act = self.activations[i]
            grad = self.gradients[i]

            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = torch.relu(cam)

            cam_np = cam.squeeze().detach().cpu().numpy()

            denom = cam_np.max() - cam_np.min()
            if denom > 1e-8:
                cam_np = (cam_np - cam_np.min()) / denom
            else:
                cam_np = np.zeros_like(cam_np)

            cam_resized = cv2.resize(cam_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            cams.append(cam_resized)

        # Probabilities
        raw_probs = torch.softmax(preds["binary_logits"], dim=1).detach().cpu().numpy()[0]
        calibrated_probs = torch.softmax(preds["calibrated_binary_logits"], dim=1).detach().cpu().numpy()[0]
        multi_probs = torch.softmax(preds["multiclass_logits"], dim=1).detach().cpu().numpy()[0]
        flux_val = float(preds["log_flux_pred"].detach().cpu().numpy()[0])

        clean_preds = {
            "raw_binary_probs": raw_probs,
            "calibrated_binary_probs": calibrated_probs,
            "multiclass_probs": multi_probs,
            "log_flux_pred": flux_val,
            "confidence": float(max(calibrated_probs))
        }

        return cams, clean_preds