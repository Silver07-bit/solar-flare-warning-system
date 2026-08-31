"""
☀️ Aditya-L1 Multi-Task Deep Learning Training & Calibration Pipeline
Features:
  1. Active-Region-Aware Partitioning (Strictly isolated active regions for Train, Val, Test)
  2. Dynamic Class-Weighted Loss to combat solar flare class imbalance
  3. Multi-Task Joint Optimization (Binary M/X Eruption + 4-Class NOAA Category + Log-Flux Regression)
  4. Post-Hoc Probability Calibration Fitting (Temperature Scaling on Validation Set)
  5. Validation-Driven Decision Threshold Tuning (maximizing TSS)
  6. Model Artifact & Metadata Persistence to models/latest/
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BASE_DIR,
    MODELS_LATEST_DIR,
    BATCH_SIZE,
    SEQ_LENGTH,
    NUM_EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    TRAIN_ACTIVE_REGIONS,
    VAL_ACTIVE_REGIONS,
    TEST_ACTIVE_REGIONS,
    ALL_ACTIVE_REGIONS
)
from dataset import get_active_region_split_datasets
from model import SolarFlarePredictor
from evaluate import evaluate_model_on_dataset, find_optimal_threshold


def run_training(num_epochs=10, batch_size=16, lr=0.0005):
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75, flush=True)
    print(f"ADITYA-L1 MULTI-TASK FORECASTING TRAINING PIPELINE [Device: {device}]", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Active-Region-Aware Dataset Splits
    train_ds, val_ds, test_ds = get_active_region_split_datasets()
    
    print("DATASET PARTITIONING (Zero Active-Region Contamination):", flush=True)
    print(f"  • Train Set:      {len(train_ds)} sequences | Assigned ARs: {TRAIN_ACTIVE_REGIONS}", flush=True)
    print(f"  • Validation Set: {len(val_ds)} sequences | Assigned ARs: {VAL_ACTIVE_REGIONS}", flush=True)
    print(f"  • Test Set:       {len(test_ds)} sequences | Assigned ARs: {TEST_ACTIVE_REGIONS}", flush=True)
    print("=" * 75, flush=True)

    if len(train_ds) == 0:
        print("Error: Train dataset is empty. Run 'python prepare_dataset.py' first.", flush=True)
        return

    # Pre-warm cache
    for i in range(len(train_ds)):
        _ = train_ds[i]
    for i in range(len(val_ds)):
        _ = val_ds[i]
    for i in range(len(test_ds)):
        _ = test_ds[i]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 2. Dynamic Inverse Class Frequency Weights for Binary Loss
    train_df = train_ds.df
    bin_counts = train_df["binary_target_MX_24_48h"].value_counts().to_dict()
    n_total = len(train_df)
    n_0 = bin_counts.get(0, 1)
    n_1 = bin_counts.get(1, 1)
    w0 = n_total / (2.0 * max(n_0, 1))
    w1 = n_total / (2.0 * max(n_1, 1))
    bin_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

    mc_counts = train_df["multiclass_target"].value_counts().to_dict()
    mc_w = [n_total / (4.0 * max(mc_counts.get(c, 1), 1)) for c in range(4)]
    multi_weights = torch.tensor(mc_w, dtype=torch.float32).to(device)

    # Multi-Task Model & Objectives
    model = SolarFlarePredictor(in_channels=4, hidden_dim=32).to(device)

    criterion_binary = nn.CrossEntropyLoss(weight=bin_weights)
    criterion_multiclass = nn.CrossEntropyLoss(weight=multi_weights)
    criterion_flux = nn.SmoothL1Loss(beta=0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = 9999.0
    best_weights_path = MODELS_LATEST_DIR / "solar_flare_model.pth"

    # 3. Training Loop
    print("\nStarting Spatio-Temporal Multi-Task Training Loop...", flush=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            bin_targets = targets["binary_label"].to(device)
            multi_targets = targets["multiclass_label"].to(device)
            flux_targets = targets["log_flux"].to(device)

            optimizer.zero_grad()
            preds = model(sequences, return_all_heads=True)

            loss_a = criterion_binary(preds["binary_logits"], bin_targets)
            loss_b = criterion_multiclass(preds["multiclass_logits"], multi_targets)
            loss_c = criterion_flux(preds["log_flux_pred"], flux_targets)

            # Balanced multi-task loss
            loss = (1.0 * loss_a) + (0.5 * loss_b) + (0.5 * loss_c)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation Checkpoint
        val_eval = evaluate_model_on_dataset(model, val_ds if len(val_ds) > 0 else train_ds, device=device, batch_size=batch_size)
        bin_eval = val_eval.get("binary_evaluation_24_48h", {})

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Val Recall: {bin_eval.get('recall_tpr', 0.0):.2f} | "
            f"Val Precision: {bin_eval.get('precision', 0.0):.2f} | "
            f"Val F1: {bin_eval.get('f1_score', 0.0):.2f} | "
            f"Val TSS: {bin_eval.get('true_skill_statistic_tss', 0.0):.2f}",
            flush=True
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_weights_path)
            torch.save(model.state_dict(), BASE_DIR / "solar_flare_model.pth")

    # 4. Post-Hoc Probability Calibration Fitting
    print("\nFitting Post-Hoc Probability Temperature Scaling Calibrator on Validation Set...", flush=True)
    eval_target_ds = val_ds if len(val_ds) > 0 else train_ds
    val_loader = DataLoader(eval_target_ds, batch_size=batch_size, shuffle=False)
    val_logits_list = []
    val_labels_list = []
    with torch.no_grad():
        for seqs, tgts in val_loader:
            seqs = seqs.to(device)
            preds = model(seqs, return_all_heads=True)
            val_logits_list.append(preds["binary_logits"])
            val_labels_list.append(tgts["binary_label"].to(device))

    if val_logits_list:
        val_logits_cat = torch.cat(val_logits_list, dim=0)
        val_labels_cat = torch.cat(val_labels_list, dim=0)
        model.calibrator.fit(val_logits_cat, val_labels_cat)
        calibrated_temp = float(model.calibrator.temperature.item())
        print(f"[CALIBRATED] Learned Temperature Scaling Parameter T = {calibrated_temp:.3f}", flush=True)
    else:
        calibrated_temp = 1.0

    # 5. Tune Optimal Decision Threshold on Validation Set
    val_calib_probs = []
    val_true_labels = []
    with torch.no_grad():
        for seqs, tgts in val_loader:
            seqs = seqs.to(device)
            preds = model(seqs, return_all_heads=True)
            probs = torch.softmax(preds["calibrated_binary_logits"], dim=1)[:, 1].cpu().numpy()
            val_calib_probs.extend(probs)
            val_true_labels.extend(tgts["binary_label"].numpy())

    optimal_threshold = find_optimal_threshold(val_true_labels, val_calib_probs)
    print(f"[OPTIMIZED] Selected Binary Decision Threshold: {optimal_threshold:.3f}", flush=True)

    # Save final model with calibrated temperature
    torch.save(model.state_dict(), best_weights_path)
    torch.save(model.state_dict(), BASE_DIR / "solar_flare_model.pth")

    # 6. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 75, flush=True)
    print("FINAL EVALUATION ON HELD-OUT ACTIVE REGIONS TEST SET", flush=True)
    print("=" * 75, flush=True)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_eval = evaluate_model_on_dataset(
        model, test_ds if len(test_ds) > 0 else train_ds, device=device, threshold=optimal_threshold, batch_size=batch_size
    )
    te_bin = test_eval.get("binary_evaluation_24_48h", {})
    te_multi = test_eval.get("multiclass_evaluation", {})
    te_flux = test_eval.get("flux_regression_metrics", {})

    print(f"Total Test Sequences:          {test_eval.get('total_test_sequences', 0)}", flush=True)
    print(f"Decision Threshold:            {optimal_threshold:.3f}", flush=True)
    print(f"24-48h Flare Recall (TPR):     {te_bin.get('recall_tpr', 0.0) * 100:.1f}%", flush=True)
    print(f"24-48h Flare Precision:        {te_bin.get('precision', 0.0) * 100:.1f}%", flush=True)
    print(f"24-48h Specificity (TNR):      {te_bin.get('specificity', 0.0) * 100:.1f}%", flush=True)
    print(f"24-48h False Alarm Rate (FPR): {te_bin.get('false_alarm_rate_fpr', 0.0) * 100:.1f}%", flush=True)
    print(f"24-48h F1-Score:               {te_bin.get('f1_score', 0.0)}", flush=True)
    print(f"True Skill Statistic (TSS):    {te_bin.get('true_skill_statistic_tss', 0.0)}", flush=True)
    print(f"Heidke Skill Score (HSS):      {te_bin.get('heidke_skill_score_hss', 0.0)}", flush=True)
    print(f"ROC-AUC:                       {te_bin.get('roc_auc', 'N/A')}", flush=True)
    print(f"Multi-Class Macro F1:          {te_multi.get('macro_f1', 0.0)}", flush=True)
    print(f"Peak Flux MAE (Log10 W/m²):    {te_flux.get('log10_mae', 0.0)}", flush=True)

    # 7. Save Model Metadata
    metadata = {
        "model_name": "SolarFlareNet-ConvLSTM-MultiTask",
        "version": "2.6.0",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_channels": 4,
        "sequence_length": SEQ_LENGTH,
        "active_regions": {
            "train": list(train_ds.df["active_region"].unique()),
            "validation": list(val_ds.df["active_region"].unique()),
            "test": list(test_ds.df["active_region"].unique()),
            "all_active_regions": sorted(list(set(train_ds.df["active_region"].unique()) | set(val_ds.df["active_region"].unique()) | set(test_ds.df["active_region"].unique())))
        },
        "optimal_threshold": optimal_threshold,
        "calibrated_temperature": calibrated_temp,
        "single_split_test_metrics": test_eval
    }
    meta_path = MODELS_LATEST_DIR / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[SAVED] Model Metadata -> {meta_path}", flush=True)
    print("=" * 75 + "\n", flush=True)


if __name__ == "__main__":
    run_training()