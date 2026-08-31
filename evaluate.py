"""
📊 Space-Weather Evaluation & Verification Engine
Features:
  1. Standard Scientific Skill Scores (TSS, HSS, Precision, Recall, Specificity, FPR, F1)
  2. Multi-Class Macro/Weighted F1 & ROC-AUC / PR-AUC (with honest N/A handling)
  3. Continuous Log-Peak Flux Regression Metrics (MAE, RMSE, R²)
  4. Validation-Driven Optimal Decision Threshold Tuning (arg max TSS)
  5. Leave-One-Region-Out Cross-Validation (LORO-CV) Engine across all NOAA Active Regions
  6. CV Persistence & Aggregation (models/latest/cv_results.json)
Zero fabrication: If a metric cannot be calculated due to single-class subsets, returns honest N/A.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from config import (
    BASE_DIR,
    MODELS_LATEST_DIR,
    CATALOGS_DIR,
    ALL_ACTIVE_REGIONS,
    BATCH_SIZE,
    SEQ_LENGTH,
    RANDOM_SEED
)


def find_optimal_threshold(y_true, y_probs, min_th=0.10, max_th=0.90, step=0.02):
    """
    Sweeps probability thresholds to find the threshold that maximizes True Skill Statistic (TSS = TPR - FPR).
    """
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.50

    best_threshold = 0.50
    best_tss = -2.0

    for th in np.arange(min_th, max_th + step, step):
        y_pred = (y_probs >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tss = tpr - fpr

        if tss > best_tss:
            best_tss = tss
            best_threshold = round(float(th), 3)

    return best_threshold


def compute_space_weather_skill_scores(y_true, y_pred_prob, threshold=0.5):
    """
    Computes genuine scientific skill scores for binary solar flare forecasting (>= M1.0).
    """
    y_true = np.array(y_true, dtype=int)
    y_pred_prob = np.array(y_pred_prob, dtype=float)
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Confusion matrix components: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity / True Positive Rate (Recall)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # False Positive Rate (False Alarm Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Specificity / True Negative Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Miss Rate (False Negative Rate)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 Score
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    # True Skill Statistic (TSS): TSS = TPR - FPR
    tss = tpr - fpr

    # Heidke Skill Score (HSS)
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = numerator / denominator if denominator > 0 else 0.0

    # ROC-AUC & PR-AUC with honest N/A handling
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = round(float(roc_auc_score(y_true, y_pred_prob)), 4)
            pr_auc = round(float(average_precision_score(y_true, y_pred_prob)), 4)
        except Exception:
            roc_auc = "N/A"
            pr_auc = "N/A"
    else:
        roc_auc = "N/A (Single class present)"
        pr_auc = "N/A (Single class present)"

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "recall_tpr": round(float(tpr), 4),
        "precision": round(float(precision), 4),
        "specificity": round(float(tnr), 4),
        "false_alarm_rate_fpr": round(float(fpr), 4),
        "miss_rate_fnr": round(float(fnr), 4),
        "f1_score": round(float(f1), 4),
        "true_skill_statistic_tss": round(float(tss), 4),
        "heidke_skill_score_hss": round(float(hss), 4),
        "decision_threshold_used": round(float(threshold), 3),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def evaluate_model_on_dataset(model, dataset, device="cpu", batch_size=4, threshold=0.5):
    """
    Evaluates multi-task forecasting model on a dataset using calibrated logits and chosen threshold.
    """
    if len(dataset) == 0:
        return {
            "status": "NO_DATA",
            "message": "Dataset is empty."
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_bin_true = []
    all_bin_probs = []
    all_multi_true = []
    all_multi_preds = []
    all_flux_true = []
    all_flux_preds = []

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            preds = model(sequences, return_all_heads=True)

            bin_probs = torch.softmax(preds["calibrated_binary_logits"], dim=1)[:, 1].cpu().numpy()
            multi_preds = torch.argmax(preds["multiclass_logits"], dim=1).cpu().numpy()
            flux_preds = preds["log_flux_pred"].cpu().numpy()

            all_bin_true.extend(targets["binary_label"].numpy())
            all_bin_probs.extend(bin_probs)
            all_multi_true.extend(targets["multiclass_label"].numpy())
            all_multi_preds.extend(multi_preds)
            all_flux_true.extend(targets["log_flux"].numpy())
            all_flux_preds.extend(flux_preds)

    binary_metrics = compute_space_weather_skill_scores(all_bin_true, all_bin_probs, threshold=threshold)

    # Multi-class evaluation
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_multi_true, all_multi_preds, average='macro', zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_multi_true, all_multi_preds, average='weighted', zero_division=0
    )

    # Flux Regression Metrics
    flux_arr_true = np.array(all_flux_true)
    flux_arr_pred = np.array(all_flux_preds)
    mae = float(mean_absolute_error(flux_arr_true, flux_arr_pred))
    rmse = float(np.sqrt(mean_squared_error(flux_arr_true, flux_arr_pred)))
    try:
        r2 = float(r2_score(flux_arr_true, flux_arr_pred)) if len(np.unique(flux_arr_true)) > 1 else "N/A"
    except Exception:
        r2 = "N/A"

    return {
        "binary_evaluation_24_48h": binary_metrics,
        "multiclass_evaluation": {
            "macro_f1": round(float(f1_macro), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "classes_evaluated": ["Quiet/B", "C-Class", "M-Class", "X-Class"]
        },
        "flux_regression_metrics": {
            "log10_mae": round(mae, 4),
            "log10_rmse": round(rmse, 4),
            "r2_score": round(float(r2), 4) if isinstance(r2, float) else r2
        },
        "total_test_sequences": len(all_bin_true)
    }


def run_leave_one_region_out_cv(num_epochs=5, lr=0.001, batch_size=32, device=None):
    """
    Executes Leave-One-Region-Out Cross-Validation (LORO-CV) across all NOAA active regions.
    For each fold:
      1. Isolates held-out Active Region R_i as the test set.
      2. Trains SolarFlarePredictor on all remaining active regions (R != R_i).
      3. Uses class-weighted binary loss and balanced multi-task regression.
      4. Fits probability temperature scaling calibrator and tunes threshold on training fold.
      5. Evaluates on held-out region R_i and logs all metrics.
    Computes cross-fold aggregate mean ± std and saves to models/latest/cv_results.json.
    """
    from dataset import SolarSequenceDataset
    from model import SolarFlarePredictor

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_csv = CATALOGS_DIR / "sequence_labels.csv"
    if not labels_csv.exists():
        labels_csv = BASE_DIR / "sequence_labels.csv"

    if not labels_csv.exists():
        raise FileNotFoundError("sequence_labels.csv not found. Run build_labels.py first.")

    labels_df = pd.read_csv(labels_csv)
    regions = sorted(labels_df["active_region"].unique())
    num_folds = len(regions)

    print("=" * 80, flush=True)
    print(f"LEAVE-ONE-REGION-OUT CROSS-VALIDATION (LORO-CV) | {num_folds} Active Region Folds", flush=True)
    print(f"Active Regions Pool: {regions}", flush=True)
    print("Pre-warming dataset tensor cache in RAM...", flush=True)
    full_ds = SolarSequenceDataset(split_df=labels_df)
    for i in range(len(full_ds)):
        _ = full_ds[i]
    print(f"Cached {len(full_ds)} sequence tensors in memory. Starting LORO-CV folds...", flush=True)
    print("=" * 80, flush=True)

    per_fold_results = {}
    fold_tss_list = []
    fold_hss_list = []
    fold_f1_list = []
    fold_recall_list = []
    fold_prec_list = []
    fold_spec_list = []
    fold_fpr_list = []
    fold_roc_list = []
    fold_flux_mae_list = []
    fold_flux_r2_list = []

    for fold_idx, held_out_ar in enumerate(regions, start=1):
        print(f"\n[Fold {fold_idx:02d}/{num_folds:02d}] Held-Out Active Region: {held_out_ar}", flush=True)
        
        train_df = labels_df[labels_df["active_region"] != held_out_ar].copy().reset_index(drop=True)
        test_df = labels_df[labels_df["active_region"] == held_out_ar].copy().reset_index(drop=True)

        train_ds = SolarSequenceDataset(split_df=train_df)
        test_ds = SolarSequenceDataset(split_df=test_df)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Dynamic Inverse Class Frequency Weights for Binary Loss
        bin_counts = train_df["binary_target_MX_24_48h"].value_counts().to_dict()
        n_total = len(train_df)
        n_0 = bin_counts.get(0, 1)
        n_1 = bin_counts.get(1, 1)
        w0 = n_total / (2.0 * max(n_0, 1))
        w1 = n_total / (2.0 * max(n_1, 1))
        bin_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

        # Multiclass weights
        mc_counts = train_df["multiclass_target"].value_counts().to_dict()
        mc_w = [n_total / (4.0 * max(mc_counts.get(c, 1), 1)) for c in range(4)]
        multi_weights = torch.tensor(mc_w, dtype=torch.float32).to(device)

        # Initialize fresh fold model
        torch.manual_seed(RANDOM_SEED + fold_idx)
        model = SolarFlarePredictor(in_channels=4, hidden_dim=32).to(device)

        criterion_binary = nn.CrossEntropyLoss(weight=bin_weights)
        criterion_multiclass = nn.CrossEntropyLoss(weight=multi_weights)
        criterion_flux = nn.SmoothL1Loss(beta=0.5)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for sequences, targets in train_loader:
                sequences = sequences.to(device)
                bin_tgts = targets["binary_label"].to(device)
                mc_tgts = targets["multiclass_label"].to(device)
                flx_tgts = targets["log_flux"].to(device)

                optimizer.zero_grad()
                preds = model(sequences, return_all_heads=True)

                l_bin = criterion_binary(preds["binary_logits"], bin_tgts)
                l_mc = criterion_multiclass(preds["multiclass_logits"], mc_tgts)
                l_flx = criterion_flux(preds["log_flux_pred"], flx_tgts)

                loss = (1.0 * l_bin) + (0.5 * l_mc) + (0.5 * l_flx)
                loss.backward()
                optimizer.step()

        # Fit probability temperature calibrator on training fold
        train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        train_logits_list, train_labels_list = [], []
        model.eval()
        with torch.no_grad():
            for seqs, tgts in train_eval_loader:
                seqs = seqs.to(device)
                preds = model(seqs, return_all_heads=True)
                train_logits_list.append(preds["binary_logits"])
                train_labels_list.append(tgts["binary_label"].to(device))

        if train_logits_list:
            cat_logits = torch.cat(train_logits_list, dim=0)
            cat_labels = torch.cat(train_labels_list, dim=0)
            model.calibrator.fit(cat_logits, cat_labels)
            
            with torch.no_grad():
                scaled_logits = model.calibrator(cat_logits)
                probs = torch.softmax(scaled_logits, dim=1)[:, 1].cpu().numpy()
            opt_th = find_optimal_threshold(cat_labels.cpu().numpy(), probs)
        else:
            opt_th = 0.50

        # Evaluate on held-out region
        fold_eval = evaluate_model_on_dataset(model, test_ds, device=device, threshold=opt_th)
        fold_bin = fold_eval.get("binary_evaluation_24_48h", {})
        fold_flux = fold_eval.get("flux_regression_metrics", {})
        fold_mc = fold_eval.get("multiclass_evaluation", {})

        per_fold_results[held_out_ar] = {
            "held_out_ar": held_out_ar,
            "test_sequences": len(test_df),
            "optimal_threshold": opt_th,
            "calibrated_temperature": round(float(model.calibrator.temperature.item()), 3),
            "binary_metrics": fold_bin,
            "multiclass_macro_f1": fold_mc.get("macro_f1", 0.0),
            "flux_regression": fold_flux
        }

        tss_val = fold_bin.get("true_skill_statistic_tss", 0.0)
        hss_val = fold_bin.get("heidke_skill_score_hss", 0.0)
        f1_val = fold_bin.get("f1_score", 0.0)
        rec_val = fold_bin.get("recall_tpr", 0.0)
        prec_val = fold_bin.get("precision", 0.0)
        spec_val = fold_bin.get("specificity", 0.0)
        fpr_val = fold_bin.get("false_alarm_rate_fpr", 0.0)
        mae_val = fold_flux.get("log10_mae", 0.0)

        fold_tss_list.append(tss_val)
        fold_hss_list.append(hss_val)
        fold_f1_list.append(f1_val)
        fold_recall_list.append(rec_val)
        fold_prec_list.append(prec_val)
        fold_spec_list.append(spec_val)
        fold_fpr_list.append(fpr_val)
        fold_flux_mae_list.append(mae_val)

        if isinstance(fold_bin.get("roc_auc"), (int, float)):
            fold_roc_list.append(fold_bin["roc_auc"])
        if isinstance(fold_flux.get("r2_score"), (int, float)):
            fold_flux_r2_list.append(fold_flux["r2_score"])

        print(f"  -> Fold Metrics: TSS={tss_val:.3f} | HSS={hss_val:.3f} | F1={f1_val:.3f} | Recall={rec_val:.3f} | Specificity={spec_val:.3f} | Flux MAE={mae_val:.3f}")

    # Compute aggregate statistics (mean ± std)
    def _agg(lst):
        arr = np.array(lst)
        return {"mean": round(float(np.mean(arr)), 4), "std": round(float(np.std(arr)), 4)}

    aggregate_summary = {
        "num_folds": num_folds,
        "true_skill_statistic_tss": _agg(fold_tss_list),
        "heidke_skill_score_hss": _agg(fold_hss_list),
        "f1_score": _agg(fold_f1_list),
        "recall_tpr": _agg(fold_recall_list),
        "precision": _agg(fold_prec_list),
        "specificity_tnr": _agg(fold_spec_list),
        "false_alarm_rate_fpr": _agg(fold_fpr_list),
        "roc_auc": _agg(fold_roc_list) if fold_roc_list else {"mean": "N/A", "std": "N/A"},
        "flux_mae": _agg(fold_flux_mae_list),
        "flux_r2": _agg(fold_flux_r2_list) if fold_flux_r2_list else {"mean": "N/A", "std": "N/A"}
    }

    cv_payload = {
        "evaluation_protocol": "Leave-One-Region-Out Cross-Validation (LORO-CV)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_active_regions_evaluated": num_folds,
        "aggregate_summary": aggregate_summary,
        "per_fold_breakdown": per_fold_results
    }

    MODELS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    cv_output_path = MODELS_LATEST_DIR / "cv_results.json"
    with open(cv_output_path, "w", encoding="utf-8") as f:
        json.dump(cv_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("LEAVE-ONE-REGION-OUT CROSS-VALIDATION SUMMARY RESULTS")
    print("=" * 80)
    print(f"Total Folds:                   {num_folds} NOAA Active Regions")
    print(f"True Skill Statistic (TSS):    {aggregate_summary['true_skill_statistic_tss']['mean']:.4f} ± {aggregate_summary['true_skill_statistic_tss']['std']:.4f}")
    print(f"Heidke Skill Score (HSS):      {aggregate_summary['heidke_skill_score_hss']['mean']:.4f} ± {aggregate_summary['heidke_skill_score_hss']['std']:.4f}")
    print(f"24-48h Flare Recall (TPR):     {aggregate_summary['recall_tpr']['mean']*100:.1f}% ± {aggregate_summary['recall_tpr']['std']*100:.1f}%")
    print(f"24-48h Flare Precision:        {aggregate_summary['precision']['mean']*100:.1f}% ± {aggregate_summary['precision']['std']*100:.1f}%")
    print(f"24-48h Flare Specificity:      {aggregate_summary['specificity_tnr']['mean']*100:.1f}% ± {aggregate_summary['specificity_tnr']['std']*100:.1f}%")
    print(f"24-48h False Alarm Rate (FPR): {aggregate_summary['false_alarm_rate_fpr']['mean']*100:.1f}% ± {aggregate_summary['false_alarm_rate_fpr']['std']*100:.1f}%")
    print(f"24-48h Flare F1-Score:         {aggregate_summary['f1_score']['mean']:.4f} ± {aggregate_summary['f1_score']['std']:.4f}")
    print(f"Peak Flux MAE (Log10 W/m²):    {aggregate_summary['flux_mae']['mean']:.4f} ± {aggregate_summary['flux_mae']['std']:.4f}")
    print(f"[SAVED] LORO-CV Results -> {cv_output_path}")
    print("=" * 80 + "\n")

    return cv_payload


if __name__ == "__main__":
    import sys
    if "--cv" in sys.argv:
        run_leave_one_region_out_cv()
    else:
        # Quick test of skill score calculation
        test_scores = compute_space_weather_skill_scores([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2])
        print("Skill Scores Engine Smoke Test:", test_scores)
