# File: src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
)


def calculate_metrics(all_preds, all_targets):
    """
    Tính metrics với Youden's J threshold tối ưu từ ROC curve.
    Không dùng threshold cứng 0.5 — sai với BPR/GNN score chưa calibrate.
    """
    y_true = all_targets.astype(int)
    preds  = np.array(all_preds, dtype=np.float64)

    try:
        auc = roc_auc_score(y_true, preds)
    except ValueError:
        auc = 0.5

    # FIX: Youden's J = argmax(tpr - fpr) → threshold tối ưu
    try:
        fpr, tpr, thresholds = roc_curve(y_true, preds)
        best_idx       = int(np.argmax(tpr - fpr))
        best_threshold = float(thresholds[best_idx])
    except ValueError:
        best_threshold = 0.5

    y_pred = (preds >= best_threshold).astype(int)

    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-score":  f1_score(y_true, y_pred, zero_division=0),
        "AUC":       auc,
        "Threshold": best_threshold,
    }


def print_metrics_table(metrics_dict, title="EVALUATION RESULT"):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    for k, v in metrics_dict.items():
        print(f"{k:<12}: {v:.4f}")
    print("=" * 40 + "\n")