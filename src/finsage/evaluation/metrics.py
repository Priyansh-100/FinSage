from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        label_names: Class names
        
    Returns:
        Dictionary with all metrics
    """
    if label_names is None:
        label_names = ["negative", "neutral", "positive"]
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    
    # ROC-AUC (if probabilities provided)
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            pass  # Not enough classes in y_true
    
    results = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    
    # Add per-class named metrics
    for i, label in enumerate(label_names):
        results[f"precision_{label}"] = float(per_class_precision[i])
        results[f"recall_{label}"] = float(per_class_recall[i])
        results[f"f1_{label}"] = float(per_class_f1[i])
    
    if roc_auc is not None:
        results["roc_auc_macro"] = roc_auc
    
    return results


def compare_models(
    results_dict: Dict[str, Dict[str, Any]],
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """Compare multiple models' results.
    
    Args:
        results_dict: Dict of {model_name: metrics_dict}
        metric: Primary metric for sorting
        
    Returns:
        Comparison DataFrame
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"model": model_name}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    
    return df


def print_comparison_table(df: pd.DataFrame, metrics: Optional[List[str]] = None) -> None:
    """Pretty print model comparison table.
    
    Args:
        df: Comparison DataFrame from compare_models
        metrics: Metrics to display (default: key metrics)
    """
    if metrics is None:
        metrics = ["model", "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    
    display_df = df[metrics].copy()
    for col in display_df.columns:
        if col != "model" and display_df[col].dtype in ["float64", "float32"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    # Quick test
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2])
    y_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.3, 0.5],
        [0.9, 0.05, 0.05],
        [0.1, 0.2, 0.7],
        [0.15, 0.15, 0.7],
        [0.85, 0.1, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
    ])
    
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print("Metrics:", {k: v for k, v in metrics.items() if isinstance(v, (int, float))})