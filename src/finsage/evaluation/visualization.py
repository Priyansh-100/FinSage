from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (6, 5),
    cmap: str = "Blues",
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix array
        label_names: Class labels
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        normalize: Whether to normalize rows
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if label_names is None:
        label_names = ["negative", "neutral", "positive"]
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        fmt = "d"
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training history (loss, accuracy, F1, etc.).
    
    Args:
        history: Keras History.history dict
        metrics: Metrics to plot (default: all)
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        # Separate train/val metrics
        train_metrics = [k for k in history.keys() if not k.startswith("val_")]
        val_metrics = [k for k in history.keys() if k.startswith("val_")]
        metrics = train_metrics + val_metrics
    
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    epochs = range(1, len(next(iter(history.values()))) + 1)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric in history:
            ax.plot(epochs, history[metric], label=metric, marker="o", markersize=3)
            val_key = f"val_{metric}" if not metric.startswith("val_") else metric.replace("val_", "")
            if val_key in history:
                ax.plot(epochs, history[val_key], label=f"val_{val_key}" if not metric.startswith("val_") else val_key, marker="s", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training history to {save_path}")
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot bar chart comparing models.
    
    Args:
        comparison_df: DataFrame from compare_models
        metrics: Metrics to plot
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    
    # Filter available metrics
    available = [m for m in metrics if m in comparison_df.columns]
    
    fig, axes = plt.subplots(1, len(available), figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, metric in enumerate(available):
        ax = axes[i]
        df_sorted = comparison_df.sort_values(metric, ascending=True)
        bars = ax.barh(df_sorted["model"], df_sorted[metric], color="steelblue", edgecolor="white")
        ax.set_xlabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlim(0, min(1.1, df_sorted[metric].max() * 1.2))
        
        # Add value labels
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved model comparison to {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba_dict: Dict[str, np.ndarray],
    label_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curves for multiple models.
    
    Args:
        y_true: True labels
        y_proba_dict: Dict of {model_name: predicted_probas}
        label_names: Class names
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if label_names is None:
        label_names = ["negative", "neutral", "positive"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot for each class (one-vs-rest)
    for class_idx, class_name in enumerate(label_names):
        y_binary = (y_true == class_idx).astype(int)
        
        for model_name, y_proba in y_proba_dict.items():
            if y_proba.shape[1] > class_idx:
                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, class_idx])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{model_name} - {class_name} (AUC={roc_auc:.3f})", alpha=0.8)
    
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved ROC curves to {save_path}")
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba_dict: Dict[str, np.ndarray],
    label_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Precision-Recall curves for multiple models.
    
    Args:
        y_true: True labels
        y_proba_dict: Dict of {model_name: predicted_probas}
        label_names: Class names
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if label_names is None:
        label_names = ["negative", "neutral", "positive"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for class_idx, class_name in enumerate(label_names):
        y_binary = (y_true == class_idx).astype(int)
        
        for model_name, y_proba in y_proba_dict.items():
            if y_proba.shape[1] > class_idx:
                precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, class_idx])
                pr_auc = auc(recall, precision)
                ax.plot(recall, precision, label=f"{model_name} - {class_name} (AUC={pr_auc:.3f})", alpha=0.8)
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved PR curves to {save_path}")
    
    return fig


def save_all_plots(
    results: Dict[str, Any],
    output_dir: str = "results/plots",
) -> Dict[str, str]:
    """Generate and save all standard plots.
    
    Args:
        results: Dictionary with test results
        output_dir: Output directory
        
    Returns:
        Dict of plot names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    
    # Confusion matrix
    if "confusion_matrix" in results:
        fig = plot_confusion_matrix(
            np.array(results["confusion_matrix"]),
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
        )
        plt.close(fig)
        saved["confusion_matrix"] = os.path.join(output_dir, "confusion_matrix.png")
    
    # Training history
    if "history" in results and results["history"]:
        fig = plot_training_history(
            results["history"],
            save_path=os.path.join(output_dir, "training_history.png"),
        )
        plt.close(fig)
        saved["training_history"] = os.path.join(output_dir, "training_history.png")
    
    return saved


if __name__ == "__main__":
    # Quick test
    cm = np.array([[50, 5, 2], [3, 45, 7], [1, 4, 52]])
    fig = plot_confusion_matrix(cm)
    plt.close(fig)
    print("Visualization module loaded successfully")