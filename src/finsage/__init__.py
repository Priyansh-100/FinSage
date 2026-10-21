"""FinSage: Financial News Sentiment Classification Using Deep Learning."""

from .data import (
    load_phrasebank,
    split_dataframe,
    preprocess_splits_no_leakage,
    get_class_weights,
)
from .models import (
    BaselineModel,
    create_model,
    build_rnn_model,
    build_lstm_model,
    build_bilstm_model,
)
from .training import Trainer, train_from_config
from .evaluation import compute_metrics, compare_models
from .experiments import run_model_comparison, run_ablation

__version__ = "0.1.0"

__all__ = [
    "load_phrasebank",
    "split_dataframe",
    "preprocess_splits_no_leakage",
    "get_class_weights",
    "BaselineModel",
    "create_model",
    "build_rnn_model",
    "build_lstm_model",
    "build_bilstm_model",
    "Trainer",
    "train_from_config",
    "compute_metrics",
    "compare_models",
    "run_model_comparison",
    "run_ablation",
]