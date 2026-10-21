import tensorflow as tf

from .baseline import BaselineModel, train_baseline
from .rnn import build_rnn_model, get_model_summary
from .lstm import (
    build_lstm_model,
    build_lstm_1layer,
    build_lstm_2layer,
    build_bilstm_1layer,
    build_bilstm_2layer,
    get_model_summary,
)
from .bilstm import (
    build_bilstm_model,
    build_bilstm_concat,
    build_bilstm_sum,
    build_bilstm_ave,
    get_model_summary,
)

# Re-export get_model_summary from one place
from .rnn import get_model_summary as _get_model_summary


def create_model(
    model_type: str,
    vocab_size: int,
    config: dict,
) -> tf.keras.Model:
    """Factory function to create model from config.
    
    Args:
        model_type: "rnn", "lstm", "bilstm", "baseline"
        vocab_size: Vocabulary size
        config: Model configuration dictionary
        
    Returns:
        Compiled Keras model (or BaselineModel for baseline)
    """
    import tensorflow as tf
    
    common_args = {
        "vocab_size": vocab_size,
        "embedding_dim": config.get("embedding", {}).get("output_dim", 128),
        "hidden_units": config.get("rnn", {}).get("hidden_units", 128),
        "num_classes": config.get("dense", {}).get("units", 3),
        "dropout_rate": config.get("rnn", {}).get("dropout", 0.3),
        "recurrent_dropout": config.get("rnn", {}).get("recurrent_dropout", 0.0),
        "learning_rate": config.get("learning_rate", 0.001),
        "optimizer": config.get("optimizer", "adam"),
        "mask_zero": config.get("embedding", {}).get("mask_zero", True),
    }
    
    if model_type == "rnn":
        return build_rnn_model(**common_args)
    
    elif model_type == "lstm":
        return build_lstm_model(
            **common_args,
            num_layers=config.get("rnn", {}).get("num_layers", 1),
            bidirectional=config.get("rnn", {}).get("bidirectional", False),
        )
    
    elif model_type == "bilstm":
        return build_bilstm_model(
            **common_args,
            num_layers=config.get("rnn", {}).get("num_layers", 1),
            merge_mode=config.get("rnn", {}).get("merge_mode", "concat"),
        )
    
    elif model_type == "baseline":
        from .baseline import BaselineModel
        baseline_config = config.get("baseline", {})
        return BaselineModel(
            max_features=baseline_config.get("tfidf", {}).get("max_features", 10000),
            ngram_range=tuple(baseline_config.get("tfidf", {}).get("ngram_range", [1, 2])),
            min_df=baseline_config.get("tfidf", {}).get("min_df", 2),
            max_df=baseline_config.get("tfidf", {}).get("max_df", 0.95),
            classifier_type=baseline_config.get("classifier", {}).get("type", "logistic_regression"),
            C=baseline_config.get("classifier", {}).get("C", 1.0),
            class_weight=baseline_config.get("classifier", {}).get("class_weight", "balanced"),
            max_iter=baseline_config.get("classifier", {}).get("max_iter", 1000),
            random_state=baseline_config.get("classifier", {}).get("random_state", 42),
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_params(model) -> int:
    """Count trainable parameters."""
    if hasattr(model, "count_params"):
        return model.count_params()
    elif hasattr(model, "pipeline"):
        # Baseline model - check if fitted
        clf = model.pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return sum(p.size for p in clf.coef_.ravel())
        # Not fitted yet - return approximate based on config
        tfidf = model.pipeline.named_steps["tfidf"]
        max_features = tfidf.max_features or 10000
        n_classes = 3
        return max_features * n_classes  # approximate
    return 0


__all__ = [
    "BaselineModel",
    "train_baseline",
    "build_rnn_model",
    "build_lstm_model",
    "build_lstm_1layer",
    "build_lstm_2layer",
    "build_bilstm_1layer",
    "build_bilstm_2layer",
    "build_bilstm_model",
    "build_bilstm_concat",
    "build_bilstm_sum",
    "build_bilstm_ave",
    "create_model",
    "count_params",
    "get_model_summary",
]