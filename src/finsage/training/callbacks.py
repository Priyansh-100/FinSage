from typing import Dict, Any, Optional, List
import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
import mlflow
import json


class MacroF1Callback(Callback):
    """Custom callback to compute and track macro F1 score during training."""
    
    def __init__(self, validation_data, label_names: Optional[List[str]] = None):
        super().__init__()
        self.validation_data = validation_data
        self.label_names = label_names or ["negative", "neutral", "positive"]
        self.macro_f1s = []
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        X_val, y_val = self.validation_data
        
        # Predict
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = y_pred.argmax(axis=1)
        
        # Compute macro F1
        from sklearn.metrics import f1_score, classification_report
        macro_f1 = f1_score(y_val, y_pred_classes, average="macro")
        self.macro_f1s.append(macro_f1)
        
        logs["val_macro_f1"] = macro_f1
        
        # Per-class F1
        report = classification_report(y_val, y_pred_classes, target_names=self.label_names, output_dict=True)
        for label in self.label_names:
            logs[f"val_f1_{label}"] = report[label]["f1-score"]
        
        print(f" - val_macro_f1: {macro_f1:.4f}")


class MLflowCallback(Callback):
    """MLflow logging callback for Keras training."""
    
    def __init__(
        self,
        experiment_name: str = "finsage-sentiment",
        tracking_uri: str = "file:./mlruns",
        log_model: bool = True,
        log_artifacts: bool = True,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.log_model = log_model
        self.log_artifacts = log_artifacts
        self.params = params or {}
        self.tags = tags or {}
        self.run_id = None
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        self.run = mlflow.start_run(experiment_id=self.experiment_id)
        self.run_id = self.run.info.run_id
        
        # Log params
        if self.params:
            mlflow.log_params(self.params)
        
        # Set tags
        if self.tags:
            mlflow.set_tags(self.tags)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Log metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=epoch)
    
    def on_train_end(self, logs=None):
        # Log final model
        if self.log_model and self.model is not None:
            try:
                mlflow.keras.log_model(self.model, "model")
            except Exception as e:
                print(f"Warning: Could not log model to MLflow: {e}")
        
        # Log artifacts
        if self.log_artifacts:
            # Log config if available
            if hasattr(self, "config_path") and os.path.exists(self.config_path):
                mlflow.log_artifact(self.config_path, "config")
        
        mlflow.end_run()


def create_callbacks(
    config: Dict[str, Any],
    validation_data=None,
    label_names: Optional[List[str]] = None,
    checkpoint_dir: str = "checkpoints",
    logs_dir: str = "logs",
    config_path: Optional[str] = None,
) -> List[Callback]:
    """Create training callbacks from config.
    
    Args:
        config: Training configuration dictionary
        validation_data: Validation data for MacroF1Callback
        label_names: Class label names
        checkpoint_dir: Directory for model checkpoints
        logs_dir: Directory for TensorBoard logs
        config_path: Path to config file for MLflow logging
        
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    cb_config = config.get("callbacks", {})
    
    # Early Stopping
    es_config = cb_config.get("early_stopping", {})
    if es_config.get("enabled", True):
        callbacks.append(EarlyStopping(
            monitor=es_config.get("monitor", "val_macro_f1"),
            mode=es_config.get("mode", "max"),
            patience=es_config.get("patience", 7),
            min_delta=es_config.get("min_delta", 0.001),
            restore_best_weights=es_config.get("restore_best_weights", True),
            verbose=1,
        ))
    
    # Model Checkpoint
    mc_config = cb_config.get("model_checkpoint", {})
    if mc_config.get("enabled", True):
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor=mc_config.get("monitor", "val_macro_f1"),
            mode=mc_config.get("mode", "max"),
            save_best_only=mc_config.get("save_best_only", True),
            save_weights_only=mc_config.get("save_weights_only", False),
            verbose=1,
        ))
    
    # Reduce LR on Plateau
    rl_config = cb_config.get("reduce_lr", {})
    if rl_config.get("enabled", True):
        callbacks.append(ReduceLROnPlateau(
            monitor=rl_config.get("monitor", "val_loss"),
            factor=rl_config.get("factor", 0.5),
            patience=rl_config.get("patience", 3),
            min_lr=rl_config.get("min_lr", 1e-6),
            verbose=1,
        ))
    
    # Macro F1 Callback (requires validation data)
    if validation_data is not None:
        callbacks.append(MacroF1Callback(validation_data, label_names))
    
    # TensorBoard
    os.makedirs(logs_dir, exist_ok=True)
    callbacks.append(TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
    ))
    
    # MLflow
    mlflow_config = config.get("mlflow", {})
    if mlflow_config.get("enabled", True):
        mlflow_cb = MLflowCallback(
            experiment_name=mlflow_config.get("experiment_name", "finsage-sentiment"),
            tracking_uri=mlflow_config.get("tracking_uri", "file:./mlruns"),
            log_model=mlflow_config.get("log_model", True),
            log_artifacts=mlflow_config.get("log_artifacts", True),
            params=config.get("model", {}),
            tags={
                "model_type": config.get("model", {}).get("type", "unknown"),
                "framework": "tensorflow/keras",
            },
        )
        if config_path:
            mlflow_cb.config_path = config_path
        callbacks.append(mlflow_cb)
    
    return callbacks


if __name__ == "__main__":
    # Quick test
    import numpy as np
    X_val = np.random.randint(0, 100, (10, 20))
    y_val = np.random.randint(0, 3, 10)
    
    cb = MacroF1Callback((X_val, y_val))
    print("MacroF1Callback created successfully")