from typing import Dict, Any, Tuple, Optional, Union
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback

from ..models import create_model, count_params
from ..data import (
    load_phrasebank,
    split_dataframe,
    preprocess_splits_no_leakage,
    create_tf_datasets,
    get_class_weights,
)
from .callbacks import create_callbacks
from ..utils.seeding import set_seed, set_global_determinism


class Trainer:
    """Training orchestrator for FinSage models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration.
        
        Args:
            config: Full configuration dictionary (data, model, train)
        """
        self.config = config
        self.data_config = config.get("data", {})
        self.model_config = config.get("model", {})
        self.train_config = config.get("train", {})
        
        # Set seeds for reproducibility
        seed = self.train_config.get("seed", 42)
        deterministic = self.train_config.get("deterministic", True)
        if deterministic:
            set_global_determinism(seed)
        else:
            set_seed(seed, deterministic=False)
        
        # Mixed precision
        if self.train_config.get("mixed_precision", False):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        
        # Output directories
        self.output_dir = self.train_config.get("output_dir", "outputs")
        self.checkpoint_dir = self.train_config.get("checkpoint_dir", "checkpoints")
        self.logs_dir = self.train_config.get("logs_dir", "logs")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Model and data (initialized in prepare())
        self.model = None
        self.tokenizer = None
        self.word_index = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_weights = None
        self.history = None
        self.label_names = ["negative", "neutral", "positive"]
    
    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess data with no leakage.
        
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        print("Loading dataset...")
        df = load_phrasebank(
            variant=self.data_config.get("variant", "50Agree"),
            data_dir=self.data_config.get("data_dir", "data/raw"),
        )
        
        # Split DataFrame FIRST (before any preprocessing)
        train_df, val_df, test_df = split_dataframe(
            df,
            label_col="label",
            train_ratio=self.data_config.get("train_ratio", 0.7),
            val_ratio=self.data_config.get("val_ratio", 0.15),
            test_ratio=self.data_config.get("test_ratio", 0.15),
            seed=self.data_config.get("split_seed", 42),
        )
        
        # Preprocess with NO LEAKAGE (tokenizer fit only on train)
        prep_config = self.data_config.get("preprocessing", {})
        vocab_config = self.data_config.get("vocab", {})
        
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         self.tokenizer, self.word_index) = preprocess_splits_no_leakage(
            train_df, val_df, test_df,
            text_col="text",
            label_col="label",
            max_seq_length=prep_config.get("max_seq_length", 100),
            max_vocab_size=vocab_config.get("max_size", 20000),
            min_word_freq=prep_config.get("min_word_freq", 2),
            oov_token=vocab_config.get("oov_token", "<OOV>"),
            pad_token=vocab_config.get("pad_token", "<PAD>"),
            lowercase=prep_config.get("lowercase", True),
            remove_special_chars=prep_config.get("remove_special_chars", True),
            remove_numbers=prep_config.get("remove_numbers", False),
        )
        
        # Create TF datasets
        batch_size = self.data_config.get("batch_size", 32)
        self.train_ds, self.val_ds, self.test_ds = create_tf_datasets(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=batch_size,
            shuffle_buffer=self.data_config.get("shuffle_buffer", 1000),
            prefetch=self.data_config.get("prefetch", True),
            seed=self.data_config.get("split_seed", 42),
        )
        
        # Compute class weights
        cw_config = self.train_config.get("class_weight", {})
        if cw_config.get("enabled", True):
            method = cw_config.get("method", "auto")
            if method == "auto":
                self.class_weights = get_class_weights(train_df, method="auto")
            elif method == "balanced":
                self.class_weights = get_class_weights(train_df, method="balanced")
            else:
                self.class_weights = method  # Assume dict
            print(f"Class weights: {self.class_weights}")
        
        return self.train_ds, self.val_ds, self.test_ds
    
    def build_model(self) -> Model:
        """Build model from config.
        
        Returns:
            Compiled Keras model
        """
        model_type = self.model_config.get("type", "lstm")
        vocab_size = len(self.word_index) if self.word_index else self.data_config.get("vocab", {}).get("max_size", 20000)
        
        print(f"Building {model_type} model with vocab_size={vocab_size}...")
        
        # Merge train config into model config for optimizer, lr, etc.
        model_config = {**self.model_config, **self.train_config}
        
        self.model = create_model(model_type, vocab_size, model_config)
        
        print(f"Model parameters: {count_params(self.model):,}")
        print(self.model.summary())
        
        return self.model
    
    def train(
        self,
        train_ds: Optional[tf.data.Dataset] = None,
        val_ds: Optional[tf.data.Dataset] = None,
        callbacks: Optional[list] = None,
    ) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            train_ds: Training dataset (uses self.train_ds if None)
            val_ds: Validation dataset (uses self.val_ds if None)
            callbacks: Additional callbacks
            
        Returns:
            Training history
        """
        train_ds = train_ds or self.train_ds
        val_ds = val_ds or self.val_ds
        
        if train_ds is None or val_ds is None:
            raise ValueError("Datasets not prepared. Call prepare_data() first.")
        
        if self.model is None:
            self.build_model()
        
        # Create callbacks
        all_callbacks = create_callbacks(
            self.train_config,
            validation_data=(val_ds.unbatch().batch(len(list(val_ds.unbatch())))).as_numpy_iterator().__next__() if hasattr(val_ds, 'unbatch') else None,
            label_names=self.label_names,
            checkpoint_dir=self.checkpoint_dir,
            logs_dir=self.logs_dir,
        )
        if callbacks:
            all_callbacks.extend(callbacks)
        
        # Training
        epochs = self.train_config.get("epochs", 50)
        
        print(f"Starting training for {epochs} epochs...")
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=all_callbacks,
            class_weight=self.class_weights,
            verbose=1,
        )
        
        return self.history
    
    def evaluate(self, test_ds: Optional[tf.data.Dataset] = None) -> Dict[str, Any]:
        """Evaluate model on test set.
        
        Args:
            test_ds: Test dataset (uses self.test_ds if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        test_ds = test_ds or self.test_ds
        
        if test_ds is None:
            raise ValueError("Test dataset not available. Call prepare_data() first.")
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Evaluating on test set...")
        results = self.model.evaluate(test_ds, verbose=1, return_dict=True)
        
        # Get predictions for detailed metrics
        y_true = []
        y_pred = []
        for X_batch, y_batch in test_ds.unbatch().batch(1000):
            preds = self.model.predict(X_batch, verbose=0)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.argmax(axis=1))
        
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        detailed_results = {
            "test_loss": results.get("loss"),
            "test_accuracy": results.get("accuracy"),
            "test_precision": results.get("precision"),
            "test_recall": results.get("recall"),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "per_class_precision": precision_score(y_true, y_pred, average=None).tolist(),
            "per_class_recall": recall_score(y_true, y_pred, average=None).tolist(),
            "per_class_f1": f1_score(y_true, y_pred, average=None).tolist(),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.label_names, output_dict=True
            ),
        }
        
        # Add per-class metrics to results
        for i, label in enumerate(self.label_names):
            detailed_results[f"f1_{label}"] = detailed_results["per_class_f1"][i]
            detailed_results[f"precision_{label}"] = detailed_results["per_class_precision"][i]
            detailed_results[f"recall_{label}"] = detailed_results["per_class_recall"][i]
        
        return detailed_results
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save model and tokenizer.
        
        Args:
            path: Save directory (defaults to output_dir)
            
        Returns:
            Path to saved model
        """
        if path is None:
            path = os.path.join(self.output_dir, "model")
        
        os.makedirs(path, exist_ok=True)
        
        # Save Keras model
        model_path = os.path.join(path, "model.keras")
        self.model.save(model_path)
        
        # Save tokenizer
        import pickle
        tokenizer_path = os.path.join(path, "tokenizer.pkl")
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        
        # Save config
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Save word_index
        word_index_path = os.path.join(path, "word_index.json")
        with open(word_index_path, "w") as f:
            json.dump(self.word_index, f)
        
        print(f"Model saved to {path}")
        return path
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline: data -> model -> train -> evaluate.
        
        Returns:
            Dictionary with all results
        """
        self.prepare_data()
        self.build_model()
        self.train()
        results = self.evaluate()
        self.save_model()
        
        return {
            "history": self.history.history if self.history else None,
            "test_results": results,
            "model_path": os.path.join(self.output_dir, "model"),
        }


def train_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to run full pipeline from config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Training results
    """
    trainer = Trainer(config)
    return trainer.run_full_pipeline()


if __name__ == "__main__":
    # Quick test with minimal config
    test_config = {
        "data": {
            "variant": "50Agree",
            "data_dir": "data/raw",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "split_seed": 42,
            "preprocessing": {"max_seq_length": 50, "min_word_freq": 1},
            "vocab": {"max_size": 1000},
            "batch_size": 16,
        },
        "model": {
            "type": "lstm",
            "embedding": {"output_dim": 64, "mask_zero": True},
            "rnn": {"hidden_units": 64, "num_layers": 1, "dropout": 0.3},
            "dense": {"units": 3},
        },
        "train": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "seed": 42,
            "deterministic": True,
            "class_weight": {"enabled": True, "method": "auto"},
            "callbacks": {
                "early_stopping": {"enabled": True, "patience": 3},
                "model_checkpoint": {"enabled": True},
            },
            "mlflow": {"enabled": False},
        },
    }
    
    # Note: This will fail without actual data, but shows usage
    print("Trainer module loaded successfully")
    print("Config keys:", test_config.keys())