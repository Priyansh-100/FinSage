import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock

from finsage.training import create_callbacks, Trainer
from finsage.data import split_dataframe


class TestCreateCallbacks:
    """Tests for callback creation."""
    
    def test_create_callbacks_basic(self):
        config = {
            "callbacks": {
                "early_stopping": {"enabled": True, "patience": 5},
                "model_checkpoint": {"enabled": True},
                "reduce_lr": {"enabled": True},
            },
            "mlflow": {"enabled": False},
        }
        
        callbacks = create_callbacks(config)
        
        # Should have at least EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
        assert len(callbacks) >= 4
        
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert "EarlyStopping" in callback_types
        assert "ModelCheckpoint" in callback_types
        assert "ReduceLROnPlateau" in callback_types
        assert "TensorBoard" in callback_types
    
    def test_callbacks_disabled(self):
        config = {
            "callbacks": {
                "early_stopping": {"enabled": False},
                "model_checkpoint": {"enabled": False},
                "reduce_lr": {"enabled": False},
            },
            "mlflow": {"enabled": False},
        }
        
        callbacks = create_callbacks(config)
        
        # Only TensorBoard should remain
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert "TensorBoard" in callback_types
        assert "EarlyStopping" not in callback_types
    
    def test_mlflow_callback(self):
        config = {
            "callbacks": {},
            "mlflow": {"enabled": True, "experiment_name": "test"},
            "model": {"type": "lstm"},
        }
        
        with patch("finsage.training.callbacks.mlflow") as mock_mlflow:
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "1"
            
            callbacks = create_callbacks(config)
            
            # Check MLflowCallback is in list
            callback_types = [type(cb).__name__ for cb in callbacks]
            assert "MLflowCallback" in callback_types


class TestTrainer:
    """Tests for Trainer class."""
    
    @pytest.fixture
    def minimal_config(self):
        return {
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
                "embedding": {"output_dim": 32, "mask_zero": True},
                "rnn": {"hidden_units": 32, "num_layers": 1, "dropout": 0.3},
                "dense": {"units": 3},
            },
            "train": {
                "epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "seed": 42,
                "deterministic": True,
                "class_weight": {"enabled": False},
                "callbacks": {
                    "early_stopping": {"enabled": True, "patience": 3},
                    "model_checkpoint": {"enabled": True},
                },
                "mlflow": {"enabled": False},
                "output_dir": "test_outputs",
                "checkpoint_dir": "test_checkpoints",
                "logs_dir": "test_logs",
            },
        }
    
    def test_trainer_init(self, minimal_config):
        trainer = Trainer(minimal_config)
        
        assert trainer.config == minimal_config
        assert trainer.output_dir == "test_outputs"
    
    @patch("finsage.training.trainer.load_phrasebank")
    def test_prepare_data(self, mock_load, minimal_config):
        # Mock data
        df = pd.DataFrame({
            "text": [f"sample text {i}" for i in range(100)],
            "label": np.random.randint(0, 3, 100),
            "label_name": ["neutral"] * 100,
        })
        mock_load.return_value = df
        
        trainer = Trainer(minimal_config)
        train_ds, val_ds, test_ds = trainer.prepare_data()
        
        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None
        assert trainer.tokenizer is not None
        assert trainer.word_index is not None
    
    def test_build_model(self, minimal_config):
        # Need to mock prepare_data first to get vocab
        with patch("finsage.training.trainer.load_phrasebank") as mock_load:
            df = pd.DataFrame({
                "text": [f"sample text {i}" for i in range(100)],
                "label": np.random.randint(0, 3, 100),
            })
            mock_load.return_value = df
            
            trainer = Trainer(minimal_config)
            trainer.prepare_data()
            model = trainer.build_model()
            
            assert model is not None
            assert isinstance(model, tf.keras.Model)


class TestSplitDataFrame:
    """Tests for DataFrame splitting in training context."""
    
    def test_split_preserves_distribution(self):
        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(300)],
            "label": [0] * 100 + [1] * 100 + [2] * 100,
        })
        
        train_df, val_df, test_df = split_dataframe(
            df, label_col="label", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        # Check each split has roughly balanced classes
        for split_df in [train_df, val_df, test_df]:
            counts = split_df["label"].value_counts()
            # Each class should be represented
            assert len(counts) == 3
            # Proportions should be roughly equal
            proportions = counts / len(split_df)
            assert all(0.25 < p < 0.4 for p in proportions)


class TestClassWeights:
    """Tests for class weight computation."""
    
    def test_balanced_weights(self):
        from finsage.data import get_class_weights
        
        df = pd.DataFrame({"label": [0] * 100 + [1] * 50 + [2] * 25})
        weights = get_class_weights(df, method="balanced")
        
        # Class 0 (most frequent) should have lowest weight
        # Class 2 (least frequent) should have highest weight
        assert weights[0] < weights[1] < weights[2]
    
    def test_auto_weights(self):
        from finsage.data import get_class_weights
        
        df = pd.DataFrame({"label": [0] * 100 + [1] * 50 + [2] * 25})
        weights = get_class_weights(df, method="auto")
        
        # Inverse frequency weighting
        assert weights[0] < weights[1] < weights[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])