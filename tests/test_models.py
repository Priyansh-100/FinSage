import pytest
import numpy as np
import tensorflow as tf

from finsage.models import (
    build_rnn_model,
    build_lstm_model,
    build_bilstm_model,
    create_model,
    BaselineModel,
    count_params,
)


class TestRNNModel:
    """Tests for RNN model."""
    
    def test_build_rnn_model(self):
        model = build_rnn_model(vocab_size=1000, embedding_dim=64, hidden_units=64)
        
        assert isinstance(model, tf.keras.Model)
        assert model.name == "SimpleRNN"
        
        # Check layers
        layer_names = [layer.name for layer in model.layers]
        assert "embedding" in layer_names
        assert "rnn" in layer_names
        assert "output" in layer_names
    
    def test_rnn_forward_pass(self):
        model = build_rnn_model(vocab_size=1000, embedding_dim=64, hidden_units=64)
        
        X = np.random.randint(0, 1000, (2, 20))
        y_pred = model.predict(X, verbose=0)
        
        assert y_pred.shape == (2, 3)
        # Probabilities should sum to 1
        np.testing.assert_allclose(y_pred.sum(axis=1), 1.0, atol=1e-6)
    
    def test_rnn_output_shape(self):
        for batch_size in [1, 4, 16]:
            for seq_len in [10, 50, 100]:
                model = build_rnn_model(vocab_size=1000, embedding_dim=32, hidden_units=32)
                X = np.random.randint(0, 1000, (batch_size, seq_len))
                y_pred = model.predict(X, verbose=0)
                assert y_pred.shape == (batch_size, 3)


class TestLSTMModel:
    """Tests for LSTM model."""
    
    def test_build_lstm_model(self):
        model = build_lstm_model(vocab_size=1000, embedding_dim=64, hidden_units=64)
        
        assert isinstance(model, tf.keras.Model)
        assert model.name == "LSTM"
        
        layer_names = [layer.name for layer in model.layers]
        assert "embedding" in layer_names
        assert "lstm_1" in layer_names
        assert "output" in layer_names
    
    def test_build_bilstm_model(self):
        model = build_lstm_model(vocab_size=1000, embedding_dim=64, hidden_units=64, bidirectional=True)
        
        assert model.name == "BiLSTM"
        layer_names = [layer.name for layer in model.layers]
        assert "bilstm_1" in layer_names
    
    def test_multi_layer_lstm(self):
        model = build_lstm_model(vocab_size=1000, num_layers=2, hidden_units=64)
        
        layer_names = [layer.name for layer in model.layers]
        assert "lstm_1" in layer_names
        assert "lstm_2" in layer_names
        assert "dropout_1" in layer_names  # Inter-layer dropout
    
    def test_lstm_forward_pass(self):
        model = build_lstm_model(vocab_size=1000, embedding_dim=64, hidden_units=64)
        
        X = np.random.randint(0, 1000, (2, 20))
        y_pred = model.predict(X, verbose=0)
        
        assert y_pred.shape == (2, 3)
        np.testing.assert_allclose(y_pred.sum(axis=1), 1.0, atol=1e-6)


class TestBiLSTMModel:
    """Tests for BiLSTM model."""
    
    def test_build_bilstm_concat(self):
        model = build_bilstm_model(vocab_size=1000, merge_mode="concat")
        assert "concat" in model.name
    
    def test_build_bilstm_sum(self):
        model = build_bilstm_model(vocab_size=1000, merge_mode="sum")
        assert "sum" in model.name
        
        # With sum merge, output dim = hidden_units (not 2*hidden_units)
        # This is a key difference from concat
    
    def test_bilstm_forward_pass(self):
        for merge_mode in ["concat", "sum", "ave", "mul"]:
            model = build_bilstm_model(vocab_size=1000, merge_mode=merge_mode)
            X = np.random.randint(0, 1000, (2, 20))
            y_pred = model.predict(X, verbose=0)
            assert y_pred.shape == (2, 3)


class TestModelFactory:
    """Tests for model factory function."""
    
    def test_create_rnn(self):
        config = {
            "embedding": {"output_dim": 64},
            "rnn": {"hidden_units": 64, "dropout": 0.3},
            "dense": {"units": 3},
            "learning_rate": 0.001,
            "optimizer": "adam",
        }
        model = create_model("rnn", vocab_size=1000, config=config)
        assert isinstance(model, tf.keras.Model)
    
    def test_create_lstm(self):
        config = {
            "embedding": {"output_dim": 64},
            "rnn": {"hidden_units": 64, "num_layers": 1, "dropout": 0.3, "bidirectional": False},
            "dense": {"units": 3},
            "learning_rate": 0.001,
            "optimizer": "adam",
        }
        model = create_model("lstm", vocab_size=1000, config=config)
        assert isinstance(model, tf.keras.Model)
    
    def test_create_bilstm(self):
        config = {
            "embedding": {"output_dim": 64},
            "rnn": {"hidden_units": 64, "num_layers": 1, "dropout": 0.3},
            "dense": {"units": 3},
            "learning_rate": 0.001,
            "optimizer": "adam",
        }
        model = create_model("bilstm", vocab_size=1000, config=config)
        assert isinstance(model, tf.keras.Model)
    
    def test_create_baseline(self):
        config = {
            "baseline": {
                "tfidf": {"max_features": 1000, "ngram_range": [1, 2]},
                "classifier": {"type": "logistic_regression", "C": 1.0},
            }
        }
        model = create_model("baseline", vocab_size=1000, config=config)
        assert isinstance(model, BaselineModel)
    
    def test_invalid_model_type(self):
        config = {}
        with pytest.raises(ValueError):
            create_model("invalid", vocab_size=1000, config=config)


class TestCountParams:
    """Tests for parameter counting."""
    
    def test_count_params_rnn(self):
        model = build_rnn_model(vocab_size=1000, embedding_dim=64, hidden_units=64)
        n_params = count_params(model)
        assert n_params > 0
        # Embedding: 1000*64 = 64000
        # RNN: 64*(64+64+1) = 8256 (approx)
        # Dense: 64*3 + 3 = 195
        # Total ~ 72k
        assert 60000 < n_params < 100000
    
    def test_count_params_baseline(self):
        model = BaselineModel(max_features=1000)
        # Not fitted yet, returns approximate based on config
        n_params = count_params(model)
        assert n_params == 1000 * 3  # max_features * n_classes


class TestModelCompilation:
    """Tests for model compilation."""
    
    def test_model_compiled(self):
        model = build_rnn_model(vocab_size=1000)
        
        assert model.optimizer is not None
        assert model.loss == "sparse_categorical_crossentropy"
        # In Keras 3, metrics are in compile_metrics._user_metrics
        compile_metrics = model.metrics[1]  # CompileMetrics object
        user_metrics = compile_metrics._user_metrics
        # Check that accuracy is in user metrics (may be string or metric object)
        metric_names = [m.name if hasattr(m, 'name') else str(m) for m in user_metrics]
        assert any("accuracy" in name.lower() or name.lower() == "acc" for name in metric_names)
    
    def test_custom_optimizer(self):
        model = build_rnn_model(vocab_size=1000, optimizer="rmsprop", learning_rate=0.01)
        assert isinstance(model.optimizer, tf.keras.optimizers.RMSprop)
        assert model.optimizer.learning_rate == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])