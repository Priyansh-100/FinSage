from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_units: int = 128,
    num_layers: int = 1,
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.0,
    bidirectional: bool = False,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    mask_zero: bool = True,
) -> Model:
    """Build an LSTM model for sentiment classification.
    
    Args:
        vocab_size: Vocabulary size (including padding and OOV tokens)
        embedding_dim: Embedding dimension
        hidden_units: LSTM hidden units per layer
        num_layers: Number of LSTM layers
        num_classes: Number of output classes
        dropout_rate: Dropout rate between layers
        recurrent_dropout: Recurrent dropout rate
        bidirectional: Whether to use Bidirectional LSTM
        learning_rate: Learning rate
        optimizer: Optimizer name
        mask_zero: Whether to mask zero padding
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=(None,), dtype=tf.int32, name="input_sequence")
    
    # Embedding layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=mask_zero,
        name="embedding",
    )(inputs)
    
    # LSTM layers
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        
        lstm_layer = layers.LSTM(
            units=hidden_units,
            dropout=dropout_rate if i < num_layers - 1 else 0.0,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences,
            name=f"lstm_{i+1}",
        )
        
        if bidirectional:
            x = layers.Bidirectional(lstm_layer, name=f"bilstm_{i+1}")(x)
        else:
            x = lstm_layer(x)
        
        # Inter-layer dropout (except after last LSTM layer)
        if i < num_layers - 1 and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)
    
    # Final dropout before output
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout_final")(x)
    
    # Output layer
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="output",
    )(x)
    
    model_name = "BiLSTM" if bidirectional else "LSTM"
    if num_layers > 1:
        model_name = f"{num_layers}Layer_{model_name}"
    
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    # Compile
    opt = _get_optimizer(optimizer, learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    
    return model


def _get_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    """Get optimizer instance."""
    name = name.lower()
    if name == "adam":
        return Adam(learning_rate=learning_rate)
    elif name == "rmsprop":
        return RMSprop(learning_rate=learning_rate)
    elif name == "sgd":
        return SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_model_summary(model: Model) -> str:
    """Get model summary as string."""
    from io import StringIO
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


# Convenience functions for specific architectures
def build_lstm_1layer(vocab_size: int, **kwargs) -> Model:
    """Build single-layer LSTM."""
    return build_lstm_model(vocab_size, num_layers=1, **kwargs)


def build_lstm_2layer(vocab_size: int, **kwargs) -> Model:
    """Build two-layer LSTM."""
    return build_lstm_model(vocab_size, num_layers=2, **kwargs)


def build_bilstm_1layer(vocab_size: int, **kwargs) -> Model:
    """Build single-layer Bidirectional LSTM."""
    return build_lstm_model(vocab_size, num_layers=1, bidirectional=True, **kwargs)


def build_bilstm_2layer(vocab_size: int, **kwargs) -> Model:
    """Build two-layer Bidirectional LSTM."""
    return build_lstm_model(vocab_size, num_layers=2, bidirectional=True, **kwargs)


if __name__ == "__main__":
    # Quick tests
    print("=== LSTM 1-layer ===")
    model = build_lstm_1layer(vocab_size=20000)
    print(get_model_summary(model))
    
    print("\n=== BiLSTM 1-layer ===")
    model = build_bilstm_1layer(vocab_size=20000)
    print(get_model_summary(model))
    
    # Test forward pass
    import numpy as np
    X_test = np.random.randint(0, 20000, (2, 50))
    y_pred = model.predict(X_test, verbose=0)
    print(f"\nOutput shape: {y_pred.shape}")
    print(f"Sum of probs: {y_pred[0].sum():.4f}")