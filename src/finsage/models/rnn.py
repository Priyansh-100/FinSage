from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


def build_rnn_model(
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_units: int = 128,
    num_classes: int = 3,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.0,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    mask_zero: bool = True,
) -> Model:
    """Build a Simple RNN model for sentiment classification.
    
    Args:
        vocab_size: Vocabulary size (including padding and OOV tokens)
        embedding_dim: Embedding dimension
        hidden_units: RNN hidden units
        num_classes: Number of output classes
        dropout_rate: Dropout rate after RNN
        recurrent_dropout: Recurrent dropout rate
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
    
    # Simple RNN layer
    x = layers.SimpleRNN(
        units=hidden_units,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
        name="rnn",
    )(x)
    
    # Dropout
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    
    # Output layer
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="output",
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SimpleRNN")
    
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


if __name__ == "__main__":
    # Quick test
    model = build_rnn_model(vocab_size=20000, embedding_dim=128, hidden_units=128)
    print(get_model_summary(model))
    
    # Test forward pass
    import numpy as np
    X_test = np.random.randint(0, 20000, (2, 50))
    y_pred = model.predict(X_test, verbose=0)
    print(f"Output shape: {y_pred.shape}")
    print(f"Sum of probs: {y_pred[0].sum():.4f}")