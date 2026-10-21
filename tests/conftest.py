"""Pytest configuration and fixtures."""
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing."""
    return [
        "The company reported strong earnings growth.",
        "Stock prices fell sharply after the announcement.",
        "Market remains neutral amid uncertainty.",
        "Profits surged beyond expectations.",
        "Revenue declined due to market conditions.",
    ]


@pytest.fixture(scope="session")
def sample_labels():
    """Sample labels for testing."""
    return np.array([2, 0, 1, 2, 0])  # positive, negative, neutral, positive, negative


@pytest.fixture(scope="session")
def sample_dataframe(sample_texts, sample_labels):
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "text": sample_texts,
        "label": sample_labels,
        "label_name": ["positive", "negative", "neutral", "positive", "negative"],
    })


@pytest.fixture(scope="session")
def balanced_dataframe():
    """Balanced DataFrame for testing."""
    texts = []
    labels = []
    for i in range(30):
        texts.append(f"positive sample {i}")
        labels.append(2)
    for i in range(30):
        texts.append(f"negative sample {i}")
        labels.append(0)
    for i in range(30):
        texts.append(f"neutral sample {i}")
        labels.append(1)
    
    return pd.DataFrame({
        "text": texts,
        "label": labels,
        "label_name": ["positive"] * 30 + ["negative"] * 30 + ["neutral"] * 30,
    })


@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility."""
    import random
    import os
    
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# Configure TensorFlow for testing
tf.config.experimental.enable_op_determinism()
# Disable GPU for consistent testing
tf.config.set_visible_devices([], "GPU")