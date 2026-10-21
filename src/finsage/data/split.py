from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/validation/test split with no data leakage.
    
    Splits are performed on raw indices BEFORE any feature engineering
    to prevent data leakage.
    
    Args:
        X: Feature array
        y: Label array
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=train_ratio,
        stratify=y,
        random_state=seed,
    )
    
    # Second split: val vs test from remaining
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_size,
        stratify=y_temp,
        random_state=seed,
    )
    
    print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train class dist: {np.bincount(y_train)}")
    print(f"Val class dist: {np.bincount(y_val)}")
    print(f"Test class dist: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle_buffer: int = 1000,
    prefetch: bool = True,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create TensorFlow datasets from arrays.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size
        shuffle_buffer: Shuffle buffer size for training
        prefetch: Whether to prefetch
        seed: Random seed
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    def _to_dataset(X, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_ds = _to_dataset(X_train, y_train, shuffle=True)
    val_ds = _to_dataset(X_val, y_val, shuffle=False)
    test_ds = _to_dataset(X_test, y_test, shuffle=False)
    
    return train_ds, val_ds, test_ds


def split_dataframe(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split on DataFrame (use BEFORE any preprocessing).
    
    This is the recommended approach to prevent data leakage:
    1. Split raw DataFrame first
    2. Then apply preprocessing separately to each split
    3. Fit tokenizer ONLY on training data
    
    Args:
        df: Input DataFrame
        label_col: Name of label column
        train_ratio: Training ratio
        val_ratio: Validation ratio
        test_ratio: Test ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df[label_col],
        random_state=seed,
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df[label_col],
        random_state=seed,
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"DataFrame split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def preprocess_splits_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    max_seq_length: int = 100,
    max_vocab_size: int = 20000,
    min_word_freq: int = 2,
    oov_token: str = "<OOV>",
    pad_token: str = "<PAD>",
    lowercase: bool = True,
    remove_special_chars: bool = True,
    remove_numbers: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    Any, Dict[str, int]
]:
    """Preprocess train/val/test splits with NO DATA LEAKAGE.
    
    Key principle: Tokenizer is fitted ONLY on training data.
    Validation and test data are transformed using the training vocabulary.
    
    Args:
        train_df, val_df, test_df: Pre-split DataFrames
        text_col: Text column name
        label_col: Label column name
        max_seq_length: Maximum sequence length
        max_vocab_size: Maximum vocabulary size
        min_word_freq: Minimum word frequency
        oov_token: OOV token
        pad_token: Padding token
        lowercase: Lowercase text
        remove_special_chars: Remove special chars
        remove_numbers: Remove numbers
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, word_index)
    """
    from .preprocessing import clean_text, build_vocab, encode_sequences
    
    # Clean texts
    def clean_series(series):
        return series.apply(
            lambda x: clean_text(
                x,
                lowercase=lowercase,
                remove_special_chars=remove_special_chars,
                remove_numbers=remove_numbers,
            )
        )
    
    train_texts = clean_series(train_df[text_col]).tolist()
    val_texts = clean_series(val_df[text_col]).tolist()
    test_texts = clean_series(test_df[text_col]).tolist()
    
    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values
    
    # Build vocabulary ONLY on training data
    tokenizer, word_index = build_vocab(
        train_texts,
        max_size=max_vocab_size,
        min_freq=min_word_freq,
        oov_token=oov_token,
        pad_token=pad_token,
    )
    
    # Encode all splits using training vocabulary
    X_train = encode_sequences(tokenizer, train_texts, max_seq_length=max_seq_length)
    X_val = encode_sequences(tokenizer, val_texts, max_seq_length=max_seq_length)
    X_test = encode_sequences(tokenizer, test_texts, max_seq_length=max_seq_length)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, word_index


if __name__ == "__main__":
    # Quick test
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 3, 1000)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    print(f"Shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")