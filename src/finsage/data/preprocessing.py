import re
import string
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_special_chars: bool = True,
    remove_numbers: bool = False,
) -> str:
    """Clean and normalize text.
    
    Args:
        text: Input text string
        lowercase: Convert to lowercase
        remove_special_chars: Remove punctuation and special characters
        remove_numbers: Remove numeric digits
        
    Returns:
        Cleaned text string
    """
    if lowercase:
        text = text.lower()
    
    if remove_numbers:
        text = re.sub(r"\d+", "", text)
    
    if remove_special_chars:
        # Keep basic punctuation that might be meaningful in financial text
        # Remove other special chars
        text = re.sub(r"[^\w\s\.\,\;\:\$\%\-\+\(\)]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def build_vocab(
    texts: List[str],
    max_size: int = 20000,
    min_freq: int = 2,
    oov_token: str = "<OOV>",
    pad_token: str = "<PAD>",
) -> Tuple[Tokenizer, Dict[str, int]]:
    """Build vocabulary from texts using Keras Tokenizer.
    
    Args:
        texts: List of text strings
        max_size: Maximum vocabulary size
        min_freq: Minimum word frequency
        oov_token: Out-of-vocabulary token
        pad_token: Padding token
        
    Returns:
        Tuple of (Tokenizer, word_index dict)
    """
    tokenizer = Tokenizer(
        num_words=max_size,
        oov_token=oov_token,
        filters="",  # We handle cleaning separately
        lower=False,  # Already handled
        split=" ",
    )
    
    tokenizer.fit_on_texts(texts)
    
    # Filter by min_freq
    word_counts = tokenizer.word_counts
    filtered_words = {w for w, c in word_counts.items() if c >= min_freq}
    
    # Rebuild word_index with filtered vocabulary
    word_index = {w: i for i, (w, _) in enumerate(tokenizer.word_index.items()) if w in filtered_words}
    word_index = {w: i+2 for i, (w, _) in enumerate(word_index.items())}  # Reserve 0, 1
    word_index[pad_token] = 0
    word_index[oov_token] = 1
    
    tokenizer.word_index = word_index
    tokenizer.index_word = {i: w for w, i in word_index.items()}
    
    print(f"Vocabulary size: {len(word_index)} (max: {max_size}, min_freq: {min_freq})")
    
    return tokenizer, word_index


def encode_sequences(
    tokenizer: Tokenizer,
    texts: List[str],
    max_seq_length: int = 100,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """Encode texts to padded sequences.
    
    Args:
        tokenizer: Fitted Keras Tokenizer
        texts: List of text strings
        max_seq_length: Maximum sequence length
        padding: Padding direction ('pre' or 'post')
        truncating: Truncation direction ('pre' or 'post')
        
    Returns:
        Padded sequences array of shape (n_samples, max_seq_length)
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=max_seq_length,
        padding=padding,
        truncating=truncating,
        value=0,  # pad_token index
    )
    return padded


def preprocess_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    lowercase: bool = True,
    remove_special_chars: bool = True,
    remove_numbers: bool = False,
    max_seq_length: int = 100,
    max_vocab_size: int = 20000,
    min_word_freq: int = 2,
    oov_token: str = "<OOV>",
    pad_token: str = "<PAD>",
) -> Tuple[np.ndarray, np.ndarray, Tokenizer, Dict[str, int]]:
    """Full preprocessing pipeline: clean -> tokenize -> encode -> pad.
    
    Args:
        df: DataFrame with text and label columns
        text_col: Name of text column
        label_col: Name of label column
        lowercase: Convert to lowercase
        remove_special_chars: Remove special characters
        remove_numbers: Remove numbers
        max_seq_length: Maximum sequence length
        max_vocab_size: Maximum vocabulary size
        min_word_freq: Minimum word frequency
        oov_token: OOV token
        pad_token: Padding token
        
    Returns:
        Tuple of (X_padded, y_labels, tokenizer, word_index)
    """
    # Clean texts
    texts = df[text_col].apply(
        lambda x: clean_text(
            x,
            lowercase=lowercase,
            remove_special_chars=remove_special_chars,
            remove_numbers=remove_numbers,
        )
    ).tolist()
    
    labels = df[label_col].values
    
    # Build vocabulary
    tokenizer, word_index = build_vocab(
        texts,
        max_size=max_vocab_size,
        min_freq=min_word_freq,
        oov_token=oov_token,
        pad_token=pad_token,
    )
    
    # Encode and pad
    X = encode_sequences(tokenizer, texts, max_seq_length=max_seq_length)
    
    return X, labels, tokenizer, word_index


def analyze_sequence_lengths(texts: List[str], tokenizer: Tokenizer) -> Dict[str, float]:
    """Analyze sequence length statistics.
    
    Args:
        texts: List of text strings
        tokenizer: Fitted tokenizer
        
    Returns:
        Dictionary with length statistics
    """
    sequences = tokenizer.texts_to_sequences(texts)
    lengths = [len(s) for s in sequences]
    
    return {
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
        "p95": np.percentile(lengths, 95),
        "p99": np.percentile(lengths, 99),
    }


if __name__ == "__main__":
    # Quick test
    sample_texts = [
        "The company reported strong earnings growth.",
        "Stock prices fell sharply after the announcement.",
        "Market remains neutral amid uncertainty.",
    ]
    tokenizer, word_index = build_vocab(sample_texts, max_size=1000)
    X = encode_sequences(tokenizer, sample_texts, max_seq_length=20)
    print("Encoded shape:", X.shape)
    print("Sample encoded:", X[0])
    print("Word index sample:", dict(list(word_index.items())[:10]))