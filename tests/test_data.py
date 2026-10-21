import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from finsage.data import (
    clean_text,
    build_vocab,
    encode_sequences,
    stratified_split,
    split_dataframe,
    preprocess_splits_no_leakage,
    LABEL_MAP,
)


class TestCleanText:
    """Tests for text cleaning function."""
    
    def test_lowercase(self):
        text = "HELLO WORLD"
        result = clean_text(text, lowercase=True)
        assert result == "hello world"
    
    def test_remove_special_chars(self):
        text = "Hello! @#$% World"
        result = clean_text(text, remove_special_chars=True)
        assert "!" not in result
        assert "@" not in result
    
    def test_remove_numbers(self):
        text = "Stock price $100"
        result = clean_text(text, remove_numbers=True)
        assert "100" not in result
    
    def test_whitespace_normalization(self):
        text = "Hello    World\t\n"
        result = clean_text(text)
        assert result == "hello world"


class TestBuildVocab:
    """Tests for vocabulary building."""
    
    def test_build_vocab_basic(self):
        texts = ["hello world", "hello there", "world peace"]
        tokenizer, word_index = build_vocab(texts, max_size=1000, min_freq=1)
        
        assert "<PAD>" in word_index
        assert "<OOV>" in word_index
        assert word_index["<PAD>"] == 0
        assert word_index["<OOV>"] == 1
        assert "hello" in word_index
        assert "world" in word_index
    
    def test_min_freq_filtering(self):
        texts = ["hello world", "hello there", "rare word"]
        tokenizer, word_index = build_vocab(texts, max_size=1000, min_freq=2)
        
        # "rare" and "word" appear once, should be filtered
        assert "rare" not in word_index or word_index.get("rare", 1) == 1  # OOV
        assert "hello" in word_index
    
    def test_oov_handling(self):
        texts = ["hello world"]
        tokenizer, word_index = build_vocab(texts, max_size=1000)
        
        # Unknown word should map to OOV
        sequences = tokenizer.texts_to_sequences(["unknown word"])
        assert sequences[0][0] == 1  # OOV index


class TestEncodeSequences:
    """Tests for sequence encoding."""
    
    def test_padding_and_truncating(self):
        texts = ["hello world", "this is a longer sentence here"]
        tokenizer, _ = build_vocab(texts, max_size=1000)
        
        # Test padding
        X = encode_sequences(tokenizer, texts, max_seq_length=10, padding="post")
        assert X.shape == (2, 10)
        
        # Test truncating
        X = encode_sequences(tokenizer, texts, max_seq_length=3, truncating="post")
        assert X.shape == (2, 3)


class TestStratifiedSplit:
    """Tests for stratified splitting."""
    
    def test_split_ratios(self):
        X = np.random.rand(1000, 10)
        y = np.random.randint(0, 3, 1000)
        
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        assert len(X_train) == 700
        assert len(X_val) == 150
        assert len(X_test) == 150
    
    def test_stratification(self):
        # Create imbalanced data
        X = np.random.rand(1000, 10)
        y = np.array([0] * 500 + [1] * 300 + [2] * 200)
        
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        
        # Check class proportions are preserved
        for split_y in [y_train, y_val, y_test]:
            props = np.bincount(split_y) / len(split_y)
            original_props = np.bincount(y) / len(y)
            # Should be close (within 5%)
            assert np.allclose(props, original_props, atol=0.05)
    
    def test_reproducibility(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 3, 100)
        
        splits1 = stratified_split(X, y, seed=42)
        splits2 = stratified_split(X, y, seed=42)
        
        for a, b in zip(splits1, splits2):
            np.testing.assert_array_equal(a, b)


class TestSplitDataFrame:
    """Tests for DataFrame splitting."""
    
    def test_split_dataframe(self):
        df = pd.DataFrame({
            "text": [f"sample {i}" for i in range(100)],
            "label": np.random.randint(0, 3, 100),
        })
        
        train_df, val_df, test_df = split_dataframe(
            df, label_col="label", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        assert len(train_df) == 70
        assert len(val_df) == 15
        assert len(test_df) == 15
        
        # Check no overlapping text content between splits
        train_texts = set(train_df["text"])
        val_texts = set(val_df["text"])
        test_texts = set(test_df["text"])
        
        assert train_texts.isdisjoint(val_texts)
        assert train_texts.isdisjoint(test_texts)
        assert val_texts.isdisjoint(test_texts)
        
        # Check total coverage
        assert len(train_texts | val_texts | test_texts) == 100


class TestPreprocessNoLeakage:
    """Tests for no-leakage preprocessing."""
    
    def test_tokenizer_fit_only_on_train(self):
        # Create data with distinct vocab in train vs test
        train_df = pd.DataFrame({
            "text": ["apple banana"] * 50 + ["cherry date"] * 50,
            "label": [0] * 50 + [1] * 50,
        })
        val_df = pd.DataFrame({
            "text": ["elderberry fig"] * 20,
            "label": [2] * 20,
        })
        test_df = pd.DataFrame({
            "text": ["grape honeydew"] * 20,
            "label": [0] * 20,
        })
        
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         tokenizer, word_index) = preprocess_splits_no_leakage(
            train_df, val_df, test_df,
            max_vocab_size=1000,
            min_word_freq=1,
        )
        
        # Train vocab should have apple, banana, cherry, date
        assert "apple" in word_index
        assert "banana" in word_index
        assert "cherry" in word_index
        assert "date" in word_index
        
        # Val/test words should be OOV (index 1)
        val_sequences = tokenizer.texts_to_sequences(val_df["text"].tolist())
        test_sequences = tokenizer.texts_to_sequences(test_df["text"].tolist())
        
        # All tokens in val/test should be OOV (1) or padding (0)
        for seq in val_sequences + test_sequences:
            for token in seq:
                assert token in [0, 1]  # Only PAD or OOV


class TestLabelMap:
    """Tests for label mapping."""
    
    def test_label_map_values(self):
        assert LABEL_MAP["negative"] == 0
        assert LABEL_MAP["neutral"] == 1
        assert LABEL_MAP["positive"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])