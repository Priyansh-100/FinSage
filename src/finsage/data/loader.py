import os
import requests
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm


PHRASEBANK_URLS = {
    "50Agree": "https://raw.githubusercontent.com/ahmedrachid/FinancialPhraseBank/master/FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    "75Agree": "https://raw.githubusercontent.com/ahmedrachid/FinancialPhraseBank/master/FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "100Agree": "https://raw.githubusercontent.com/ahmedrachid/FinancialPhraseBank/master/FinancialPhraseBank-v1.0/Sentences_100Agree.txt",
}

PHRASEBANK_ENCODING = "latin-1"
LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def download_phrasebank(variant: str, data_dir: Path) -> Path:
    """Download Financial PhraseBank dataset if not present.
    
    Args:
        variant: Dataset variant ("50Agree", "75Agree", "100Agree")
        data_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded file
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"Sentences_{variant}.txt"
    
    if file_path.exists():
        print(f"Dataset already exists at {file_path}")
        return file_path
    
    url = PHRASEBANK_URLS.get(variant)
    if not url:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(PHRASEBANK_URLS.keys())}")
    
    print(f"Downloading Financial PhraseBank {variant} from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    with open(file_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"Downloaded to {file_path}")
    return file_path


def load_phrasebank(variant: str = "50Agree", data_dir: str = "data/raw") -> pd.DataFrame:
    """Load Financial PhraseBank dataset into a DataFrame.
    
    Args:
        variant: Dataset variant ("50Agree", "75Agree", "100Agree")
        data_dir: Directory containing the dataset
        
    Returns:
        DataFrame with columns: text, label, label_name
    """
    data_path = Path(data_dir)
    file_path = data_path / f"Sentences_{variant}.txt"
    
    if not file_path.exists():
        file_path = download_phrasebank(variant, data_path)
    
    texts = []
    labels = []
    label_names = []
    
    with open(file_path, "r", encoding=PHRASEBANK_ENCODING) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "text @label@"
            if "@" not in line:
                continue
            text_part, label_part = line.rsplit("@", 1)
            label_name = label_part.strip().lower()
            if label_name not in LABEL_MAP:
                continue
            texts.append(text_part.strip())
            labels.append(LABEL_MAP[label_name])
            label_names.append(label_name)
    
    df = pd.DataFrame({
        "text": texts,
        "label": labels,
        "label_name": label_names,
    })
    
    print(f"Loaded {len(df)} samples from {variant}")
    print(f"Class distribution:\n{df['label_name'].value_counts()}")
    
    return df


def get_class_weights(df: pd.DataFrame, method: str = "balanced") -> dict:
    """Compute class weights for imbalanced datasets.
    
    Args:
        df: DataFrame with 'label' column
        method: Weighting method ("balanced", "auto", or dict)
        
    Returns:
        Dictionary mapping class indices to weights
    """
    if isinstance(method, dict):
        return method
    
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(df["label"].values)
    if method == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=df["label"].values)
    elif method == "auto":
        # Inverse frequency weighting
        counts = df["label"].value_counts().sort_index()
        weights = len(df) / (len(classes) * counts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return dict(zip(classes, weights))


if __name__ == "__main__":
    # Quick test
    df = load_phrasebank("50Agree", "data/raw")
    print(df.head())
    print(f"\nClass weights: {get_class_weights(df)}")