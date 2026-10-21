from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import json


class BaselineModel:
    """TF-IDF + Logistic Regression/SVM baseline for sentiment classification."""
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        classifier_type: str = "logistic_regression",
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """Initialize baseline model.
        
        Args:
            max_features: Maximum TF-IDF features
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            classifier_type: "logistic_regression" or "svm"
            C: Regularization parameter
            class_weight: Class weighting strategy
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.classifier_type = classifier_type
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            strip_accents="unicode",
        )
        
        if classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=-1,
                solver="lbfgs",
            )
        elif classifier_type == "svm":
            self.classifier = LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=random_state,
                dual="auto",
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")
        
        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier),
        ])
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        """Fit the model.
        
        Args:
            X: Text data (array of strings)
            y: Labels
            
        Returns:
            Self
        """
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels.
        
        Args:
            X: Text data
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (LogisticRegression only).
        
        Args:
            X: Text data
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.classifier_type == "svm":
            raise ValueError("SVM does not support predict_proba. Use LogisticRegression.")
        return self.pipeline.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            X: Text data
            y: True labels
            label_names: Class names for report
            
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)
        
        if label_names is None:
            label_names = ["negative", "neutral", "positive"]
        
        report = classification_report(y, y_pred, target_names=label_names, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average="macro")
        
        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "macro_f1": macro_f1,
            "accuracy": report["accuracy"],
        }
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "f1_macro",
    ) -> Dict[str, float]:
        """Cross-validation evaluation.
        
        Args:
            X: Text data
            y: Labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV scores
        """
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {
            "mean_cv_score": float(np.mean(scores)),
            "std_cv_score": float(np.std(scores)),
            "cv_scores": scores.tolist(),
        }
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Save path
        """
        joblib.dump(self.pipeline, path)
        # Save config separately
        config = {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "classifier_type": self.classifier_type,
            "C": self.C,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
        with open(path.replace(".pkl", "_config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        """Load model from disk.
        
        Args:
            path: Model path
            
        Returns:
            Loaded BaselineModel
        """
        pipeline = joblib.load(path)
        # Load config
        with open(path.replace(".pkl", "_config.json"), "r") as f:
            config = json.load(f)
        
        model = cls(**config)
        model.pipeline = pipeline
        model.is_fitted = True
        return model


def train_baseline(
    train_texts: np.ndarray,
    train_labels: np.ndarray,
    val_texts: np.ndarray,
    val_labels: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[BaselineModel, Dict[str, Any]]:
    """Train and evaluate baseline model.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        config: Model configuration
        
    Returns:
        Tuple of (model, evaluation_results)
    """
    model = BaselineModel(
        max_features=config.get("tfidf", {}).get("max_features", 10000),
        ngram_range=tuple(config.get("tfidf", {}).get("ngram_range", [1, 2])),
        min_df=config.get("tfidf", {}).get("min_df", 2),
        max_df=config.get("tfidf", {}).get("max_df", 0.95),
        classifier_type=config.get("classifier", {}).get("type", "logistic_regression"),
        C=config.get("classifier", {}).get("C", 1.0),
        class_weight=config.get("classifier", {}).get("class_weight", "balanced"),
        max_iter=config.get("classifier", {}).get("max_iter", 1000),
        random_state=config.get("classifier", {}).get("random_state", 42),
    )
    
    model.fit(train_texts, train_labels)
    
    # Evaluate on validation set
    val_results = model.evaluate(val_texts, val_labels)
    
    # Cross-validation on training set
    cv_results = model.cross_validate(train_texts, train_labels)
    
    results = {
        "validation": val_results,
        "cross_validation": cv_results,
    }
    
    return model, results


if __name__ == "__main__":
    # Quick test
    texts = np.array([
        "The company reported strong earnings growth.",
        "Stock prices fell sharply after the announcement.",
        "Market remains neutral amid uncertainty.",
        "Profits surged beyond expectations.",
        "Revenue declined due to market conditions.",
    ])
    labels = np.array([2, 0, 1, 2, 0])  # positive, negative, neutral, positive, negative
    
    model = BaselineModel(max_features=1000, classifier_type="logistic_regression")
    model.fit(texts, labels)
    preds = model.predict(texts)
    print("Predictions:", preds)
    print("Evaluation:", model.evaluate(texts, labels))