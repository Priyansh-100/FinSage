from typing import Dict, Any, Optional, List
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import os
import json
import pandas as pd


class ExperimentTracker:
    """MLflow experiment tracking for FinSage."""
    
    def __init__(
        self,
        experiment_name: str = "finsage-sentiment",
        tracking_uri: str = "file:./mlruns",
    ):
        """Initialize experiment tracker.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = None
        self.experiment_id = None
        self._setup()
    
    def _setup(self):
        """Setup MLflow client and experiment."""
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional run name
            tags: Optional tags
            
        Returns:
            Active MLflow run
        """
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
        )
        
        if tags:
            mlflow.set_tags(tags)
        
        return run
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        **kwargs,
    ) -> None:
        """Log Keras model.
        
        Args:
            model: Keras model
            artifact_path: Artifact path
            **kwargs: Additional arguments to mlflow.keras.log_model
        """
        mlflow.keras.log_model(model, artifact_path, **kwargs)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact file.
        
        Args:
            local_path: Local file path
            artifact_path: Optional artifact subdirectory
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact directory.
        
        Args:
            local_dir: Local directory path
            artifact_path: Optional artifact subdirectory
        """
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def end_run(self) -> None:
        """End current run."""
        mlflow.end_run()
    
    def get_best_run(
        self,
        metric: str = "val_macro_f1",
        mode: str = "max",
    ) -> Optional[Dict[str, Any]]:
        """Get best run by metric.
        
        Args:
            metric: Metric name
            mode: "max" or "min"
            
        Returns:
            Best run info or None
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"],
            max_results=1,
        )
        return runs[0] if runs else None
    
    def load_best_model(
        self,
        metric: str = "val_macro_f1",
        mode: str = "max",
    ):
        """Load best model from MLflow.
        
        Args:
            metric: Metric name
            mode: "max" or "min"
            
        Returns:
            Loaded Keras model
        """
        best_run = self.get_best_run(metric, mode)
        if best_run is None:
            raise ValueError("No runs found")
        
        model_uri = f"runs:/{best_run.info.run_id}/model"
        return mlflow.keras.load_model(model_uri)
    
    def get_run_history(self, run_id: str) -> Dict[str, List]:
        """Get metric history for a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Dictionary of metric histories
        """
        run = self.client.get_run(run_id)
        return run.data.metrics
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str],
    ) -> pd.DataFrame:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs
            metrics: Metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        import pandas as pd
        
        rows = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {"run_id": run_id, "run_name": run.data.tags.get("mlflow.runName", "")}
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentTracker._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


def log_experiment(
    config: Dict[str, Any],
    results: Dict[str, Any],
    model=None,
    experiment_name: str = "finsage-sentiment",
    tracking_uri: str = "file:./mlruns",
    run_name: Optional[str] = None,
) -> str:
    """Convenience function to log a complete experiment.
    
    Args:
        config: Experiment configuration
        results: Training/evaluation results
        model: Optional Keras model to log
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking URI
        run_name: Optional run name
        
    Returns:
        Run ID
    """
    tracker = ExperimentTracker(experiment_name, tracking_uri)
    
    with tracker.start_run(run_name=run_name) as run:
        # Log config as params
        tracker.log_params(config)
        
        # Log final metrics
        test_results = results.get("test_results", {})
        tracker.log_metrics({k: v for k, v in test_results.items() if isinstance(v, (int, float))})
        
        # Log model
        if model is not None:
            tracker.log_model(model)
        
        # Log artifacts (plots, etc.)
        if "plots" in results:
            for plot_name, plot_path in results["plots"].items():
                if os.path.exists(plot_path):
                    tracker.log_artifact(plot_path, "plots")
        
        return run.info.run_id


if __name__ == "__main__":
    # Quick test
    tracker = ExperimentTracker("test-experiment", "file:./mlruns_test")
    print(f"Experiment ID: {tracker.experiment_id}")
    print("MLflow tracking module loaded successfully")