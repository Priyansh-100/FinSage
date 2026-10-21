from typing import Dict, Any, List, Optional, Callable
import itertools
import copy
import pandas as pd
from dataclasses import dataclass, field
import json
import os

from ..training.trainer import train_from_config
from ..evaluation.metrics import compare_models, print_comparison_table
from ..experiments.tracking import log_experiment


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    # Base configuration (will be overridden)
    base_config: Dict[str, Any]
    
    # Parameter grid to explore
    param_grid: Dict[str, List[Any]]
    
    # Fixed parameters (not varied)
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metric to optimize
    target_metric: str = "macro_f1"
    
    # Output directory
    output_dir: str = "ablation_results"
    
    # Whether to use MLflow
    use_mlflow: bool = True
    mlflow_experiment: str = "finsage-ablation"
    
    # Maximum runs (None for all combinations)
    max_runs: Optional[int] = None


def generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from grid.
    
    Args:
        param_grid: Dict of {param_name: [values]}
        
    Returns:
        List of parameter dictionaries
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def apply_params_to_config(
    base_config: Dict[str, Any],
    params: Dict[str, Any],
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply parameter combination to base config.
    
    Args:
        base_config: Base configuration
        params: Parameter values to apply
        fixed_params: Fixed parameter values
        
    Returns:
        Updated configuration
    """
    config = copy.deepcopy(base_config)
    
    # Apply fixed params first
    for key, value in fixed_params.items():
        _set_nested(config, key, value)
    
    # Apply varying params
    for key, value in params.items():
        _set_nested(config, key, value)
    
    return config


def _set_nested(d: Dict[str, Any], key: str, value: Any) -> None:
    """Set nested dictionary value using dot notation."""
    keys = key.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def run_ablation(
    ablation_config: AblationConfig,
    train_fn: Callable = train_from_config,
) -> pd.DataFrame:
    """Run ablation study.
    
    Args:
        ablation_config: Ablation configuration
        train_fn: Training function (default: train_from_config)
        
    Returns:
        Results DataFrame
    """
    os.makedirs(ablation_config.output_dir, exist_ok=True)
    
    # Generate combinations
    combinations = generate_param_combinations(ablation_config.param_grid)
    
    if ablation_config.max_runs:
        combinations = combinations[:ablation_config.max_runs]
    
    print(f"Running ablation study with {len(combinations)} combinations...")
    print(f"Parameter grid: {ablation_config.param_grid}")
    
    results_list = []
    
    for i, params in enumerate(combinations):
        print(f"\n=== Run {i+1}/{len(combinations)}: {params} ===")
        
        # Create config for this run
        run_config = apply_params_to_config(
            ablation_config.base_config,
            params,
            ablation_config.fixed_params,
        )
        
        # Add run-specific tags
        run_name = f"ablation_{i+1}"
        for k, v in params.items():
            run_name += f"_{k}={v}"
        run_name = run_name.replace(".", "_").replace(" ", "")
        
        try:
            # Train model
            results = train_fn(run_config)
            
            # Extract metrics
            test_results = results.get("test_results", {})
            row = {**params, **test_results}
            row["run_name"] = run_name
            row["status"] = "success"
            results_list.append(row)
            
            # Log to MLflow
            if ablation_config.use_mlflow:
                log_experiment(
                    config=run_config,
                    results=results,
                    experiment_name=ablation_config.mlflow_experiment,
                    run_name=run_name,
                )
            
            # Print progress
            metric_val = test_results.get(ablation_config.target_metric, "N/A")
            print(f"  {ablation_config.target_metric}: {metric_val}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            row = {**params, "run_name": run_name, "status": f"failed: {e}"}
            results_list.append(row)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save results
    results_path = os.path.join(ablation_config.output_dir, "ablation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Save best config
    if len(results_df[results_df["status"] == "success"]) > 0:
        successful = results_df[results_df["status"] == "success"]
        if ablation_config.target_metric in successful.columns:
            best_idx = successful[ablation_config.target_metric].idxmax()
            best_config = successful.loc[best_idx].to_dict()
            
            best_config_path = os.path.join(ablation_config.output_dir, "best_config.json")
            with open(best_config_path, "w") as f:
                # Only save the params that were varied
                best_params = {k: v for k, v in best_config.items() if k in ablation_config.param_grid}
                json.dump(best_params, f, indent=2)
            print(f"Best config saved to {best_config_path}")
    
    # Print comparison table
    print_comparison_table(results_df)
    
    return results_df


def run_model_comparison(
    base_config: Dict[str, Any],
    model_types: List[str],
    output_dir: str = "model_comparison",
    use_mlflow: bool = True,
) -> pd.DataFrame:
    """Compare different model architectures.
    
    Args:
        base_config: Base configuration
        model_types: List of model types to compare
        output_dir: Output directory
        use_mlflow: Whether to log to MLflow
        
    Returns:
        Comparison DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results_list = []
    
    for model_type in model_types:
        print(f"\n=== Training {model_type} ===")
        
        run_config = copy.deepcopy(base_config)
        run_config["model"]["type"] = model_type
        
        run_name = f"compare_{model_type}"
        
        try:
            results = train_from_config(run_config)
            test_results = results.get("test_results", {})
            
            row = {"model": model_type, **test_results, "status": "success"}
            results_list.append(row)
            
            if use_mlflow:
                log_experiment(
                    config=run_config,
                    results=results,
                    experiment_name="finsage-model-comparison",
                    run_name=run_name,
                )
            
        except Exception as e:
            print(f"  ERROR: {e}")
            row = {"model": model_type, "status": f"failed: {e}"}
            results_list.append(row)
    
    results_df = pd.DataFrame(results_list)
    
    # Save
    results_path = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    
    print_comparison_table(results_df)
    
    return results_df


# Predefined ablation grids
ABLATION_GRIDS = {
    "embedding_dim": {
        "model.embedding.output_dim": [64, 128, 256],
    },
    "hidden_units": {
        "model.rnn.hidden_units": [64, 128, 256],
    },
    "num_layers": {
        "model.rnn.num_layers": [1, 2],
    },
    "dropout": {
        "model.rnn.dropout": [0.0, 0.2, 0.3, 0.5],
    },
    "bidirectional": {
        "model.rnn.bidirectional": [False, True],
    },
    "learning_rate": {
        "train.learning_rate": [0.001, 0.0005, 0.0001],
    },
    "batch_size": {
        "data.batch_size": [16, 32, 64],
        "train.batch_size": [16, 32, 64],
    },
    "seq_length": {
        "data.preprocessing.max_seq_length": [50, 100, 150],
    },
    "optimizer": {
        "train.optimizer": ["adam", "rmsprop", "sgd"],
    },
}


def get_ablation_grid(name: str) -> Dict[str, List[Any]]:
    """Get predefined ablation grid by name.
    
    Args:
        name: Grid name
        
    Returns:
        Parameter grid
    """
    return ABLATION_GRIDS.get(name, {})


def run_predefined_ablation(
    base_config: Dict[str, Any],
    grid_name: str,
    output_dir: str = "ablation_results",
    **kwargs,
) -> pd.DataFrame:
    """Run ablation with predefined grid.
    
    Args:
        base_config: Base configuration
        grid_name: Name of predefined grid
        output_dir: Output directory
        **kwargs: Additional AblationConfig args
        
    Returns:
        Results DataFrame
    """
    param_grid = get_ablation_grid(grid_name)
    if not param_grid:
        raise ValueError(f"Unknown grid: {grid_name}. Available: {list(ABLATION_GRIDS.keys())}")
    
    ablation_config = AblationConfig(
        base_config=base_config,
        param_grid=param_grid,
        output_dir=output_dir,
        **kwargs,
    )
    
    return run_ablation(ablation_config)


if __name__ == "__main__":
    # Quick test
    print("Available ablation grids:", list(ABLATION_GRIDS.keys()))
    for name, grid in ABLATION_GRIDS.items():
        print(f"  {name}: {grid}")