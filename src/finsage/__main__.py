"""Main entry point for FinSage."""
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

from .training.trainer import train_from_config
from .experiments.ablation import run_model_comparison, run_predefined_ablation


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function with Hydra config."""
    # Convert OmegaConf to dict
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Determine mode
    mode = config.get("mode", "train")
    
    if mode == "train":
        # Single training run
        results = train_from_config(config)
        print("\nTraining completed!")
        print(f"Test Macro F1: {results['test_results'].get('macro_f1', 'N/A')}")
        
    elif mode == "compare":
        # Model comparison
        model_types = config.get("model_types", ["rnn", "lstm", "bilstm"])
        results = run_model_comparison(
            base_config=config,
            model_types=model_types,
            output_dir=config.get("output_dir", "model_comparison"),
        )
        
    elif mode == "ablation":
        # Ablation study
        grid_name = config.get("ablation_grid", "embedding_dim")
        results = run_predefined_ablation(
            base_config=config,
            grid_name=grid_name,
            output_dir=config.get("output_dir", "ablation_results"),
        )
        
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


def train_cli() -> None:
    """CLI entry point for training."""
    main()


def compare_cli() -> None:
    """CLI entry point for model comparison."""
    sys.argv = [sys.argv[0], "mode=compare"] + sys.argv[1:]
    main()


def ablation_cli() -> None:
    """CLI entry point for ablation study."""
    sys.argv = [sys.argv[0], "mode=ablation"] + sys.argv[1:]
    main()


if __name__ == "__main__":
    main()