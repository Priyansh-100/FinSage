import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable TensorFlow deterministic operations
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1" if deterministic else "0"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1" if deterministic else "0"
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def set_global_determinism(seed: int = 42) -> None:
    """Configure TensorFlow for full determinism.
    
    Args:
        seed: Random seed value
    """
    set_seed(seed, deterministic=True)
    
    # Additional TF settings for determinism
    tf.keras.utils.set_random_seed(seed)
    
    # Limit GPU memory growth to avoid non-determinism from memory allocation
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def get_seed() -> int:
    """Get the current global seed from TensorFlow."""
    return tf.keras.utils.get_random_seed()


class SeedContext:
    """Context manager for temporary seed setting."""
    
    def __init__(self, seed: int):
        self.seed = seed
        self.prev_seed = None
    
    def __enter__(self):
        self.prev_seed = get_seed()
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_seed(self.prev_seed)