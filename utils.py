import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Unified configuration for DeepSparser experiments."""

    # DCT patching
    dct_width: int = 256
    dct_stride_percent: float = 0.3
    dct_stride: int = 76
    embed_dim: int = 256
    patch_n: int = 15
    dae_dims: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 128, 64, 32, 16]
    )

    # Learnable transform
    init_embedding: bool = True
    fix_embedding: bool = False
    embed_loss_weight: float = 0.1

    # Training
    batchsize: int = 32
    lr: float = 1e-2
    epochs: int = 2000
    scheduler_step: int = 20
    scheduler_gamma: float = 0.95

    # Paths
    data_path: str = None
    real_data_path: str = None
    checkpoint_path: str = None


def load_config(config_path: str) -> Config:
    """Load a Config from a YAML file, merging with defaults."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
