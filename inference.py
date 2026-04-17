"""
Inference script for DeepSparser.

Usage:
    python inference.py --config config/config_synthetic.yaml --indices 10 20 30
    python inference.py --config config/config_real.yaml --indices 0 1 2 3 4
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from model.network import DeepSparser
from utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(config, device):
    model = DeepSparser(config).to(device)
    state = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded model from %s", config.checkpoint_path)
    return model


def infer_synthetic(config, indices, output_path):
    from dataset.dataset_synthetic import TestDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)

    data = np.load(Path(config.data_path) / "testing_data.npz")
    test_dataset = TestDataset(data["x"], data["n"], config)

    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 3 * n_rows), tight_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        s, n = test_dataset[idx]
        y = test_dataset.add_noise(s, n, snr_db=0)
        s_hat = model.denoise(y)

        axes[i, 0].plot(s, lw=0.8)
        axes[i, 1].plot(y, lw=0.8)
        axes[i, 2].plot(s_hat, lw=0.8)

    axes[0, 0].set_title("Clean Signal")
    axes[0, 1].set_title("Noisy Signal")
    axes[0, 2].set_title("Denoised Signal")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")

    plt.savefig(output_path, dpi=150)
    logger.info("Saved figure to %s", output_path)
    plt.close()


def infer_real(config, indices, output_path):
    from dataset.dataset_real import TestDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)

    test_dataset = TestDataset(config)

    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows), tight_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        s = test_dataset[idx]
        s_hat = model.denoise(s)
        display_len = min(4001, len(s))

        axes[i, 0].plot(s[:display_len], lw=0.5)
        axes[i, 1].plot(s_hat[:display_len], lw=0.5)

    axes[0, 0].set_title("Original (Noisy)")
    axes[0, 1].set_title("Denoised")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")

    plt.savefig(output_path, dpi=150)
    logger.info("Saved figure to %s", output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="DeepSparser Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Indices of test samples to visualize")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path (default: results_<mode>.png)")
    args = parser.parse_args()

    config = load_config(args.config)

    is_real = config.real_data_path is not None
    output_path = args.output or f"results_{'real' if is_real else 'synthetic'}.png"

    if is_real:
        infer_real(config, args.indices, output_path)
    else:
        infer_synthetic(config, args.indices, output_path)


if __name__ == "__main__":
    main()
