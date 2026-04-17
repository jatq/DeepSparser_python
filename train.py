"""
Training script for DeepSparser.

Usage:
    python train.py --config config/config_synthetic.yaml
    python train.py --config config/config_real.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.network import DeepSparser
from utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_dataloader(config):
    """Build training DataLoader based on config (synthetic or real)."""
    if config.real_data_path is not None:
        from dataset.dataset_real import TrainDataset
        dataset = TrainDataset(config)
    else:
        from dataset.dataset_synthetic import TrainDataset
        data = np.load(Path(config.data_path) / "training_data.npz")
        dataset = TrainDataset(data["x"], data["n"], config)
    return DataLoader(dataset, batch_size=config.batchsize, shuffle=True)


def train(config):
    checkpoint_path = Path(config.checkpoint_path)
    if checkpoint_path.exists():
        logger.info("Checkpoint already exists at %s, skipping training.", checkpoint_path)
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    trainloader = build_dataloader(config)
    logger.info("Training samples: %d", len(trainloader.dataset))

    model = DeepSparser(config).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d (%.2f K)", trainable_params, trainable_params / 1e3)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_losses = []
        for s, y in trainloader:
            s, y = s.to(device), y.to(device)
            loss = model.compute_loss(y, s, config.embed_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        scheduler.step()

        avg_loss = np.mean(batch_losses)
        if epoch % max(1, config.epochs // 20) == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            logger.info("Epoch %4d/%d  loss=%.6f  lr=%.2e", epoch, config.epochs, avg_loss, lr)

    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Model saved to %s", checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train DeepSparser")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
