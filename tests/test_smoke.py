"""
Smoke tests for DeepSparser.

These tests verify that the full pipeline (forward, backward, denoise) runs
without errors using random tensors — no real data download required.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import Config
from model.network import DeepSparser


def _make_config():
    return Config(
        dct_width=256,
        dct_stride=76,
        embed_dim=256,
        patch_n=15,
        dae_dims=[32, 64, 128, 256, 128, 64, 32, 16],
        init_embedding=True,
        fix_embedding=False,
        embed_loss_weight=0.1,
    )


def test_forward_shape():
    """Model forward pass produces the correct output shape."""
    config = _make_config()
    model = DeepSparser(config)
    model.eval()

    signal_len = 6000
    x = torch.randn(2, signal_len)
    with torch.no_grad():
        y = model(x)
    assert y.ndim == 3, f"Expected 3D output, got {y.ndim}D"
    assert y.shape[0] == 2, f"Batch dim mismatch: {y.shape[0]}"


def test_trainloss_backward():
    """Training loss computes and backpropagates without error."""
    config = _make_config()
    model = DeepSparser(config)
    model.train()

    signal_len = 6000
    y = torch.randn(2, signal_len)
    s = torch.randn(2, signal_len)

    loss = model.trainloss(y, s, embed_loss_weight=0.1)
    assert loss.ndim == 0, "Loss should be scalar"
    loss.backward()

    grad_found = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            grad_found = True
            break
    assert grad_found, "No gradients found after backward"


def test_denoise_numpy():
    """The denoise() method accepts numpy input and returns numpy output."""
    config = _make_config()
    model = DeepSparser(config)

    signal_len = 6000
    x_np = np.random.randn(signal_len).astype(np.float32)

    result = model.denoise(x_np)
    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    assert result.ndim == 1


def test_one_step_training():
    """One complete training step (forward + backward + optimizer step) works."""
    config = _make_config()
    model = DeepSparser(config)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    signal_len = 6000
    y = torch.randn(4, signal_len)
    s = torch.randn(4, signal_len)

    model.train()
    loss = model.trainloss(y, s, embed_loss_weight=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"


if __name__ == "__main__":
    print("Running smoke tests...")
    test_forward_shape()
    print("  [PASS] test_forward_shape")
    test_trainloss_backward()
    print("  [PASS] test_trainloss_backward")
    test_denoise_numpy()
    print("  [PASS] test_denoise_numpy")
    test_one_step_training()
    print("  [PASS] test_one_step_training")
    print("All smoke tests passed!")
