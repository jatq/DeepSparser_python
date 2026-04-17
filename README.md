# DeepSparser

**DeepSparser: End-to-End Dual-Sparse Transform Learning for Seismic Signal Denoising**

A lightweight end-to-end framework for seismic signal denoising that combines a fixed-basis transform (DCT) with data-adaptive feature learning and a compact denoising autoencoder.

## Highlights

- **Dual-sparse architecture**: Fixed DCT transform → learnable linear transform → denoising autoencoder, cascaded in a unified differentiable pipeline.
- **Lightweight**: Only 0.43M parameters (95% fewer than DeepSeg), 24.49M FLOPs per forward pass.
- **Fast inference**: Processes 24-hour continuous 100 Hz seismic recordings in 9.6 s on a single NVIDIA 2080Ti GPU.
- **State-of-the-art**: +9.3 dB SNR gain over DeepDenoiser, +5.6 dB over DeepSeg on synthetic benchmarks.

## Project Structure

```
DeepSparser/
├── model/
│   └── network.py              # DeepSparser & DAE network definition
├── dataset/
│   ├── dataset_synthetic.py    # Synthetic wavelet dataset
│   └── dataset_real.py         # Real STEAD seismic dataset
├── config/
│   ├── config_synthetic.yaml   # Hyperparameters for synthetic experiments
│   └── config_real.yaml        # Hyperparameters for real-data experiments
├── train.py                    # CLI training script
├── inference.py                # CLI inference & visualization script
├── download_data.py            # Dataset download utility
├── utils.py                    # Config loader
├── demo_synthetic.ipynb        # Interactive demo (synthetic)
├── demo_real.ipynb             # Interactive demo (real data)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/jatq/DeepSparser_python.git
cd DeepSparser_python
pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 1.10, CUDA (optional, for GPU acceleration).

## Quick Start

### 1. Download Data

```bash
python download_data.py
```

If automatic download fails (Gitee redirect), manually download from [https://gitee.com/jatq33/data/tree/master/dataset/](https://gitee.com/jatq33/data/tree/master/dataset/) and place files into `dataset/real/` and `dataset/synthetic/`.

### 2. Train

```bash
# Synthetic wavelet experiment
python train.py --config config/config_synthetic.yaml

# Real seismic data (STEAD) experiment
python train.py --config config/config_real.yaml
```

Training skips automatically if a checkpoint already exists at the configured path.

### 3. Inference

```bash
# Synthetic: visualize denoising on test samples 10, 20, 30
python inference.py --config config/config_synthetic.yaml --indices 10 20 30

# Real data: visualize denoising on test samples 0-4
python inference.py --config config/config_real.yaml --indices 0 1 2 3 4
```

### 4. Interactive Demos

We provide two Jupyter Notebook demos for quick exploration:

- **`demo_synthetic.ipynb`** — Train and visualize denoising on synthetic wavelet signals.
- **`demo_real.ipynb`** — Train and visualize denoising on real STEAD seismic data.

Open them in Jupyter Notebook or JupyterLab and run all cells:

```bash
jupyter notebook demo_synthetic.ipynb
jupyter notebook demo_real.ipynb
```

## Method Overview

```
Noisy signal y
    │
    ▼
┌─────────────────────┐
│  Patching (overlap)  │   y → Y ∈ R^{N×K}
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Fixed-basis DCT B  │   Y_b = B · Y
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Learnable W₁       │   Y_f = W₁ · Y_b
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Denoising AE F_θ   │   Ŝ_f = F_θ(Y_f)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Learnable W₂       │   Ŝ_b = W₂ · Ŝ_f
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Fixed-basis Bᵀ     │   Ŝ = Bᵀ · Ŝ_b
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Unpatching (avg)    │   ŝ = Unpatch(Ŝ)
└─────────────────────┘
```

The loss function combines L₁ reconstruction loss and inverse-consistency regularization:

```
L = L_dae + λ · ‖W₂W₁ − I‖²_F
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{deepsparser2026,
  title   = {DeepSparser: End-to-End Dual-Sparse Transform Learning for Seismic Signal Denoising},
  author  = {},
  journal = {submitted},
  year    = {2026}
}
```

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant No. 62171291), Shenzhen Science and Technology Innovation Commission Project (No. JCYJ20220818101609021) and Shenzhen Stability Support Program (No. 20231127153416001).

## License

This project is released for academic research purposes. Please contact the authors for commercial use.
