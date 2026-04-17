import math

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


def load_real_data(config, is_train=True):
    """Load real seismic data (STEAD) from disk."""
    data_dir = Path(config.real_data_path)
    if is_train:
        signal = np.load(data_dir / "high_snr.npy")
        noise = np.load(data_dir / "noises.npy")
        return signal, noise
    else:
        signal = np.load(data_dir / "low_snr.npy")
        return signal


class BaseDataset(Dataset):
    """Base dataset with shared utilities for real seismic data."""

    def __init__(self, config, is_train=True):
        self.config = config
        if is_train:
            s, n = load_real_data(config, is_train=True)
            self.s = self._normalize(self._pad(s))
            self.n = self._normalize(self._pad(n))
        else:
            s = load_real_data(config, is_train=False)
            self.s = self._normalize(self._pad(s))
            self.n = None

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _pad(self, x):
        width, stride = self.config.dct_width, self.config.dct_stride
        length = x.shape[-1]
        target_length = math.ceil((length - width) / stride) * stride + width
        pad_width = target_length - length
        if pad_width == 0:
            return x
        if x.ndim == 2:
            return np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
        return np.pad(x, (0, pad_width), mode="constant")

    @staticmethod
    def _normalize(x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / std

    @staticmethod
    def add_noise(signal, noise, snr_db):
        k = 10.0 ** (-snr_db / 20.0)
        return signal + k * noise

    @staticmethod
    def compute_snr(signal, noise):
        return 10.0 * np.log10(np.var(signal) / np.var(noise))


class TrainDataset(BaseDataset):
    """Training dataset for real data: returns (clean, noisy) pairs."""

    def __init__(self, config):
        super().__init__(config, is_train=True)

    def __getitem__(self, idx):
        s = self.s[idx]
        n = self.n[np.random.randint(len(self.n))]
        snr = np.random.uniform(-10, 10)
        y = self.add_noise(s, n, snr)
        return s, y


class TestDataset(BaseDataset):
    """Test dataset for real data: returns noisy traces for inference."""

    def __init__(self, config):
        super().__init__(config, is_train=False)

    def __getitem__(self, idx):
        return self.s[idx]
