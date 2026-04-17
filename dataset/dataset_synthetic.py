import math

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset with shared utilities for signal padding, normalization, and noise mixing."""

    def __init__(self, s, n, config):
        self.config = config
        self.s = self._normalize(self._pad(s))
        self.n = self._normalize(self._pad(n))

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
        """Mix signal and noise at a given SNR (dB).

        Assumes both signal and noise are pre-normalized to unit variance, so
        that SNR = -20 * log10(k) where k is the noise scaling factor.
        """
        k = 10.0 ** (-snr_db / 20.0)
        return signal + k * noise

    @staticmethod
    def compute_snr(signal, noise):
        return 10.0 * np.log10(np.var(signal) / np.var(noise))


class TrainDataset(BaseDataset):
    """Training dataset: returns (clean, noisy) pairs with random SNR mixing."""

    def __getitem__(self, idx):
        s = self.s[idx]
        n = self.n[np.random.randint(len(self.n))]
        snr = np.random.uniform(-10, 10)
        y = self.add_noise(s, n, snr)
        return s, y


class TestDataset(BaseDataset):
    """Test dataset: returns (clean, noise) pairs for controlled evaluation."""

    def __getitem__(self, idx):
        return self.s[idx], self.n[idx]
