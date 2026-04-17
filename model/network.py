import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 output_padding=1):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2, output_padding=output_padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class DAE(nn.Module):
    """Denoising Autoencoder with U-Net-style skip connections.

    Architecture (default dae_dims=[32,64,128,256,128,64,32,16]):
        Encoder : patch_n → 32 → 64 → 128 → 256   (ConvBlocks, stride=2)
        Decoder : 256 → 128 → 64 → 32 → 16         (DeconvBlocks, stride=2)
        Head    : 16 → 1                             (Conv1d 1×1)
    Skip connections link encoder layers {0,1,2} to decoder layers {5,6,7}.
    """

    _SKIP_SAVE = {0, 1, 2}
    _SKIP_CAT = {5, 6, 7}

    def __init__(self, config):
        super().__init__()
        channels = [config.patch_n] + config.dae_dims

        self.unet = nn.ModuleList()
        skip_channels = []
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            if i in self._SKIP_SAVE:
                skip_channels.append(out_ch)
            extra = skip_channels.pop() if i in self._SKIP_CAT else 0
            block = DeconvBlock if (in_ch + extra > out_ch) else ConvBlock
            self.unet.append(block(in_ch + extra, out_ch))
        self.unet.append(nn.Conv1d(channels[-1], 1, kernel_size=1))

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.unet):
            if i in self._SKIP_CAT:
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x)
            if i in self._SKIP_SAVE:
                skips.append(x)
        return x


class DeepSparser(nn.Module):
    """End-to-end dual-sparse transform learning for signal denoising.

    Pipeline: DCT → learnable W₁ → DAE → inverse W₂ → IDCT
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.register_buffer('dct_weight', self._build_dct_basis(inverse=False))
        self.register_buffer('idct_weight', self._build_dct_basis(inverse=True))
        self.register_buffer('identity', torch.eye(config.embed_dim, config.dct_width))

        self.embed = nn.Conv1d(config.dct_width, config.embed_dim, 1, bias=False)
        self.inverse_embed = nn.Conv1d(config.embed_dim, config.dct_width, 1, bias=False)

        if config.init_embedding:
            self._init_embedding_weights()
        self._set_embedding_trainable(not config.fix_embedding)

        self.dae = DAE(config)

    def _init_embedding_weights(self):
        self.embed.weight.data = torch.eye(
            self.config.embed_dim, self.config.dct_width,
        ).unsqueeze(-1)
        self.inverse_embed.weight.data = torch.eye(
            self.config.dct_width, self.config.embed_dim,
        ).unsqueeze(-1)

    def _set_embedding_trainable(self, trainable=True):
        for p in self.embed.parameters():
            p.requires_grad_(trainable)
        for p in self.inverse_embed.parameters():
            p.requires_grad_(trainable)

    def forward(self, y):
        """y: (bs, signal_len) → DCT coefficients (bs, dct_width, n_patches)."""
        y_dct = self._dct(y.unsqueeze(1).float())
        bs, dct_width, n_patches = y_dct.shape

        y_embed = self.embed(y_dct)                              # (bs, embed_dim, n_patches)
        patches = self._extract_patches(y_embed)                 # (bs*n_patches, patch_n, embed_dim)
        denoised = self.dae(patches).permute(0, 2, 1)            # (bs*n_patches, embed_dim, 1)
        reconstructed = self.inverse_embed(denoised).squeeze(-1) # (bs*n_patches, dct_width)

        return reconstructed.reshape(bs, n_patches, dct_width).permute(0, 2, 1)

    def compute_loss(self, y, s, embed_loss_weight=0.0):
        y, y_std = self._normalize(y)
        s = s / y_std

        y_dct_hat = self.forward(y)
        s_dct = self._dct(s.unsqueeze(1).float())
        loss = F.l1_loss(y_dct_hat, s_dct)

        if embed_loss_weight > 0:
            w_product = torch.matmul(
                self.embed.weight.squeeze(-1),
                self.inverse_embed.weight.squeeze(-1),
            )
            loss = loss + embed_loss_weight * F.mse_loss(w_product, self.identity)
        return loss

    # keep backward-compatible alias
    trainloss = compute_loss

    def denoise(self, y):
        device = self.dct_weight.device
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=device)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        y, y_std = self._normalize(y)
        with torch.no_grad():
            self.eval()
            y_hat = self._idct(self.forward(y))
        return (y_hat * y_std).squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------ #
    #  DCT / IDCT
    # ------------------------------------------------------------------ #

    def _dct(self, x):
        return F.conv1d(x, self.dct_weight, stride=self.config.dct_stride)

    def _idct(self, x):
        """Inverse DCT with overlap-add averaging (vectorised via F.fold)."""
        x = F.conv1d(x, self.idct_weight)
        bs, width, n_patches = x.shape
        out_len = (n_patches - 1) * self.config.dct_stride + width

        output = F.fold(
            x, output_size=(1, out_len),
            kernel_size=(1, width), stride=(1, self.config.dct_stride),
        )
        norm = F.fold(
            torch.ones_like(x), output_size=(1, out_len),
            kernel_size=(1, width), stride=(1, self.config.dct_stride),
        )
        return (output / norm).reshape(bs, out_len)

    def _build_dct_basis(self, inverse=False):
        N = self.config.dct_width
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32)
        basis = torch.cos(torch.pi * k[:, None] * (2 * n[None, :] + 1) / (2 * N))
        basis[0] *= (1 / N) ** 0.5
        basis[1:] *= (2 / N) ** 0.5
        return basis.T.unsqueeze(2) if inverse else basis.unsqueeze(1)

    # ------------------------------------------------------------------ #
    #  Patching / normalisation
    # ------------------------------------------------------------------ #

    def _extract_patches(self, x):
        """Sliding-window context patches along the temporal axis.

        x: (bs, embed_dim, n_patches) → (bs*n_patches, patch_n, embed_dim)
        """
        bs, embed_dim, n_patches = x.shape
        half = self.config.patch_n // 2
        x_padded = F.pad(x, (half, half))
        patches = x_padded.unfold(2, self.config.patch_n, 1)
        return patches.permute(0, 2, 3, 1).reshape(
            bs * n_patches, self.config.patch_n, embed_dim,
        )

    @staticmethod
    def _normalize(x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / std, std
