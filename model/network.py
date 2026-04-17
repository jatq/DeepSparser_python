import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv
class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks = 3, s=2):
        super().__init__()
        self.conv = nn.Conv1d(ic, oc, kernel_size=ks, stride=s, padding=ks//2)
        self.bn = nn.BatchNorm1d(oc)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Deconv
class DeconvBlock(nn.Module):
    def __init__(self, ic, oc, ks = 3, s = 2, output_padding = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(ic, oc, kernel_size=ks, stride=s, padding=ks//2, output_padding=output_padding)
        self.bn = nn.BatchNorm1d(oc)
        self.relu = nn.ReLU() 
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

# Transpose
class Transpose(nn.Module):
    def __init__(self, dims=(1,2)):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.transpose(*self.dims)

# Squeeze
class Squeeze(nn.Module):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(dim=self.dim)

# Denosing Autoencoder
class DAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()    
        self.shourtcut_index = [np.array([0, 1, 2]), np.array([5,6,7])]
        encoder_chans = [config.patch_n] + config.dae_dims
        shortcut_chans = []
        self.unet = nn.ModuleList()
        for i, (ic, oc) in enumerate(zip(encoder_chans[:-1], encoder_chans[1:])):
            shortcut_chans.append(oc) if i in self.shourtcut_index[0] else None
            extra_ic = shortcut_chans.pop() if i in self.shourtcut_index[1] else 0
            self.unet.append(
                DeconvBlock(ic + extra_ic, oc)
                if (ic > oc) else
                ConvBlock(ic + extra_ic, oc))
        
        self.unet.append(nn.Conv1d(oc, 1, kernel_size=1))

    def forward(self, x):
        # inputs:(batch, patch_n, embed_dim)
        shortcuts = []
        for i, layer in enumerate(self.unet):
            x  = layer(x)
            if i in self.shourtcut_index[0]:
                shortcuts.append(x)
                
            elif i in self.shourtcut_index[1]-1:
                x = torch.cat([x, shortcuts.pop()], dim=1)
        return x

# Deep Embedding
class DeepSparser(nn.Module):
    def __init__(self, config):
        super(DeepSparser, self).__init__()
        self.config = config
        self.register_buffer('dct_weight', self._generate_dct_weight(False))
        self.register_buffer('idct_weight', self._generate_dct_weight(True))
        self.register_buffer('identity', torch.eye(config.embed_dim, config.dct_width))
        self.embed = nn.Conv1d(config.dct_width, config.embed_dim, 1, bias=False)
        self.inverse_embed = nn.Conv1d(config.embed_dim, config.dct_width, 1, bias=False)
        if config.init_embedding:
            self.init_embeddding_layer()
        self.fix_embedding_layer(config.fix_embedding)

        self.dae = DAE(config)
        self.mae = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def init_embeddding_layer(self):
        self.embed.weight.data = torch.eye(self.config.embed_dim, self.config.dct_width).unsqueeze(-1)
        self.inverse_embed.weight.data = torch.eye(self.config.dct_width, self.config.embed_dim).unsqueeze(-1)

    def fix_embedding_layer(self, is_fixed=False):
        self.embed.requires_grad = not is_fixed
        self.inverse_embed.requires_grad = not is_fixed


    def forward(self, y):
        # y|s: (bs, n)
        y_dct = self._dct(y.unsqueeze(1).float())                # -> (bs, dct_width, patches)
        (bs, dct_width, patches) = y_dct.shape
        y_embed = self.embed(y_dct)                             # -> (bs, embed_dim, patches)
        y_patches = self._split_patches(y_embed)                # -> (bs * patches, patch_n, embed_dim)
        y_dae = self.dae(y_patches).permute(0, 2, 1)            # -> (bs * patches, embed_dim, 1)
        y_dct_hat = self.inverse_embed(y_dae).squeeze(-1)       # -> (bs  * patches, dct_width)
        y_dct_hat = y_dct_hat.reshape(bs, patches, dct_width).permute(0, 2, 1)  # -> (bs, dct_width, patches)
        return y_dct_hat
        
    def trainloss(self, y, s, embed_loss_weight=0.0):
        y, y_std = self.normalization_1d(y)
        s = s / y_std
        y_dct_hat = self.forward(y)
        s_dct = self._dct(s.unsqueeze(1).float())
        loss = self.mae(y_dct_hat, s_dct)
        
        if embed_loss_weight > 0:
            loss += embed_loss_weight * self.embed_loss()
        return loss
    
    def embed_loss(self):
        return self.mse(torch.matmul(self.embed.weight.squeeze(-1), self.inverse_embed.weight.squeeze(-1)), self.identity)
    
    def denoise(self, y):
        device = next(self.parameters()).device
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).to(device)
        if y.ndim  == 1:
            y = y.unsqueeze(0)
        y, y_std = self.normalization_1d(y)
        with torch.no_grad():
            self.eval()
            y_dct_hat = self.forward(y)
            y_hat = self._idct(y_dct_hat)
        return (y_hat * y_std).squeeze(0).cpu().numpy()

    def _dct(self, x):
        return F.conv1d(x, self.dct_weight, stride = self.config.dct_stride, bias=None)
    
    def _idct(self, x):
        device = next(self.parameters()).device
        x =  F.conv1d(x, self.idct_weight, stride = 1, bias=None)
        bs, dct_width, patches = x.shape

        res = torch.zeros((bs, (patches - 1) * self.config.dct_stride + dct_width), dtype=torch.float32).to(device)
        w = torch.zeros((patches, res.shape[1]), dtype=torch.float32).to(device)
        for i in range(patches):
            res[:, i*self.config.dct_stride:i*self.config.dct_stride+dct_width] += x[:, :, i]
            w[i, i*self.config.dct_stride:i*self.config.dct_stride+dct_width] = 1.0
        return res/w.sum(axis=0)


    def _generate_dct_weight(self, is_idct=False):
        w = torch.zeros((self.config.dct_width, self.config.dct_width), dtype=torch.float32)
        w[0, :] = 1 * np.sqrt(1 / self.config.dct_width)
        for i in range(1, self.config.dct_width):
            w[i, :] = torch.cos(torch.pi * i * (2 * torch.arange(self.config.dct_width) + 1) / (2 * self.config.dct_width)) * np.sqrt(2 / self.config.dct_width)
        return w.unsqueeze(1) if not is_idct else w.T.unsqueeze(2)
    
    def _split_patches(self, x_dct):
        # x_dct: (bs, dct_width, patches)
        patches = x_dct.shape[-1]
        x_dct_pad = F.pad(x_dct, (self.config.patch_n//2, self.config.patch_n//2))
        mask = torch.arange(patches)[None,:].repeat(self.config.patch_n, 1) + \
            torch.arange(self.config.patch_n)[:,None].repeat(1, patches)
            
        # (bs, dct_width, patch_n, patches_n)
        patches =  x_dct_pad[:, :, mask]
        # (bs * patches_n, patch_n, dct_width)
        patches = patches.permute(0, 3, 2, 1).reshape(x_dct.shape[0] * patches.shape[-1], self.config.patch_n, self.config.embed_dim)
        return patches
    
    @staticmethod
    def normalization_1d(x):
        mean = torch.mean(x, axis=-1, keepdims=True)
        std = torch.std(x, axis=-1, keepdims=True)
        x = (x-mean)/std
        return x, std
