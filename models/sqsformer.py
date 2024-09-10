import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import torch.nn.init as init
from typing import Dict, Tuple

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_heads: int, dropout: float):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: batchsize, num_patches, embedding_dim
        b, n, _, h = *x.shape, self.heads 
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # mask value: -inf, works for both FP16 and FP32
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        x_center = []
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
            index = x.shape[1] // 2
            x_center.append(x[:,index,:])
        return x, x_center 

class SE(nn.Module):
    def __init__(self, in_chnls: int, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)

class SQSFormer(nn.Module):
    def __init__(self, params: Dict):
        super(SQSFormer, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        num_classes = data_params.get("num_classes", 13)
        patch_size = data_params.get("patch_size", 5)
        self.spectral_size = data_params.get("spectral_size", 171)

        depth = net_params.get("depth", 1)
        heads = net_params.get("heads", 8)
        mlp_head_dim = net_params.get("mlp_dim", 8)
        kernel = net_params.get('kernal', 3)
        padding = net_params.get('padding', 1)
        dropout = net_params.get("dropout", 0)
        conv2d_out = 64
        dim = net_params.get("dim", 64)
        dim_heads = dim
        
        patch_flattened_size = patch_size * patch_size

        
        self.local_trans_pixel = Transformer(dim=dim, depth=depth, heads=heads, dim_heads=dim_heads, mlp_dim=mlp_head_dim, dropout=dropout)
        self.patch_flattened_size = patch_flattened_size

        self.pixel_pos_embedding_relative = nn.Parameter(torch.randn(self.patch_flattened_size, dim))
        self.pixel_pos_scale = nn.Parameter(torch.ones(1) * 0.01)
        self.center_weight = nn.Parameter(torch.ones(depth, 1, 1) * 0.001)

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(kernel, kernel), padding=(padding,padding)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
        )

        self.senet = SE(conv2d_out, 5)

        self.cls_token_pixel = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent_pixel = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)
        self.dropout = nn.Dropout(0.1)

        linear_dim = dim * 2
        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, num_classes),
        )
    def get_position_embedding(self, x, center_index, cls_token=False):
        center_h, center_w = center_index
        b, feature_maps, h, w = x.shape
        pos_index = np.array([[max(abs(i-center_h), abs(j-center_w)) for j in range(w)] for i in range(h)]).flatten()
        if cls_token:
            pos_index = np.insert(pos_index, 0, -1)
        pos_emb = self.pixel_pos_embedding_relative[pos_index, :]
        return pos_emb

    def encoder_block(self, x):
        b, s, w, h = x.shape
        x_pixel = self.conv2d_features(x)
        pos_emb = self.get_position_embedding(x_pixel, (h//2, w//2), cls_token=False)
        x_pixel = rearrange(x_pixel, 'b s w h -> b (w h) s')
        x_pixel = x_pixel + torch.unsqueeze(pos_emb, 0) * self.pixel_pos_scale
        x_pixel = self.dropout(x_pixel)
        x_pixel, x_center_list = self.local_trans_pixel(x_pixel)
        x_center_tensor = torch.stack(x_center_list, dim=0)  # shape is [depth, batch, dim] 
        logit_pixel = torch.sum(x_center_tensor * self.center_weight, dim=0)
        #reduce_x = torch.mean(x_pixel, dim=1) This is if we take the mean of all the 25 pixels in patch instead of center pixel
        return logit_pixel

    def forward(self, x):
        '''
        x: (Batch, Spectral, Width, Height)
        
        '''
        logit_x = self.encoder_block(x)
        return self.classifier_mlp(logit_x)