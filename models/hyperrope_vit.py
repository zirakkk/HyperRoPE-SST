import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict

class RoPEUtils:
    @staticmethod
    def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
        freqs_x = []
        freqs_y = []
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1) # [mag * cos(theta), mag * cos(theta + pi/2)] We know that cos(theta + pi/2) = -sin(theta), [mag[1]*cos(theta), mag[2]*cos(theta), mag[1]* -sin(theta), mag[2]* -sin(theta)]
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1) # [mag * sin(theta), mag * sin(theta + pi/2)] We know that sin(theta + pi/2) = cos(theta), [mag[1]*sin(theta), mag[2]*sin(theta), mag[1]*cos(theta), mag[2]*cos(theta)]
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)  #shape is [2, num_heads, 4] 
        return freqs

    @staticmethod
    def init_coords_xy(end_x: int, end_y: int):
        t = torch.arange(end_x * end_y, dtype=torch.float32)  #shape is [patchsize * patchsize = 25]
        t_x = (t % end_x).float() #[0 ,1 , 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        t_y = torch.div(t, end_x, rounding_mode='floor').float() ##[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        return t_x, t_y

    @staticmethod
    def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
        N = t_x.shape[0]
        depth = freqs.shape[1]
        with torch.cuda.amp.autocast(enabled=False): #This line ensures that the following operations run in full precision (float32)
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3) # [N , 3, 32]  --> [3, 25, 8, 4] --> [3, 8, 25, 4]
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3) # [N , 3, 32]  --> [3, 25, 8, 4] --> [3, 8, 25, 4]
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y) #e^(i . (freqs_x + freqs_y)) this outputs a complex number with real and imaginary parts derived from combined x and y frequencies
        return freqs_cis

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
        elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) #This converts the last two dimensions of xq into a complex number [65, 8, 25, 8] --> [65, 8, 25, 4]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) #This converts the last two dimensions of xk into a complex number [65, 8, 25, 8] --> [65, 8, 25, 4]
        freqs_cis = RoPEUtils.reshape_for_broadcast(freqs_cis, xq_) #This reshapes freqs_cis to match the shape of xq_ [ 8, 25, 4] --> [65, 8, 25, 4]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) #This applies the rotation to the complex numbers in xq_, then converts it back to real numbers [65, 8, 25, 4, 2] and flattens the last two dimensions [65, 8, 25, 8]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) #This applies the rotation to the complex numbers in xk_, then converts it back to real numbers [65, 8, 25, 4, 2] and flattens the last two dimensions [65, 8, 25, 8]
        return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

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

class RoPEAttention(nn.Module):
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

    def forward(self, x, freqs_cis=None, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if freqs_cis is not None:
            q, k = RoPEUtils.apply_rotary_emb(q, k, freqs_cis)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class RoPETransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, RoPEAttention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, freqs_cis=None, mask=None):
        x_center = []
        for i, (attention, mlp) in enumerate(self.layers):
            x = attention(x, freqs_cis=freqs_cis[i] if freqs_cis is not None else None, mask=mask)
            x = mlp(x)
            index = x.shape[1] // 2
            x_center.append(x[:,index,:])
        return x, x_center

'''    
class ZeroMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, bias=False):
        super(ZeroMaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 
                                               kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):
        # Create a mask where input is non-zero
        mask = (input != 0).float()  # Shape: [batch_size, in_channels, H, W]

        # Prepare ones_kernel for depthwise convolution
        ones_kernel = torch.ones(self.in_channels, 1, self.weight.size(2), self.weight.size(3), 
                                 device=input.device)  # Shape: [in_channels, 1, kH, kW]

        # Compute valid_count using depthwise convolution
        valid_count = F.conv2d(mask, ones_kernel, bias=None, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, groups=self.in_channels)
        # Shape of valid_count: [batch_size, in_channels, H_out, W_out]

        # Sum over the input channels to get a single channel
        valid_count = valid_count.sum(dim=1, keepdim=True)  # Shape: [batch_size, 1, H_out, W_out]

        # Avoid division by zero
        valid_count = torch.clamp(valid_count, min=1e-5)

        # Perform convolution on masked input
        masked_input = input * mask  # Element-wise multiplication

        # Standard convolution
        output = F.conv2d(masked_input, self.weight, self.bias, stride=self.stride,
                          padding=self.padding, dilation=self.dilation)

        # Normalize the output
        output = output / valid_count  # Broadcasting over [batch_size, out_channels, H_out, W_out]

        return output
'''
class HyperRopeViT(nn.Module):
    def __init__(self, params: Dict):
        super(HyperRopeViT, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        self.num_classes = data_params.get("num_classes", 13)
        self.patch_size = data_params.get("patch_size", 5)
        self.spectral_size = data_params.get("spectral_size", 171)
        self.data_sign = data_params.get("data_sign", "Plastic")

        self.depth = net_params.get("depth", 1)
        self.heads = net_params.get("heads", 8)
        self.mlp_intermediate_dim = net_params.get("mlp_intermediate_dim", 256)
        self.kernel = net_params.get('kernal', 3)
        self.padding = net_params.get('padding', 1)
        self.dropout_rate = net_params.get("dropout", 0)
        self.conv2d_out = 64
        self.dim = net_params.get("dim", 64)
        self.dim_heads = self.dim // self.heads
        
        self.conv2d_features = self._create_conv2d_features()
        self.dropout = nn.Dropout(0.1)
        
        self.rope_mixed = True
        self.rope_theta = 10.0

        self.freqs = self._init_rope_freqs()
        self.register_coords_xy_buffers()

        self.local_trans_pixel = RoPETransformer(
            dim=self.dim, depth=self.depth, heads=self.heads, 
            dim_heads=self.dim_heads, mlp_dim=self.mlp_intermediate_dim, dropout=self.dropout_rate
        )

        self.center_weight = nn.Parameter(torch.ones(self.depth, 1, 1) * 0.001)

        self.classifier_mlp = self._create_classifier_mlp()

    def _create_conv2d_features(self):
        '''
        if self.data_sign in ['IndianPine', 'Pavia', 'Houston']:
            return nn.Sequential(
                ZeroMaskedConv2d(
                    in_channels=self.spectral_size, 
                    out_channels=self.conv2d_out, 
                kernel_size=self.kernel, 
                padding=self.padding
            ),
            nn.BatchNorm2d(self.conv2d_out),
            nn.ReLU()
            )
        '''
        #elif self.data_sign == "Plastic":
        return nn.Sequential(
                nn.Conv2d(in_channels=self.spectral_size, out_channels=self.conv2d_out, 
                          kernel_size=(self.kernel, self.kernel), padding=(self.padding, self.padding)),
                nn.BatchNorm2d(self.conv2d_out),
                nn.ReLU(),
            )

    def _init_rope_freqs(self):
        if self.rope_mixed:
            freqs = []
            for _ in range(self.depth):
                freqs.append(
                    RoPEUtils.init_2d_freqs(dim=self.dim // self.heads, num_heads=self.heads, theta=self.rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, self.depth, -1)  #shape is [2, depth, num_heads * 4]
            return nn.Parameter(freqs.clone(), requires_grad=True)
        else:
            print("RoPE mixed is not set to True")
            return None

    def register_coords_xy_buffers(self):
        t_x, t_y = RoPEUtils.init_coords_xy(end_x=self.patch_size, end_y=self.patch_size)
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)

    def _create_classifier_mlp(self):
        linear_dim = self.dim * 2
        return nn.Sequential(
            nn.Linear(self.dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, self.num_classes),
        )
        
    def _compute_freqs_cis(self):
        if self.rope_mixed:
            return RoPEUtils.compute_mixed_cis(self.freqs, self.freqs_t_x, self.freqs_t_y, num_heads=self.heads)
        else:
            return None

    def encoder_block(self, x):
        b, s, w, h = x.shape
        x_pixel = self.conv2d_features(x)
        x_pixel = rearrange(x_pixel, 'b s w h -> b (w h) s')
        x_pixel = self.dropout(x_pixel)

        freqs_cis = self._compute_freqs_cis()

        x_pixel, x_center_list = self.local_trans_pixel(x_pixel, freqs_cis=freqs_cis)
        x_center_tensor = torch.stack(x_center_list, dim=0)
        logit_pixel = torch.sum(x_center_tensor * self.center_weight, dim=0)
        #reduce_x = torch.mean(x_pixel, dim=1) This is if we take the mean of all the 25 pixels in patch instead of center pixel
        return logit_pixel



    def forward(self, x):
        '''
        x: (Batch, Spectral, Width, Height)
        
        '''
        logit_x = self.encoder_block(x)
        return self.classifier_mlp(logit_x)