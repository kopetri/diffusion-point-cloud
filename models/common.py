import torch
from torch.nn import Module, Linear, MultiheadAttention, GroupNorm, LayerNorm, Conv1d
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn.functional as F
import math

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)


class CrossAttention(Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.cross_attention = MultiheadAttention(embed_dim, num_heads)

    def forward(self, point_features, text_features):
        #rint(point_features.shape, text_features.shape)
        # Transpose point_features to match MultiheadAttention input shape
        #project points to context_dim 
        point_features = point_features.permute(0, 2, 1)

        point_features = point_features.permute(2, 0, 1)  # [seq_len, batch, context_dim]
        text_features = text_features.permute(1, 0, 2)    # [seq_len, batch, context_dim]
        
        
        
        # Use text_features as key and value for conditioning on text
        attended_features, _ = self.cross_attention(query=point_features, key=text_features, value=text_features)

        # Transpose back to [batch, seq_len, embed_dim]
        return attended_features.permute(1, 0, 2)


class SelfAttention(Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.self_attention = MultiheadAttention(embed_dim, num_heads)

    def forward(self, features): # [batch, seq_len, embed_dim]
        features = features.permute(1, 0, 2)  # [seq_len, batch, embed_dim]
        attended_features, _ = self.self_attention(query=features, key=features, value=features)
        attended_features = attended_features.permute(1, 0, 2)
        return attended_features
    
class AttentionBlock(Module):
    def __init__(self, n_head: int, channels: int, context_dim: int = 768):
        super().__init__()

        self.project_points = not context_dim == channels
        if self.project_points:
            self.proj = Conv1d(context_dim, channels, 1)
        
        self.groupnorm = GroupNorm(4, channels, eps=1e-6)
        self.conv_input = Conv1d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = LayerNorm(channels)
        self.attention_1 = SelfAttention(channels, n_head)
        self.layernorm_2 = LayerNorm(channels)

        self.attention_2 = CrossAttention(channels, n_head)
        self.layernorm_3 = LayerNorm(channels)
        self.linear_geglu_1  = Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = Linear(4 * channels, channels)

        self.conv_output = Conv1d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, point_features, text_features):
        residue_long = point_features
        # point_features [B, point_dim, seq2]
        if self.project_points:
            text_features = self.proj(text_features.permute(0, 2, 1)).permute(0, 2, 1)
        # text_feat [B, seq, text_dim]
        x = self.groupnorm(point_features)
        x = self.conv_input(x)
    
        x = x.transpose(-1, -2) # [B, N, C]
        
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, text_features)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)  # (n, c, hw)
        x = self.conv_output(x)
        
        return x + residue_long
    
if __name__ == '__main__':
    B = 4
    N = 1024
    n_heads = 4
    point_dim = 96
    n_emb = 768
    
    attn = AttentionBlock(n_heads, point_dim, n_emb).cuda()
    
    point_emb = torch.randn((B, point_dim, N)).cuda()
    text_emb = torch.randn((B, 77, n_emb)).cuda()
    
    print(point_emb.shape, text_emb.shape)
    x = attn(point_emb, text_emb)
    print(x.shape)
    