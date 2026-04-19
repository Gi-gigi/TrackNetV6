
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AnyAttention(nn.Module):
    def __init__(self, dim, qkv_bias=True):
        super(AnyAttention, self).__init__()

        self.norm_q, self.norm_k = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = 1.0 / (dim ** 0.5)

    def get_qkv(self, q, k ):
        q, k = self.to_q(self.norm_q(q)), self.to_k(self.norm_k(k))
        return q, k

    def forward(self, query=None, key=None, orq=None, target=None):
        q, k = self.get_qkv(query, key)
        B, _, _ = q.size()
        H, W = target[0],target[1]

        attn = torch.matmul(q, k.permute(0, 2, 1))
        attn_max = torch.max(attn, dim=-1)[0]
        attn_avg = torch.mean(attn, dim=-1)
        attn = (attn_max + attn_avg) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn.view(B, -1, H, W)
        out = out * orq

        return out


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class AtrousTokenAggregator(nn.Module):

    def __init__(
        self,
        in_dim,
        rates=(1, 2, 4),
        out_scales=(1, 3, 6),
        reduce='max',  
    ):
        super().__init__()
        self.out_scales = out_scales
        self.reduce = reduce

        self.branches = MultiDilatedPath(in_dim)


    def _reduce_to_grid(self, y, th, tw, reduce):
        if reduce == 'avg':
            y = F.adaptive_avg_pool2d(y, (th, tw))
        elif reduce == 'max':
            y = F.adaptive_max_pool2d(y, (th, tw)) 
        elif reduce == 'area':
            y = F.interpolate(y, size=(th, tw), mode='area') 
        elif reduce == 'bilinear':
            y = F.interpolate(y, size=(th, tw), mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unknown reduce mode: {reduce}")
        return y

    def forward(self, x): 
        b, c, _, _ = x.shape
        targets = [(s, s) for s in self.out_scales]
        tokens = []

        y = self.branches(x)
        for (th, tw) in targets:
            y_red = self._reduce_to_grid(y, th, tw, self.reduce)
            tok = y_red.view(b, c, th * tw)        
            tokens.append(tok)

        kv_tokens = torch.cat(tokens, dim=2)
        return kv_tokens


class MultiDilatedPath(nn.Module):
    def __init__(self, channels):
        super(MultiDilatedPath, self).__init__()
        self.dconv_d1 = DSConv3x3(channels, channels, stride=1, dilation=1)
        self.dconv_d2 = DSConv3x3(channels, channels, stride=1, dilation=2)
        self.dconv_d4 = DSConv3x3(channels, channels, stride=1, dilation=4)
        self.dconv_d8 = DSConv3x3(channels, channels, stride=1, dilation=8)

        self.fuse_conv = DSConv3x3(channels, channels, relu=False)

    def forward(self, x):
        feat_d1 = self.dconv_d1(x)
        feat_d2 = self.dconv_d2(x + feat_d1)
        feat_d4 = self.dconv_d4(x + feat_d2)
        feat_d8 = self.dconv_d8(x + feat_d4)
        fused_feat = self.fuse_conv(feat_d1 + feat_d2 + feat_d4 + feat_d8)
        return fused_feat + x


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )
    def forward(self, x):
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, residual=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        self.residual = residual
        if use_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x1 = self.conv(x)
        if self.bn is not None:
            x1 = self.bn(x1)
        if self.residual and x1.shape[1] == x.shape[1]:
            x1 = x + x1
        if self.act is not None:
            x1 = self.act(x1)

        return x1


class SemPriorAttn(nn.Module):
    def __init__(self, in_dim,
                 atrous_rates=(1, 2, 4),
                 out_scales=(1, 3, 6),
                 reduce='max',           
                ):
        super(SemPriorAttn, self).__init__()

        drop_rate = 0.0
        self.sigmoid = nn.Sigmoid()
        self.attention = AnyAttention(in_dim)
        self.drop_path = DropPath(drop_prob=drop_rate)

        self.atrous_block = AtrousTokenAggregator(
            in_dim=in_dim,
            rates=atrous_rates,
            out_scales=out_scales,
            reduce=reduce,
        )

        self.query_conv = ConvBNReLU(in_dim, in_dim)
        self.kv_conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)

        self.block = nn.Sequential(
                    nn.Conv2d(in_dim, in_dim*2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_dim*2),
                    nn.GELU(),
                )

        self.block2 = nn.Sequential(
                    nn.Conv2d(in_dim*2, in_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_dim),
                    nn.GELU(),
                )

    def forward(self, X, Y, ds=True, scale=(18,32)):


        h_ds, w_ds = int(scale[0]), int(scale[1])

        Fea = X + Y
        _, _, h, w = Fea.shape

        # 1) Query-Y:
        if ds and Fea.shape[2:] != scale:
            Fea_scale = F.interpolate(Fea, size=(h_ds, w_ds), mode="bilinear") 
            Fea_scale = self.query_conv(Fea_scale) 
            Fea_query = rearrange(Fea_scale, "b c h w -> b (h w) c")  
        else:
            Fea_scale = self.query_conv(Fea) 
            Fea_query = rearrange(Fea_scale, "b c h w -> b (h w) c")

        # 2) Key/Value-X:
        kv_tokens = self.atrous_block(self.kv_conv(Fea)+ Fea) 
        kv_tokens = rearrange(kv_tokens, "b c n -> b n c")

        # 3) Cross-Attention
        attn_out = self.attention(query=Fea_query, key=kv_tokens, orq=Fea_scale, target=(h_ds,w_ds))

        # 4) restore
        if ds and Fea.shape[2:] != scale:

            feas_y = Fea + self.block2(self.drop_path(F.interpolate(self.block(attn_out), size=(h, w), mode="bilinear")))
        else:
            attn_out = self.block2(self.drop_path(self.block(attn_out)))
            feas_y = Fea + attn_out

        return feas_y

