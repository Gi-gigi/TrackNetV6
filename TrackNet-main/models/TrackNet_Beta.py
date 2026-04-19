

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import SemPriorAttn
from pytorch_wavelets import DWTForward



def weight_init(module):
    for _, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
       
        elif isinstance(m, (nn.ReLU, nn.MaxPool2d, nn.AvgPool2d,
                            nn.AdaptiveAvgPool2d, nn.Dropout, nn.Identity, nn.Flatten)):
            pass

        elif isinstance(m, nn.Sequential):
            weight_init(m)

        elif hasattr(m, "initialize") and callable(getattr(m, "initialize")):
            m.initialize()

        else:
            weight_init(m)
            

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


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        feats = list(models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features.children()) 
        
        self.conv1 = nn.Sequential( conv3x3(9, 64),nn.BatchNorm2d(64),nn.ReLU(),*feats[3:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        
        E1 = self.conv1(x)
        E2 = self.conv2(E1)
        E3 = self.conv3(E2)
        E4 = self.conv4(E3)
        E5 = self.conv5(E4)

        return E1,E2,E3,E4,E5


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv3x3 = ConvNormLayer(in_channels, out_channels, 3, 1, padding=1, act=None)
        self.conv1x1 = ConvNormLayer(in_channels, out_channels, 1, 1, padding=0, act=None)
        self.activation = nn.ReLU()

    def forward(self, x):
        if hasattr(self, 'conv'):
            out_feat = self.conv(x)
        else:
            out_feat = self.conv3x3(x) + self.conv1x1(x)
        return self.activation(out_feat)


class FeatureCorrector(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(FeatureCorrector, self).__init__()

        hidden_channels = int(out_channels * expansion)
        self.fuse_conv = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)

        self.rep_blocks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])

        self.skip_conv = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)

        if hidden_channels != out_channels:
            self.out_conv = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.out_conv = nn.Identity()

    def forward(self, feat_a, feat_b):
        concat_feat = torch.cat([feat_a, feat_b], dim=1)
        main_path = self.fuse_conv(concat_feat)
        main_path = self.rep_blocks(main_path)
        skip_path = self.skip_conv(concat_feat)
        return self.out_conv(main_path + skip_path)


class TrackNetBeta(nn.Module):
    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.bkbone = VGG()
        self.decoder = TrackDecoder(out_channels=out_channels)
        self.initialize()

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape
        l1, l2, l3, l4, l5 = self.bkbone(x)
        skips = [l1, l2, l3, l4, l5]
        output = self.decoder(x,skips) 
        return output
    def initialize(self):
        weight_init(self.decoder)


class DWTrans(nn.Module):

    def __init__(self, in_channels, n=1):
        super(DWTrans, self).__init__()

        self.identity_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * n,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.dwt_forward = DWTForward(J=1, wave='haar')
        self.encode_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * n, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def _transformer(self, low_freq, high_freq_list):
        concat_list = []
        high_freq = high_freq_list[0]
        concat_list.append(low_freq)
        for i in range(3):
            concat_list.append(high_freq[:, :, i, :, :])
        return torch.cat(concat_list, 1)

    def forward(self, x):
        x_in = x
        low_freq, high_freq_list = self.dwt_forward(x)
        dwt_feat = self._transformer(low_freq, high_freq_list)
        x = self.encode_conv(dwt_feat)
        residual = self.identity_conv(x_in)
        out = torch.add(x, residual)
        return out


class AffineFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.affine_mlp = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 1))

    def forward(self, fused_feat, orig_feat):
        avg_feat = self.global_avg_pool(orig_feat)
        affine_params = self.affine_mlp(avg_feat)
        scale, shift = torch.chunk(affine_params, 2, dim=1)
        return fused_feat * (1 + scale) + shift


class TrackDecoder(nn.Module):
    def __init__(self,out_channels: int):

        super().__init__()
        self.num_classes = out_channels
        encoder_stages = 4

        stages      = []
        upsamples   = []
        pre_conv1x1 = []
        seg_layers  = []
        output_channels = [64, 128, 256, 512, 512]
        decode_channels = [64, 64, 64, 64, 64]

        for s in range(1, encoder_stages+1):  
            input_features_below = output_channels[-s]
            input_features_skip  = decode_channels[-s]
            upsamples.append(upsample2d(input_features_skip, input_features_skip, 3))
            pre_conv1x1.append(nn.Conv2d(input_features_below, input_features_skip, kernel_size=1, stride=1, bias=True))
        
        for s in range(1, encoder_stages+1):   
            if s != encoder_stages:
                cd_conv1 = pre_conv1x1[s]
                af_conv1x1 = upsamples[s]
            else: af_conv1x1,cd_conv1=None,None
            CSCP = SemPriorAttn(in_dim=decode_channels[-s], reduce='max') 
            PICC = FeatureCorrector(decode_channels[-s]*2,decode_channels[-s])
            stages.append(LLM_decoder(encoder_stages,s,pre_conv1x1[s-1],cd_conv1,af_conv1x1,CSCP,PICC))
            seg_layers.append(conv1x1(input_features_skip, 3, bias=True))

        self.conv1 = conv1x1(output_channels[-1], decode_channels[-1], bias=True)
        self.conv2 = ConvBNReLU(decode_channels[-1], decode_channels[-1])

        self.conv3 = conv1x1(output_channels[-2], decode_channels[-1], bias=True)
        self.conv4 = conv1x1(output_channels[-3], decode_channels[-1], bias=True)
        self.conv5 = conv1x1(output_channels[-4], decode_channels[-1], bias=True)
        self.DWT1 = DWTrans(in_channels=decode_channels[-2], n=1)
        self.DWT2 = DWTrans(in_channels=decode_channels[-3], n=1)
        self.DWT3 = DWTrans(in_channels=decode_channels[-4], n=1)
        self.GRB = AffineFusion(decode_channels[-1])
        self.down2 = nn.Conv2d(decode_channels[-2],decode_channels[-2],kernel_size=2,stride=2,padding=0,bias=False)
        self.down3 = nn.Conv2d(decode_channels[-2],decode_channels[-2],kernel_size=2,stride=4,padding=0,bias=False)
        self.conv6 = conv1x1(decode_channels[-2]*3, decode_channels[-1], bias=True)

        self.stages     = nn.ModuleList(stages)
        self.upsamples  = nn.ModuleList(upsamples)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.out_conv = nn.Conv2d(12, 3, kernel_size=1)

    def forward(self, ori_input,skips):
        
        ori_spatial = ori_input.shape[2:]
        y1 = self.conv2(self.conv1(skips[-1]))

        g1 = self.DWT1(self.conv3(skips[-2]))
        g2 = self.down2(self.DWT2(self.conv4(skips[-3])))
        g3 = self.down3(self.DWT3(self.conv5(skips[-4])))
        y2 = torch.cat([g1, g2], dim=1)
        y2 = torch.cat([g3, y2], dim=1)
        y = self.GRB(self.conv6(y2),y1)

        seg_outputs = []
        f = []
        for s in range(0,len(self.stages)):
            y,f_current = self.stages[s](f,skips[1:],y) 
            f.append(f_current)
            seg_outputs.append( F.interpolate(self.seg_layers[s-1](y), size=ori_spatial, mode='bilinear', align_corners=True))

        seg_outputs = seg_outputs[::-1]
        final_seg = self.out_conv(torch.concat(seg_outputs, dim=1))
        if self.training:
            r = [torch.sigmoid(final_seg),torch.sigmoid(seg_outputs[0]),torch.sigmoid(seg_outputs[1]),torch.sigmoid(seg_outputs[2]),torch.sigmoid(seg_outputs[3])]
        else:
            r = torch.sigmoid(final_seg)
        return r


class upsample2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(upsample2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,bias=False)

    def forward(self, x, y):
        y = self.conv(y)     
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


class LLM_decoder(torch.nn.Module):
    def __init__(self,stages_num,stage,g1,g2,g3,F1,F2):
        super(LLM_decoder,self).__init__()
        self.decoder_num = stages_num
        self.current_stage = stage
        self.g1 = g1                        
        self.g2 = g2                       
        self.g3 = g3                       
        self.F1 = F1                        
        self.F2 = F2                       
        self.step = 1/self.decoder_num            
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def nmODE_ex(self,x,y):       
        return -y + self.F1(self.g1(x), y)  

    def nmODE_im(self,x,y):
        return -self.g3(x, y)+self.F2(self.g2(x),self.g3(x, y))
        
    def AB_step1_order1(self,x1,y1):
        f1 = self.nmODE_ex(x1,y1)
        y2 = y1 + self.step*f1
        return y2,f1

    def AM_step1_order2(self,x1,y1,x2):
        y2_pre,f1 = self.AB_step1_order1(x1,y1)
        f2_pre = self.nmODE_im(x2,y2_pre)         
        y2 = self.up(y1) + (self.step/2)*(self.up(f1) + f2_pre)  
        return y2,self.up(f1)

    def AB_steps2_order2(self,f1,x2,y2):
        f2 = self.nmODE_ex(x2,y2)
        y3 = y2 + (self.step/2)*(3*f2 - f1)
        return y3,f2

    def AM_steps2_order3(self,f1,x2,y2,x3):
        y3_pre,f2 = self.AB_steps2_order2(f1,x2,y2)
        f3_pre = self.nmODE_im(x3,y3_pre)
        y3 = self.up(y2) + (self.step/12)*(5*f3_pre + 8*self.up(f2) - self.up(f1))
        return y3,self.up(f2)

    def AB_steps3_order3(self,f1,f2,x3,y3):
        f3 = self.nmODE_ex(x3,y3)
        y4 = y3 + (self.step/12)*(23*f3-16*f2+5*f1)
        return y4,f3

    def AM_steps3_order4(self,f1,f2,x3,y3,x4):
        y4_pre,f3 = self.AB_steps3_order3(f1,f2,x3,y3)
        f4_pre = self.nmODE_im(x4,y4_pre)
        y4 = self.up(y3) + (self.step/24)*(9*f4_pre + 19*self.up(f3) - 5*self.up(f2) +self.up(f1))
        return y4,self.up(f3)

    def AB_steps4_order4(self,f1,f2,f3,x4,y4):
        f4 = self.nmODE_ex(x4,y4)
        y5 = y4 + (self.step/24)*(55*f4 - 59*f3 + 37*self.up(f2) - 9*self.up(f1))
        return y5,f4

    
    def forward(self,f,x,y):

        if self.current_stage == self.decoder_num:
            return self.AB_steps4_order4(f[0],f[1],f[2],x[-self.current_stage],y)
        else:
            if self.current_stage == 1:
                return self.AM_step1_order2(x[-self.current_stage],y,x[-self.current_stage-1])
            elif self.current_stage == 2:
                return self.AM_steps2_order3(f[0],x[-self.current_stage],y,x[-self.current_stage-1])
            elif self.current_stage == 3:
                f[0] = F.interpolate(f[0], size=x[-self.current_stage].size()[2:], mode='bilinear', align_corners=False)
                return self.AM_steps3_order4(f[0],f[1],x[-self.current_stage],y,x[-self.current_stage-1])


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TrackNet Debug Script')
    parser.add_argument('--device', type=str, default='cuda:1', help='GPU device to use')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("creating test dataset...")
    x = torch.randn(args.batch_size, 9, 288, 512).to(device)
    print(f"Input shape: {x.shape}")
    print(f"device: {x.device}")
    

    print("initialing model...")
    net = TrackNetBeta()
    net = net.to(device)
    net.eval()
    output = net(x)
    print(output.size())
