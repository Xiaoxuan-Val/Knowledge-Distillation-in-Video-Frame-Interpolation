import math
import numpy as np

import torch
import torch.nn as nn

from .common import *


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3,n_resgroups =5, n_resblocks=12):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth)

        relu = nn.LeakyReLU(0.2, False)
        self.n_resgrous = n_resgroups
        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation(n_resgroups, n_resblocks, in_channels * (4**depth), act=relu)
        
    def forward(self, x1, x2):#([16, 3, 256, 256]) ([16, 3, 256, 256])
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats1 = self.shuffler(x1)#([16, 192, 32, 32])
        feats2 = self.shuffler(x2)#([16, 192, 32, 32])

        feats,feats_layers,feats_layers_1 = self.interpolate(feats1, feats2)#([16, 192, 32, 32])

        return feats,feats_layers,feats_layers_1


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self, depth=3,n_resgroups =5, n_resblocks=12):
        super(CAIN, self).__init__()
        
        self.encoder = Encoder(in_channels=3, depth=depth,n_resgroups=n_resgroups, n_resblocks=n_resblocks)
        self.decoder = Decoder(depth=depth)
    
    def get_channel_num(self):
        return [192,192,192,192,192]

    def forward(self, x1, x2):
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)
        """
        if not self.training:
            paddingInput, paddingOutput = InOutPaddings(x1)
            x1 = paddingInput(x1)
            x2 = paddingInput(x2)
        """
        feats,feats_layers,feats_layers_1 = self.encoder(x1, x2)
        out = self.decoder(feats)
        """
        if not self.training:
            out = paddingOutput(out)
        """
        mi = (m1 + m2) / 2
        out += mi

        return feats_layers,feats_layers_1,out

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False)]
         #nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def distillation_loss(source, target):
    
    loss = torch.nn.functional.mse_loss(source, target, reduction="sum")

    return loss

class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        #self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])
        #self.Connectors = nn.ModuleList([nn.LayerNorm([192,32,32]) for i in t_channels])
        #teacher_bns = t_net.get_bn_before_relu()
        #margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        #for i, margin in enumerate(margins):
        #    self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x1,x2):
        with torch.no_grad():
            t_feats,t_feats_1, t_out = self.t_net(x1,x2)
        s_feats,s_feats_1, s_out = self.s_net(x1,x2)
        feat_num = len(t_feats)
        
        loss_distill = 0
        for i in range(feat_num):
            #s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach())#/ 2 ** (feat_num - i - 1)
            loss_distill += distillation_loss(s_feats_1[i], t_feats_1[i].detach())
        return s_out, loss_distill
