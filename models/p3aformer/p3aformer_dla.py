## TransCenter has code derived from
## (1) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
## (2) 2020 Philipp Bergmann, Tim Meinhardt. (GNU General Public License v3.0 Licence: https://github.com/phil-bergmann/tracking_wo_bnw)
## (3) 2020 Facebook. (Apache License Version 2.0: https://github.com/facebookresearch/detr/)
## (4) 2020 Xingyi Zhou.(MIT License: https://github.com/xingyizhou/CenterTrack)
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2017 NVIDIA CORPORATION. (Apache License, Version 2.0: https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)
## (3) 2019 Simon Niklaus. (GNU General Public License v3.0: https://github.com/sniklaus/pytorch-liteflownet)
## (4) 2018 Tak-Wai Hui. (Copyright (c), see details in the LICENSE file: https://github.com/twhui/LiteFlowNet)
import torch.nn as nn
from dcn_v2 import DCN
import math
import numpy as np
import torch.nn.functional as F

BN_MOMENTUM = 0.1


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

DLA_NODE = {
    'dcn': (DeformConv, DeformConv)
}


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUpV3(nn.Module):
    # bilinear upsampling version of IDA
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUpV3, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # no params

        for i in range(0, len(channels)):
            c = channels[i]
            if i == 0:
                node = node_type[1](c, o)
            else:
                node = node_type[1](c, c)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(endp-1, startp, -1):
            upsample = self.up
            layers[i] = upsample(layers[i])  # ch 256-> 256
            node = getattr(self, 'node_' + str(i))
            layers[i-1] = node(layers[i] + layers[i - 1])
        layers[startp] = self.up(layers[startp])  # 256=>256
        node = getattr(self, 'node_' + str(startp))
        layers[startp] = node(layers[startp])
        return [layers[startp]]


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x
