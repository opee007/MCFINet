import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.backbone import build_backbone
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d



class _DSPPModule(nn.Module):
    def __init__(self,inplanes, planes, num_layers, BatchNorm):
        super(_DSPPModule, self).__init__()
        self.layers = []
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        for i in range(1, num_layers):
            self.layers.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False))
            self.layers.append(BatchNorm(planes))
            self.layers.append(nn.ReLU(inplace=True))

        self.Dconv = nn.Sequential(*self.layers)

        self._init_weight()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.Dconv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class DSPP(nn.Module):
    def __init__(self, backbone, BatchNorm):
        super(DSPP, self).__init__()
        if 'resnet' in backbone:
            inplanes = 2048
        else:
            raise NotImplementedError
        self.conv1_1 = nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, bias=False)
        self.bn1_1 = BatchNorm(256)

        self.dspp1 = _DSPPModule(inplanes, 256, 2, BatchNorm=BatchNorm)
        self.dspp2 = _DSPPModule(inplanes, 256, 3, BatchNorm=BatchNorm)
        self.dspp3 = _DSPPModule(inplanes, 256, 4, BatchNorm=BatchNorm)
        # self.dspp4 = _DSPPModule(inplanes, 256, 5, BatchNorm=BatchNorm)
        # self.dspp5 = _DSPPModule(inplanes, 256, 6, BatchNorm=BatchNorm)


        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.last = nn.Conv2d(256, 6, 1)
        self._init_weight()

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.bn1_1(x1)
        x1 = self.relu(x1)
        x2 = self.dspp1(x)
        x3 = self.dspp2(x)
        x4 = self.dspp3(x)
        # k5 = self.dspp4(x)
        # k6 = self.dspp5(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)#x2, x3, x4,, k5, k6

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.dropout(x)
        x = self.last(x)

        return x#, x1, x2, x3, x4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class pspnet(nn.Module):
    def __init__(self, backbone='resnet-101', output_stride=8, num_classes=6,
                 sync_bn=True, freeze_bn=False):
        super(pspnet, self).__init__()
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # self.assp = _PSPHead(num_classes, BatchNorm)
        self.dssp = DSPP('resnet', BatchNorm)
    def forward(self, input):
        down1, down2, down3, down4, down5 = self.backbone(input)
        # x = self.assp(down5)
        x = self.dssp(down5)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


def _PSP1x1Conv(in_channels, out_channels, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        BatchNorm(out_channels),
        nn.ReLU(True)
    )


class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, BatchNorm):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, BatchNorm)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, BatchNorm)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, BatchNorm)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, BatchNorm)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class _PSPHead(nn.Module):
    def __init__(self, nclass, BatchNorm):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048, BatchNorm)
        self.block = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)
