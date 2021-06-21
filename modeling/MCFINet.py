import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.backbone import build_backbone
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class _MDCM(nn.Module):
    def __init__(self,inplanes, planes, num_layers, BatchNorm):
        super(_MDCM, self).__init__()
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



class MDCM(nn.Module):
    def __init__(self, backbone, BatchNorm):
        super(MDCM, self).__init__()
        if 'resnet' in backbone:
            inplanes = 2048
        else:
            raise NotImplementedError
        self.conv1_1 = nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, bias=False)
        self.bn1_1 = BatchNorm(256)

        self.MDCM1 = _MDCM(inplanes, 256, 2, BatchNorm=BatchNorm)
        self.MDCM2 = _MDCM(inplanes, 256, 3, BatchNorm=BatchNorm)
        self.MDCM3 = _MDCM(inplanes, 256, 4, BatchNorm=BatchNorm)
        # self.MDCM4 = _MDCM(inplanes, 256, 5, BatchNorm=BatchNorm)
        # self.MDCM5 = _MDCM(inplanes, 256, 6, BatchNorm=BatchNorm)


        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.bn1_1(x1)
        x1 = self.relu(x1)
        x2 = self.MDCM1(x)
        x3 = self.MDCM2(x)
        x4 = self.MDCM3(x)
        # k5 = self.MDCM4(x)
        # k6 = self.MDCM5(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)#x2, x3, x4,, k5, k6

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)#, x1, x2, x3, x4

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

def Conv1x1(in_channels, out_channels, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        BatchNorm(out_channels),
        nn.ReLU(True)
    )
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, BatchNorm):
        super(PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = Conv1x1(in_channels, out_channels, BatchNorm)
        self.conv2 = Conv1x1(in_channels, out_channels, BatchNorm)
        self.conv3 = Conv1x1(in_channels, out_channels, BatchNorm)
        self.conv4 = Conv1x1(in_channels, out_channels, BatchNorm)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class ARM(nn.Module):
    def __init__(self, down_inplanes, up_inplanes, planes, BatchNorm):
        super(ARM, self).__init__()
        self.conv_down = nn.Conv2d(down_inplanes, planes//4, kernel_size=1, bias=False)
        self.bn_down = BatchNorm(planes//4)
        self.conv1 = nn.Conv2d(planes//4+up_inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.bn1 = BatchNorm(planes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.SELayer = SELayer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, down, up):
        down = self.conv_down(down)
        down = self.bn_down(down)
        down = self.relu(down)
        up = F.interpolate(up, size=down.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([up, down], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        residual = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        we = self.avg_pool(x)
        x = we * x
        # x = self.SELayer(x)
        x = x +residual
        return x


class Decoder(nn.Module):
    def __init__(self, BatchNorm):
        super(Decoder, self).__init__()

        self.up4 = ARM(1024, 256, 512, BatchNorm)
        self.up3 = ARM(512, 512, 256, BatchNorm)
        self.up2 = ARM(256, 256, 256, BatchNorm)
        # self.up1 = ARM(64, 256, 256, BatchNorm)
        self._init_weight()

    def forward(self,  down1, down2, down3, down4, x):
        x = self.up4(down4, x)
        up4 = x
        x = self.up3(down3, x)
        up3 = x
        x = self.up2(down2, x)
        up2 = x
        # x = self.up1(down1, x)
        # up1 = x
        return up4, up3, up2 #, up1

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

class Mix(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super(Mix, self).__init__()
        self.last_conv = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
                                       )
        self._init_weight()

    def forward(self, up4, up3, up2):
        up4 = F.interpolate(up4, size=up2.size()[2:], mode='bilinear', align_corners=True)
        up3 = F.interpolate(up3, size=up2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([up4, up3, up2], dim=1)
        x = self.last_conv(x)
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


class MCFINet(nn.Module):
    def __init__(self, backbone='resnet-101', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(MCFINet, self).__init__()
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.mdcm = MDCM('resnet', BatchNorm)
        self.decoder = Decoder(BatchNorm)
        self.mix = Mix(num_classes, BatchNorm)
    def forward(self, input):
        down1, down2, down3, down4, down5 = self.backbone(input)
        x = self.mdcm(down5)
        up4, up3, up2 = self.decoder(down1, down2, down3, down4, x)
        x = self.mix(up4, up3, up2)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.dssp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


#------------------------------------------------------------------------------
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, BatchNorm=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, BatchNorm)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, BatchNorm)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, BatchNorm)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, BatchNorm)

        self.conv = nn.Conv2d(4096, 256, 1, bias=False)
        self.bn = BatchNorm(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def _PSP1x1Conv(in_channels, out_channels, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        BatchNorm(out_channels),
        nn.ReLU(True)
    )

#------------------------------------------------------------------------------
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)
class ASPP(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

#　-----------------------------------------------------
class CFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CFM, self).__init__()
        self.conv0 = nn.Conv2d(2048, 512, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(512)

        self.c15_1 = nn.Conv2d(in_channel, out_channel, kernel_size=15, stride=1, padding=7, bias=False)
        self.c11_1 = nn.Conv2d(in_channel, out_channel, kernel_size=11, stride=1, padding=5, bias=False)
        self.c7_1 = nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.c3_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.c15_2 = nn.Conv2d(in_channel, out_channel, kernel_size=15, stride=1, padding=7, bias=False)
        self.c11_2 = nn.Conv2d(in_channel, out_channel, kernel_size=11, stride=1, padding=5, bias=False)
        self.c7_2 = nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.c3_2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1_gpb = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(2560, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        input_size = x.size()[2:]

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x15_1 = self.c15_1(x)
        x15_1 = self.bn(x15_1)
        x15_1 = self.relu(x15_1)
        x15_2 = self.c15_2(x15_1)
        x15_2 = self.bn(x15_2)

        x11_1 = self.c11_1(x)
        x11_1 = self.bn(x11_1)
        x11_1 = self.relu(x11_1)
        x11_2 = self.c11_2(x11_1)
        x11_2 = self.bn(x11_2)

        x7_1 = self.c7_1(x)
        x7_1 = self.bn(x7_1)
        x7_1 = self.relu(x7_1)
        x7_2 = self.c7_2(x7_1)
        x7_2 = self.bn(x7_2)

        x3_1 = self.c3_1(x)
        x3_1 = self.bn(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.c3_2(x3_1)
        x3_2 = self.bn(x3_2)

        x_gp = self.avg_pool(x)
        x_gp = self.c1_gpb(x_gp)
        x_gp = self.bn(x_gp)
        x_gp = F.upsample(x_gp, size=input_size, mode='bilinear')

        out = torch.cat([x_gp, x15_2, x11_2, x7_2, x3_2], dim=1)
        x = self.conv1(out)
        x = self.bn1(x)
        x = self.relu(x)
        return x
