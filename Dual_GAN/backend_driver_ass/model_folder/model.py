# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import unetPart
sys.path.append('..')


class ROI_Block(nn.Module):
    def __init__(self, inplanes, downsample=0, Res=0):
        super(ROI_Block, self).__init__()
        self.downsample = downsample
        self.Res = Res
        self.conv1 = nn.Sequential(
            nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False, groups=inplanes),
            nn.BatchNorm1d(inplanes),
        )
        self.se = unetPart.SELayer(inplanes, reduction=16)
        if self.downsample:
            self.down = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(planes)
                 )

    def forward(self, x):
        b, c, r, l = x.size()
        x = x.reshape(b, -1, l)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        if self.Res:
            if self.downsample is not None:
                x = self.down(x)
            out += x
        out = out.reshape(b, -1, r, l)
        return out


class BasicBlock(nn.Module):
    def __init__(self,  inplanes, out_planes, stride=2, downsample=1, Res=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
                 )
        self.downsample = downsample
        self.Res = Res

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        return out


class DNet(nn.Module):
    def __init__(self, inChannel=4, out=1):
        super(DNet, self).__init__()
        self.inChannel = inChannel
        self.out = out
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.inChannel),
            nn.Conv2d(self.inChannel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=2, downsample=0, Res=0),
            ROI_Block(64 * 8),
            BasicBlock(64, 128, stride=2, downsample=0, Res=0)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=0, Res=0),
            ROI_Block(256 * 2),
            BasicBlock(256, 512, stride=2, downsample=0, Res=0)
        )
        self.AV = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, out),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.AV(out)
        out = torch.squeeze(out)
        out = self.fc(out)
        return out


class EncodeDecode(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(EncodeDecode, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.bn = nn.BatchNorm2d(n_channels)
        self.inc = unetPart.DoubleConv(n_channels, 16)
        self.down1 = unetPart.Down(16, 32)
        self.ROI_Block1 = ROI_Block(32*32)
        self.down2 = unetPart.Down(32, 64)
        self.down3 = unetPart.Down(64, 128)
        self.ROI_Block2 = ROI_Block(128 * 8)
        self.down4 = unetPart.Down(128, 128)
        self.up1 = unetPart.Up2(128, 64, bilinear)
        self.ROI_Block3 = ROI_Block(64 * 8)
        self.up2 = unetPart.Up2(64, 16, bilinear)
        self.up3 = unetPart.Up2(16, 8, bilinear)
        self.ROI_Block4 = ROI_Block(8 * 32)
        self.up4 = unetPart.Up2(8, 8, bilinear)
        self.outc = unetPart.OutConv(8, n_classes)

    def forward(self, x):
        x = self.bn(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.ROI_Block1(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.ROI_Block2(x4)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.ROI_Block3(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.ROI_Block4(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits


class rPPGNet(nn.Module):
    def __init__(self):
        super(rPPGNet, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.down1 = BasicBlock(3, 32, 2, downsample=1)
        self.ROI_Block1 = ROI_Block(32 * 32)
        self.down2 = BasicBlock(32, 64, 2, downsample=1)
        self.down3 = BasicBlock(64, 128, 2, downsample=1)
        self.ROI_Block2 = ROI_Block(128 * 8)
        self.down4 = BasicBlock(128, 256, 2, downsample=1)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 64, [2, 1], downsample=1),
            ROI_Block(64 * 2)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 16, [1, 1], downsample=1),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(16, 4, [2, 1], downsample=1),
            ROI_Block(4 * 1)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(4, 4, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(4, 1, [1, 1], downsample=1),
        )

    def forward(self, x):
        x = self.bn(x)
        x2 = self.down1(x)
        x2 = self.ROI_Block1(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.ROI_Block1(x4)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x).squeeze(dim=1)
        #x=(x-torch.min(x))/((torch.max(x))-torch.min(x))
        return x

class rPPGNet2(nn.Module):
    def __init__(self):
        super(rPPGNet2, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.down = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlock(64, 64, 2, downsample=1),
            BasicBlock(64, 64, 1, downsample=1),
            BasicBlock(64, 128, 2, downsample=1),
            BasicBlock(128, 128, 1, downsample=1),
            BasicBlock(128, 256, 2, downsample=1, Res=1),
            BasicBlock(256, 256, 1, downsample=1),
            BasicBlock(256, 512, 2, downsample=1, Res=1),
            BasicBlock(512, 512, 1, downsample=1),
        )
        self.HR = nn.Sequential(BasicBlock(256, 256, 2, downsample=1, Res=1),
                                BasicBlock(256, 256, 1, downsample=0, Res=0),
                                BasicBlock(256, 512, 2, downsample=1, Res=1),
                                BasicBlock(512, 512, 1, downsample=0, Res=0)
                                )
        self.av = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 64, [2, 1], downsample=1),
            ROI_Block(64 * 2),
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 16, [1, 1], downsample=1),
            nn.ConvTranspose2d(16, 16, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(16, 4, [2, 1], downsample=1),
            ROI_Block(4 * 1),
            nn.ConvTranspose2d(4, 4, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(4, 1, [1, 1], downsample=1),
        )

    def forward(self, x):
        x = self.bn(x)
        f = self.down(x)
        rPPG = self.up(f[:, 0:256]).squeeze(dim=1)
        return rPPG