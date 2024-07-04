import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from resnet import resnet10 as Encoder
from collections import OrderedDict


class Expert(nn.Module):
    def __init__(self, base_filter,video_length):
        super(Expert, self).__init__()
        self.conv1 = ConvBlock3D(64, base_filter, 3, 1, 1, activation='relu', norm=None)
        self.res1 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.ra = RABlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.res2 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.gap = nn.AdaptiveAvgPool3d((video_length, 1, 1))
        self.cov1d = nn.Conv1d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1)

    # def __init__(self, base_filter):
    #     super(Expert, self).__init__()
    #     self.conv1 = ConvBlock3D(64, base_filter, 3, 1, 1, activation='relu', norm=None)
    #     self.res1 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
    #                               norm=None)
    #     self.ra = RABlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True)
    #     self.res2 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
    #                               norm=None)
    #     self.gap = nn.AdaptiveAvgPool3d((video_length, 1, 1))
    #     self.cov1d = nn.Conv1d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1)
    # def __init__(self, base_filter):
    #     super(Expert, self).__init__()
    #     self.base_filter = base_filter
    #     self.res1 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
    #                               norm=None)
    #     self.ra = RABlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True)
    #     self.res2 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
    #                               norm=None)

    # def set_video_length(self, video_length):
    #     self.video_length = video_length
    #     # Now that video_length is available, you can initialize layers that depend on it
    #     self.gap = nn.AdaptiveAvgPool3d((self.video_length, 1, 1))
    #     self.cov1d = nn.Conv1d(in_channels=self.base_filter, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        feat = self.conv1(input)
        feat = self.res1(feat)
        feat = self.ra(feat)
        feat = self.res2(feat)
        feat = self.gap(feat)
        feat = feat.squeeze(3)
        feat = feat.squeeze(3)
        feat = self.cov1d(feat)
        return feat

class Gating(nn.Module):
    def __init__(self, base_filter,video_length,num_expert):
        super(Gating, self).__init__()
        self.conv1=ConvBlock3D(base_filter, base_filter, 3, 1, 1, activation='relu', norm=None)
        self.conv2=ConvBlock3D(base_filter, base_filter, 3, 1, 1, activation='relu', norm=None)

        self.gap=nn.AdaptiveAvgPool3d((video_length, 1, 1))
        self.cov1d=nn.Conv1d(in_channels=base_filter, out_channels=4,kernel_size=3,stride=1, padding=1)
        self.act = nn.Softmax(dim=1)

    def forward(self, input):
        feat = self.conv1(input)
        feat=self.conv2(feat)
        feat=self.gap(feat)
        feat=feat.squeeze(3)
        feat=feat.squeeze(3)
        feat=self.cov1d(feat)
        feat = self.act(feat)
        return feat

# class REA(nn.Module):
#     def __init__(self, base_filter, video_length, num_expert):
#         super(REA, self).__init__()
#         self.num_expert=num_expert
#         self.experts=nn.ModuleList([Expert(base_filter, video_length) for _ in range(self.num_expert)])
#         self.gating=Gating(base_filter, video_length, num_expert)
#         self.encoder=Encoder()
#         self.conv1d=nn.Conv1d(in_channels=1, out_channels=1,kernel_size=3,stride=1, padding=1)


#     def freeze_model(self, model):
#         for child in model.children():
#             for param in child.parameters():
#                 param.requires_grad = False

#     def forward(self, input):
#         feat = self.encoder(input)
#         gates = self.gating(feat)
#         B, C,T, H, W = input.size()
#         index1=index2=[i for i in range(int(self.num_expert**0.5))]
#         count=0
#         init_rppg=torch.zeros((B,T), device=torch.device('cuda:0'))
#         for m in index1:
#             for n in index2:
#                 if count < gates.size(1):
#                     cutfeat = feat[:, :, :, int(H/(self.num_expert**0.5)*m):int(H/(self.num_expert**0.5)*(m+1)),
#                                     int(W/(self.num_expert**0.5)*n):int(W/(self.num_expert**0.5)*(n+1))]
#                     cutfeat = self.experts[count](cutfeat).squeeze(1)
#                     cut_rppg = torch.mul(cutfeat, gates[:, count, :])
#                     cut_rppg = cut_rppg.to(init_rppg.device)
#                     init_rppg += cut_rppg
#                     count += 1


#         init_rppg=init_rppg.unsqueeze(1)
#         conv1d_weight = self.conv1d.weight.to(torch.device('cuda:0'))
#         conv1d_bias = self.conv1d.bias.to(torch.device('cuda:0'))
#         rppg_fuse = F.conv1d(init_rppg, conv1d_weight, bias=conv1d_bias)
#         return rppg_fuse

class REA(nn.Module):
    def __init__(self, base_filter, num_expert):
        super(REA, self).__init__()
        self.num_expert = num_expert
        self.video_length = None
        video_length = 1801
        self.experts = nn.ModuleList([Expert(base_filter, video_length) for _ in range(self.num_expert)])
        self.gating = Gating(base_filter, video_length, num_expert)
        self.encoder = Encoder()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

    def set_video_length(self, video_length):
        self.video_length = video_length
        # Update the experts with the new video length
        for expert in self.experts:
            expert.set_video_length(video_length)

    def freeze_model(self, model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input, video_length):
        feat = self.encoder(input)
        gates = self.gating(feat)
        B, C, T, H, W = input.size()
        index1 = index2 = [i for i in range(int(self.num_expert**0.5))]
        count = 0
        init_rppg = torch.zeros((B, T), device=input.device)
        
        for m in index1:
            for n in index2:
                if count < gates.size(1):
                    cutfeat = feat[:, :, :, int(H / (self.num_expert**0.5) * m):int(H / (self.num_expert**0.5) * (m + 1)),
                                    int(W / (self.num_expert**0.5) * n):int(W / (self.num_expert**0.5) * (n + 1))]
                    cutfeat = self.experts[count](cutfeat).squeeze(1)
                    print(cutfeat.shape, 'cutfeat')
                    print(gates[:, count, :].shape, 'gates')
                    cut_rppg = torch.mul(cutfeat, gates[:, count, :])
                    init_rppg += cut_rppg
                    count += 1

        init_rppg = init_rppg.unsqueeze(1)
        conv1d_weight = self.conv1d.weight.to(input.device)
        conv1d_bias = self.conv1d.bias.to(input.device)
        rppg_fuse = F.conv1d(init_rppg, conv1d_weight, bias=conv1d_bias)
        return rppg_fuse
