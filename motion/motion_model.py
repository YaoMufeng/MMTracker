import torch
import torch.nn as nn

import torch.nn.functional as F
import argparse
import os
import platform
import sys
from pathlib import Path
from loguru import logger
import torch

from trackers.mc_bytetracker import MCByteTracker
from trackers.mc_motiontracker import MCMotionTracker

# from utils.write_results import write_results

import ipdb
import cv2

from tqdm import tqdm

from spatial_correlation_sampler import SpatialCorrelationSampler
import numpy as np
import glob
from collections import OrderedDict
import motmetrics as mm

from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info
from torch.nn.parallel import DistributedDataParallel as DDP
from mamba_ssm import Mamba

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        #pdb.set_trace()
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2*max_displacement+1
        self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)
        
    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.BatchNorm2d(out_channels), 
        nn.LeakyReLU(0.1, inplace=True)
    )

class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out

class MyDecoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(MyDecoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = BaseConv(in_channels, 96, 3, 1)
        self.conv2 = BaseConv(96, 96, 3, 1, groups=groups)
        self.conv3 = BaseConv(96, 96, 3, 1, groups=groups)
        self.conv4 = BaseConv(96, 96, 3, 1, groups=groups)
        self.conv5 = BaseConv(96, 64, 3, 1)
        self.conv6 = BaseConv(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out


class LightDecoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(LightDecoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = BaseConv(in_channels, 96, 3, 1)
        self.conv2 = BaseConv(96, 96, 3, 1, groups=groups)
        # self.conv3 = BaseConv(96, 96, 3, 1, groups=groups)
        # self.conv4 = BaseConv(96, 96, 3, 1, groups=groups)
        # self.conv5 = BaseConv(96, 64, 3, 1)
        self.conv6 = BaseConv(96, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv2(self.conv1(x))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv6(out), self.groups)
            # out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(out)
        return out



from yolox.models.network_blocks import BaseConv

class CrossMambaBlock(nn.Module):
    def __init__(self,channels,**kwargs):
        super(CrossMambaBlock, self).__init__()

        # self.layer = layer
        self.mamba1 = Mamba(d_model=channels,d_state=16,d_conv=4,expand=2)
        self.mamba2 = Mamba(d_model=channels,d_state=16,d_conv=4,expand=2)

        self.cat_conv = nn.Conv2d(2*channels,channels,kernel_size=1,padding=0)
    
    def forward(self,x):
        short = x

        b,c,h,w = x.shape
        x1 = x.view(b*h,w,c)#逐列扫描
        x2 = x.view(b*w,h,c)#逐行扫描

        x1,x2 = self.mamba1(x1),self.mamba2(x2)
        x1,x2 = x1.view(b,c,h,w),x2.view(b,c,h,w)

        out = self.cat_conv(torch.cat([x1,x2],dim=1))
        # x = x.view(b,c,h,w)
        # x = self.mamba2(x)
        out = out+short
        # x = x.view(b,c,h,w) + short

        return out

class MotionMamba(nn.Module):
    def __init__(self):
        super().__init__()

        self.corr = Correlation(4)
        self.index = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 28, 29, 30, 
                                   31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 44, 46, 47, 48, 49, 50, 51, 52, 
                                   56, 57, 58, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80])
        groups = 4
        act = "softshrink"
        self.decoder3 = LightDecoder(64,groups)

        self.up_conv1 = BaseConv(52,64,3,1,act=act)
        self.mamba1 = CrossMambaBlock(64)

        self.cat_conv1 = BaseConv(52+64,64,3,1,act=act)

        self.up_conv2 = BaseConv(64,64,3,1,act=act)
        self.mamba2 = CrossMambaBlock(64)

        self.cat_conv2 = BaseConv(52+64,64,3,1,act=act)
        self.mamba3 = CrossMambaBlock(64)

        self.decoder1 = nn.Sequential(BaseConv(52,64,3,1,act=act),
                                      nn.Conv2d(64,2,3,padding=1,bias=False))
        self.decoder2 = nn.Sequential(BaseConv(52,64,3,1,act=act),
                                      nn.Conv2d(64,2,3,padding=1,bias=False))
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight,a=0.1, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
    def forward(self,x1,x2):
        f1c3,f1c2,f1c1 = x1
        f2c3,f2c2,f2c1 = x2

        # ipdb.set_trace()
        cv1 = torch.index_select(self.corr(f1c1, f2c1), dim=1, index=self.index.to(f1c1).long())
        cv1_up = F.interpolate(cv1,scale_factor=2,mode='bilinear')
        cv1_up = self.up_conv1(cv1_up)
        cv1_up = self.mamba1(cv1_up)

        cv2 = torch.index_select(self.corr(f1c2, f2c2), dim=1, index=self.index.to(f1c1).long())

        # ipdb.set_trace()
        cv2_cat = torch.cat([cv1_up,cv2],dim=1)
        cv2_cat = self.cat_conv1(cv2_cat)
        cv2_up = F.interpolate(cv2_cat,scale_factor=2,mode='bilinear')
        cv2_up = self.up_conv2(cv2_up)
        cv2_up = self.mamba2(cv2_up)
        
        cv3 = torch.index_select(self.corr(f1c3, f2c3), dim=1, index=self.index.to(f1c1).long())
        cv3_cat = torch.cat([cv2_up,cv3],dim=1)
        cv3_cat = self.cat_conv2(cv3_cat)

        cv3_cat = self.mamba3(cv3_cat)
        cv3_dec = self.decoder3(cv3_cat)
        
        
        if self.training:
            flow1 = self.decoder1(cv1)
            flow2 = self.decoder2(cv2)
            return flow1,flow2,cv3_dec
        return cv3_dec



class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction='mean')

        self.neg_rate = 0.01
    
    def forward(self,pred,label,mask):

        # ipdb.set_trace()
        mask = mask.unsqueeze(1).repeat((1,2,1,1))
        posloss = self.mse(pred[mask==1],label[mask==1])
        negloss = self.mse(pred[mask==0],label[mask==0])

        return posloss + self.neg_rate*negloss


