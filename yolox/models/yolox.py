#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
import ipdb

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

        # ipdb.set_trace()

    def forward(self, x, targets=None,ignore_mask=None,optical_flow=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x,ignore_mask=ignore_mask,optical_flow=optical_flow)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs,optical_flow=optical_flow)

        return outputs



import torch.nn.functional as F
import torch
from .warp import Warp
import numpy as np
from .network_blocks import BaseConv

class YOLOX2(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None,fpn_channels = [128,256,512]):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

        self.downscale = 8

        self.warp_layer = Warp()
        self.prev_fpn_outs = None
        # torch.autograd.set_detect_anomaly(True)
        
        
        self.fpn_fuse1 = BaseConv(fpn_channels[0]*2,fpn_channels[0],ksize=3,stride=1,bias=False,act='relu')
        self.fpn_fuse2 = BaseConv(fpn_channels[1]*2,fpn_channels[1],ksize=3,stride=1,bias=False,act='relu')
        self.fpn_fuse3 = BaseConv(fpn_channels[2]*2,fpn_channels[2],ksize=3,stride=1,bias=False,act='relu')
        self.fpn_fuses = [self.fpn_fuse1,self.fpn_fuse2,self.fpn_fuse3]
        self.spatial_attention1 = nn.Sequential(nn.Conv2d(fpn_channels[0]*2,1,kernel_size=1,padding=0,bias=False),nn.Sigmoid())
        self.spatial_attention2 = nn.Sequential(nn.Conv2d(fpn_channels[1]*2,1,kernel_size=1,padding=0,bias=False),nn.Sigmoid())
        self.spatial_attention3 = nn.Sequential(nn.Conv2d(fpn_channels[2]*2,1,kernel_size=1,padding=0,bias=False),nn.Sigmoid())
        self.spatial_attentions = [self.spatial_attention1,self.spatial_attention2,self.spatial_attention3]

    def warp(self,optical_flow,fpn_outs,fpn_outs2,h1,w1,scales):
        scales_1 = 1 / scales
        _,_,flow_h,flow_w = optical_flow.shape
        sh,sw = h1 / flow_h , w1 / flow_w
        flow_downscale = F.interpolate(optical_flow,(h1,w1),mode='bilinear')
        flow_downscale[:,0:1,:,:] = flow_downscale[:,0:1,:,:]* sw
        flow_downscale[:,1:2,:,:] = flow_downscale[:,1:2,:,:]* sh
        _,_,flow_h,flow_w = flow_downscale.shape

        flows = [flow_downscale]
        
        for i in range(1,len(fpn_outs)):
            fpn_h,fpn_w = int(flow_h / scales[i]) , int(flow_w /scales[i])
            current_flow = F.interpolate(flow_downscale,(fpn_h,fpn_w),mode='bilinear')
            current_flow[:,0:1,:,:] = current_flow[:,0:1,:,:]*scales_1[i]
            current_flow[:,1:2,:,:] = current_flow[:,1:2,:,:]*scales_1[i]
            flows.append(current_flow)
        #print(len(flows))

        fpn_outs2_new = []
        prev_fpn_warps = []
        valid_masks = []
        for i in range(len(fpn_outs)):
            prev_fpn,post_fpn = fpn_outs[i],fpn_outs2[i]
            prev_fpn_warp,valid_mask = self.warp_layer(prev_fpn,flows[i])
            prev_fpn_warp=prev_fpn_warp.to(dtype=post_fpn.dtype)
            prev_fpn_warps.append(prev_fpn_warp)
            valid_masks.append(valid_mask)
            #post_fpn[valid_mask] = 0.5*post_fpn[valid_mask] + 0.5*prev_fpn_warp[valid_mask]

        for i in range(len(fpn_outs)):
            prev_fpn_warp,post_fpn,valid_mask = prev_fpn_warps[i],fpn_outs2[i],valid_masks[i]

            # post_fpn = prev_fpn_warp + post_fpn
            
            cat = torch.cat([prev_fpn_warp,post_fpn],dim=1)
            fpn_fuse = self.fpn_fuses[i](cat)
            spatial_attention = self.spatial_attentions[i](cat)
            post_fpn = post_fpn + spatial_attention*fpn_fuse


            # post_fpn = post_fpn*(1-spatial_attention) + spatial_attention*fpn_fuse
            #post_fpn = (0.5*post_fpn + 0.5*prev_fpn_warp)*valid_mask + (1-valid_mask)*post_fpn

            fpn_outs2_new.append(post_fpn)
        return fpn_outs2_new
    def forward(self,x,x2=None,targets=None,targets2=None,ignore_mask=None,ignore_mask2=None,optical_flow=None):
        # fpn output content features of [dark3, dark4, dark5]
        b,c,h,w = x.shape
        
        if x2 is not None:
            fpn_outs_2 = self.backbone(torch.cat([x,x2],dim=0))
            fpn_outs,fpn_outs2 = [],[]
            for i in range(len(fpn_outs_2)):
                #print(fpn_outs_2[i].shape)
                fpn_outs.append(fpn_outs_2[i][0:b,...])
                fpn_outs2.append(fpn_outs_2[i][b:,...])
        else:
            fpn_outs = self.backbone(x)
        
        # fpn_outs = self.backbone(x)
        # fpn_outs2 = self.backbone(x2)

        h1,w1 = int(h/self.downscale),int(w/self.downscale)
        #sh,sw = h1/h,w1/w
        scales = np.array([1,2,4])
        scales_1 = 1 / scales
        
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x,ignore_mask=ignore_mask)
            loss2, iou_loss2, conf_loss2, cls_loss2, l1_loss2, num_fg2 = 0,0,0,0,0,0
            if x2 is not None:
                if optical_flow is not None:
                    fpn_outs2 = self.warp(optical_flow,fpn_outs,fpn_outs2,h1,w1,scales)
                loss2, iou_loss2, conf_loss2, cls_loss2, l1_loss2, num_fg2 = self.head(fpn_outs2, targets2, x2,ignore_mask=ignore_mask2)
            outputs = {
                "total_loss": loss+loss2,
                "iou_loss": iou_loss+iou_loss2,
                "l1_loss": l1_loss+l1_loss2,
                "conf_loss": conf_loss+conf_loss2,
                "cls_loss": cls_loss+cls_loss2,
                "num_fg": num_fg+num_fg2,
            }
        else:
            ## v4 origin
            # if self.prev_fpn_outs is not None and optical_flow is not None:
            #     fpn_outs = self.warp(optical_flow,self.prev_fpn_outs,fpn_outs,h1,w1,scales)
            # outputs = self.head(fpn_outs)         
            # self.prev_fpn_outs = fpn_outs


            # v4 two frame only
            fpn_before_warp = []
            for fpn in fpn_outs:
                fpn_before_warp.append(fpn.clone())
            #fpn_before_warp = fpn_outs.clone()
            if self.prev_fpn_outs is not None and optical_flow is not None:
                fpn_outs = self.warp(optical_flow,self.prev_fpn_outs,fpn_outs,h1,w1,scales)
            outputs = self.head(fpn_outs)         
            self.prev_fpn_outs = fpn_before_warp
            
            
            ##############
        return outputs


from mamba_ssm import Mamba

class CrossMambaBlock(nn.Module):
    def __init__(self,d_model=16,**kwargs):
        super(CrossMambaBlock, self).__init__()

        self.mamba = Mamba(d_model=d_model,d_state=16,d_conv=4,expand=2)
        self.mamba2 = Mamba(d_model=d_model,d_state=16,d_conv=4,expand=2)
    
    def forward(self,x):
        short = x
        b,c,h,w = x.shape
        x1 = x.view(b*h,w,c)#逐列扫描
        x2 = x.view(b*w,h,c)
        # print(x.shape,self.mamba)
        x1 = self.mamba(x1)
        x2 = self.mamba2(x2)
        x1 = x1.view(b,c,h,w)
        x2 = x2.view(b,c,h,w)
        # x = x.view(b,c,h,w)
        # x = self.mamba2(x)
        out = x1 + x2 + short
        # x = x.view(b,c,h,w) + short

        return out



# class YOLOX_Mamba(nn.Module):
#     """
#     YOLOX model module. The module list is defined by create_yolov3_modules function.
#     The network returns loss values from three YOLO layers during training
#     and detection results during test.
#     """

#     def __init__(self, backbone=None, head=None):
#         super().__init__()
#         if backbone is None:
#             backbone = YOLOPAFPN()
#         if head is None:
#             head = YOLOXHead(80)

#         self.backbone = backbone
#         self.head = head

#         # ipdb.set_trace()

#     def forward(self, x, targets=None,ignore_mask=None,optical_flow=None):
#         # fpn output content features of [dark3, dark4, dark5]
#         fpn_outs = self.backbone(x)

#         if self.training:
#             assert targets is not None
#             loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x,ignore_mask=ignore_mask,optical_flow=optical_flow)
#             outputs = {
#                 "total_loss": loss,
#                 "iou_loss": iou_loss,
#                 "l1_loss": l1_loss,
#                 "conf_loss": conf_loss,
#                 "cls_loss": cls_loss,
#                 "num_fg": num_fg,
#             }
#         else:
#             outputs = self.head(fpn_outs,optical_flow=optical_flow)

#         return outputs

class YOLOX_Mamba(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None,width = 0.5):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

        self.width = width

        # width = 0.5

        in_channels = [256,512,1024]
        c1=int(in_channels[0] * width)
        c2=int(in_channels[1] * width)
        c3=int(in_channels[2] * width)

        self.mamba1 = CrossMambaBlock(d_model=c1)
        self.mamba2 = CrossMambaBlock(d_model=c2)
        self.mamba3 = CrossMambaBlock(d_model=c3)


    def forward(self, x, targets=None,ignore_mask=None,optical_flow=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        dark3,dark4,dark5 = fpn_outs

        # ipdb.set_trace()
        dark3 = self.mamba1(dark3)
        dark4 = self.mamba2(dark4)
        dark5 = self.mamba3(dark5)

        fpn_outs = (dark3,dark4,dark5)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x,ignore_mask=ignore_mask,optical_flow=optical_flow)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs,optical_flow=optical_flow)

        return outputs