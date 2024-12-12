#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from loguru import logger
class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def iou_loss(pred, target):
    assert pred.shape[0] == target.shape[0]

    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    tl = torch.max(
        (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
    )
    br = torch.min(
        (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
    )
    area_p = torch.prod(pred[:, 2:], 1)
    area_g = torch.prod(target[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    iou = (area_i) / (area_p + area_g - area_i + 1e-16)
    loss = 1 - iou ** 2
    
    return loss


class FocalIOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(FocalIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        loss = torch.pow(iou,2)*(1-iou**2)
        # if self.loss_type == "iou":
        #     loss = 1 - iou ** 2
        # elif self.loss_type == "giou":
        #     c_tl = torch.min(
        #         (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        #     )
        #     c_br = torch.max(
        #         (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        #     )
        #     area_c = torch.prod(c_br - c_tl, 1)
        #     giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
        #     loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class FocalEIOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou",alpha=0.25,gamma=0.5):
        super(FocalEIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        # tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        # br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

        top1,left1,bottom1,right1 = pred[:,0] - pred[:,2]/2,pred[:,1] - pred[:,3]/2,pred[:,0] + pred[:,2]/2,pred[:,1] + pred[:,3]/2
        top2,left2,bottom2,right2 = target[:,0] - target[:,2]/2,target[:,1] - target[:,3]/2,target[:,0] + target[:,2]/2,target[:,1] + target[:,3]/2

        w1,h1,w2,h2 = (right1 - left1),(bottom1 - top1),(right2 - left2),(bottom2 - top2)
        cx1,cy1,cx2,cy2 = (left1 + right1)/2,(top1 + bottom1)/2,(left2 + right2)/2,(top2 + bottom2)/2

        p2 = (cx1 - cx2)**2 + (cy1 - cy2)**2
        pw,ph = (w1-w2)**2,(h1-h2)**2
        top,left,bottom,right = torch.min((top1),(top2)),torch.min((left1),(left2)),torch.max((bottom1),(bottom2)),torch.max((right1),(right2))
        hc2,wc2 = (top - bottom)**2,(left - right)**2
        

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        top0,left0,bottom0,right0 = torch.max((top1),(top2)),torch.max((left1),(left2)),torch.min((bottom1),(bottom2)),torch.min((right1),(right2))
        w0,h0 = right0 - left0,bottom0 - top0
        w0[w0<0]=0
        h0[h0<0]=0
        inter = w0*h0
        iou = (inter) / (area_p + area_g - inter + 1e-16)
        eiou = 1 - iou + p2/(hc2+wc2 + 1e-16) + pw/(wc2 + 1e-16) + ph/(hc2 + 1e-16)
        loss = torch.pow(iou,2)*eiou
        

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

def my_binary_cross_entropy(inputs,targets,class_weights=None,reduction="none"):

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if class_weights is not None:
        #print(loss.shape,class_weights.shape)
        loss = class_weights*loss
        #print(loss.shape)
    return loss

def flow_ldam_v2(inputs, targets,flow_score,scale=10,multi_class=False,class_weights=None):
    positive_mask = targets > 0
    
    if multi_class:
        num_samples,num_classes = positive_mask.shape
        flow_score_ce = flow_score.repeat(num_classes,1).reshape(num_samples,num_classes)[positive_mask]
    else:
        flow_score_ce = flow_score

    x_start = 5

    delta = torch.zeros_like(targets).type_as(flow_score_ce)
    flow_delta = scale*(torch.sigmoid((flow_score_ce-x_start)/scale))

    delta[positive_mask] = flow_delta

    loss = my_binary_cross_entropy(inputs - delta, targets, reduction="none",class_weights=class_weights)
    return loss

def flow_ldam_v3(inputs, targets,flow_score,scale=10,multi_class=False,class_weights=None):
    positive_mask = targets > 0
    
    if multi_class:
        num_samples,num_classes = positive_mask.shape
        flow_score_ce = flow_score.repeat(num_classes,1).reshape(num_samples,num_classes)[positive_mask]
    else:
        flow_score_ce = flow_score

    x_start = 5

    delta = torch.zeros_like(targets).type_as(flow_score_ce)
    flow_delta = 0.5*scale*(torch.sigmoid((flow_score_ce-x_start)/scale))

    delta[positive_mask] = flow_delta

    loss = my_binary_cross_entropy(inputs - delta, targets, reduction="none",class_weights=class_weights)
    return loss

def flow_ldam(inputs, targets,flow_score,scale=10,multi_class=False,class_weights=None):
    positive_mask = targets > 0
    
    if multi_class:
        num_samples,num_classes = positive_mask.shape
        flow_score_ce = flow_score.repeat(num_classes,1).reshape(num_samples,num_classes)[positive_mask]
    else:
        flow_score_ce = flow_score

    delta = torch.zeros_like(targets).type_as(flow_score_ce)
    flow_delta = scale*(flow_score_ce.sigmoid()-0.5)
    # if multi_class:
    #     pdb.set_trace()
    delta[positive_mask] = flow_delta

    # print(delta.min(),delta.max())
    #pdb.set_trace()
    loss = my_binary_cross_entropy(inputs - delta, targets, reduction="none",class_weights=class_weights)
    return loss



import numpy as np
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=2):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target,class_weights=None):
        index = torch.zeros_like(target, dtype=torch.float32)

        index[target > 0] = 1
        #print(index.shape,target.shape)
        #index.scatter_(1, target.data.view(-1, 1).long(), 1)
        #index.scatter_(1, target.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        #batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        #batch_m = batch_m.view((-1, 1))
        batch_m = index_float*self.m_list
        # print(x.shape,batch_m.shape)
        # pdb.set_trace()
        x_m = x - batch_m
        #output = torch.where(target > 0, x_m, x)
        
        #loss = F.cross_entropy(self.s*x_m, target, weight=self.weight,reduction="none")
        loss = my_binary_cross_entropy(self.s*x_m, target, reduction="none")
        return loss
