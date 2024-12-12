#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.distributed as dist

from yolox.utils import synchronize

import random


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class DataPrefetcherVisdrone:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcherVisdrone._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input,self.next_ignore_mask,self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_ignore_mask = self.next_ignore_mask.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        ignore_mask = self.next_ignore_mask

        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if ignore_mask is not None:
            self.record_stream(ignore_mask)
        self.preload()
        return input,ignore_mask,target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

def random_resize(data_loader, exp, epoch, rank, is_distributed):
    tensor = torch.LongTensor(1).cuda()
    if is_distributed:
        synchronize()

    if rank == 0:
        if epoch > exp.max_epoch - 10:
            size = exp.input_size
        else:
            size = random.randint(*exp.random_size)
            size = int(32 * size)
        tensor.fill_(size)

    if is_distributed:
        synchronize()
        dist.broadcast(tensor, 0)

    input_size = data_loader.change_input_dim(multiple=tensor.item(), random_range=None)
    return input_size


class DataPrefetcherVisdrone2:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcherVisdrone2._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            (self.next_input,self.next_ignore_mask,self.next_target, _, _),(self.next_input2,self.next_ignore_mask2,self.next_target2, _, _),self.next_flow = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_flow = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_ignore_mask = self.next_ignore_mask.cuda(non_blocking=True)
            self.next_target2 = self.next_target2.cuda(non_blocking=True)
            self.next_ignore_mask2 = self.next_ignore_mask2.cuda(non_blocking=True)
            self.next_flow = self.next_flow.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input,input2 = self.next_input,self.next_input2
        target,target2 = self.next_target,self.next_target2
        ignore_mask,ignore_mask2 = self.next_ignore_mask,self.next_ignore_mask2
        flow = self.next_flow

        if input is not None:
            self.record_stream(input,input2)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
            target2.record_stream(torch.cuda.current_stream())
        if ignore_mask is not None:
            self.record_stream(ignore_mask,ignore_mask2)
        if flow is not None:
            flow.record_stream(torch.cuda.current_stream())
        

        
        self.preload()
        return (input,ignore_mask,target),(input2,ignore_mask2,target2),flow

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)
        self.next_input2 = self.next_input2.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input,input2):
        input.record_stream(torch.cuda.current_stream())
        input2.record_stream(torch.cuda.current_stream())