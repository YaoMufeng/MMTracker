#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import cv2
import numpy as np

from yolox.utils import adjust_box_anns

import random

from ..data_augment import box_candidates,random_perspective,random_perspective_visdrone, augment_hsv
from .datasets_wrapper import Dataset
import pdb

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup

    def __len__(self):
        return len(self._dataset)

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, img_info, _ = self._dataset.pull_item(index)

                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)


                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]

                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                '''
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
                '''
                
                mosaic_labels = mosaic_labels[mosaic_labels[:, 0] < 2 * input_w]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 2] > 0]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 1] < 2 * input_h]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 3] > 0]
                
            #augment_hsv(mosaic_img)
            # print("======")
            #print("before perspective",mosaic_labels[...,6].min(),mosaic_labels[...,6].max())
            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove
            #print("after perspective",mosaic_labels[...,6].min(),mosaic_labels[...,6].max())
            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            #print("2",mosaic_labels.shape)
            
            if self.enable_mixup and not len(mosaic_labels) == 0:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            
            #print("3",mosaic_labels.shape)
            #print("before mixup",mosaic_labels[...,6].min(),mosaic_labels[...,6].max())
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            #print("after mixup",padded_labels[...,6].min(),padded_labels[...,6].max())
            #print("4",padded_labels.shape)
            img_info = (mix_img.shape[1], mix_img.shape[0])
            #print("mosaic shape:",padded_labels.shape,"preproc:",self.preproc)
            #print("------")
            return mix_img, padded_labels, img_info, np.array([idx])

        else:
            #print("here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self._dataset._input_dim = self.input_dim
            img, label, img_info, id_ = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            #print("non mosaic shape:",label.shape)
            return img, label, img_info, id_

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        '''
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        '''
        cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
        cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            id_labels = cp_labels[keep_list, 5:6].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            flow_labels = cp_labels[keep_list,6:7].copy()
            
            #print("cplabelshape:",cp_labels.shape,box_labels.shape,cls_labels.shape,id_labels.shape,flow_labels.shape)
            #pdb.set_trace()
            labels = np.hstack((box_labels, cls_labels, id_labels,flow_labels))
            # remove outside bbox
            labels = labels[labels[:, 0] < target_w]
            labels = labels[labels[:, 2] > 0]
            labels = labels[labels[:, 1] < target_h]
            labels = labels[labels[:, 3] > 0]
            #print(origin_labels.shape,labels.shape)
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        #print("mosaic labels:",origin_labels[0,6])
        return origin_img, origin_labels


from loguru import logger
class MosaicDetectionVisdrone(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup

    def __len__(self):
        return len(self._dataset)

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        #print("visdrone mosaic getitem!")
        if self.enable_mosaic:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels,ignore_mask, img_info, _ = self._dataset.pull_item(index)



                #ignore_mask = img_info["ignore_mask"]
                # if isinstance(_labels0,dict):
                #     _labels = _labels0["res"]
                # else:
                #     _labels = _labels0

                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                ignore_mask = cv2.resize(ignore_mask, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_NEAREST)
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                    mosaic_ignore_mask = np.full((input_h * 2, input_w * 2),0, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                #print(ignore_mask.shape,mosaic_ignore_mask.shape)
                mosaic_ignore_mask[l_y1:l_y2, l_x1:l_x2] = ignore_mask[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                '''
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
                '''
                
                mosaic_labels = mosaic_labels[mosaic_labels[:, 0] < 2 * input_w]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 2] > 0]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 1] < 2 * input_h]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 3] > 0]
                
            #augment_hsv(mosaic_img)
            # print("======")
            #print("before perspective",mosaic_labels[...,6].min(),mosaic_labels[...,6].max())
            mosaic_img,mosaic_ignore_mask,mosaic_labels = random_perspective_visdrone(
                mosaic_img,
                mosaic_ignore_mask,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove
            #print("after perspective",mosaic_labels[...,6].min(),mosaic_labels[...,6].max())
            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            #print("2",mosaic_labels.shape)
            
            if self.enable_mixup and not len(mosaic_labels) == 0:
                mosaic_img,mosaic_ignore_mask, mosaic_labels = self.mixup(mosaic_img,mosaic_ignore_mask,mosaic_labels, self.input_dim)
            
            #print("3",mosaic_labels.shape)
            #print("before mixup",mosaic_labels[...,6].min(),mosaic_labels[...,6].max())
            mix_img,mix_ignore_mask,padded_labels = self.preproc(mosaic_img,mosaic_ignore_mask,mosaic_labels, self.input_dim)
            #print("after mixup",padded_labels[...,6].min(),padded_labels[...,6].max())
            #print("4",padded_labels.shape)
            img_info = (mix_img.shape[1], mix_img.shape[0])
            #print("mosaic shape:",padded_labels.shape,"preproc:",self.preproc)
            #print("------")
            return mix_img,mix_ignore_mask,padded_labels, img_info, np.array([idx])

        else:
            self._dataset._input_dim = self.input_dim
            img, label, ignore_mask,img_info, id_ = self._dataset.pull_item(idx)
            #ignore_mask = img_info["ignore_mask"]
            #logger.info(self.preproc)
            #logger.info(img.shape,ignore_mask.shape)
            #print("before:",img.shape,ignore_mask.shape)
            img, ignore_mask,label = self.preproc(img,ignore_mask,label, self.input_dim)
            #print("after:",img.shape,ignore_mask.shape)
            return img,ignore_mask,label,img_info, id_

    def mixup(self, origin_img,origin_ignore_mask,origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels,new_ignore_mask, cp_info, _ = self._dataset.pull_item(cp_index)
        #new_ignore_mask = cp_info["ignore_mask"]

        #pdb.set_trace()
        
        cp_ignore_mask = np.zeros(input_dim)
        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(img,(int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),interpolation=cv2.INTER_LINEAR,).astype(np.float32)
        resized_new_ignore_mask = cv2.resize(new_ignore_mask,(int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),interpolation=cv2.INTER_NEAREST,).astype(np.float32)
        
        

        cp_img[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_img
        cp_ignore_mask[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_new_ignore_mask

        #print("cpignoremask1:",cp_img.shape,cp_ignore_mask.shape)
        new_size = (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor))
        cp_img = cv2.resize(cp_img,new_size,)
        cp_ignore_mask = cv2.resize(cp_ignore_mask,new_size,)

        #print("cpignoremask2:",cp_img.shape,cp_ignore_mask.shape)

        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]
            cp_ignore_mask = cp_ignore_mask[:, ::-1]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        padded_ignore_mask = np.zeros((max(origin_h, target_h), max(origin_w, target_w))).astype(np.uint8)
        padded_ignore_mask[:origin_h, :origin_w] = cp_ignore_mask

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset: y_offset + target_h, x_offset: x_offset + target_w]
        padded_cropped_ignore_mask = padded_ignore_mask[y_offset: y_offset + target_h, x_offset: x_offset + target_w]

        cp_bboxes_origin_np = adjust_box_anns(cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h)
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1])
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        '''
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        '''
        cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
        cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            id_labels = cp_labels[keep_list, 5:6].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            flow_labels = cp_labels[keep_list,6:7].copy()
            
            #print("cplabelshape:",cp_labels.shape,box_labels.shape,cls_labels.shape,id_labels.shape,flow_labels.shape)
            #pdb.set_trace()
            labels = np.hstack((box_labels, cls_labels, id_labels,flow_labels))
            # remove outside bbox
            labels = labels[labels[:, 0] < target_w]
            labels = labels[labels[:, 2] > 0]
            labels = labels[labels[:, 1] < target_h]
            labels = labels[labels[:, 3] > 0]
            #print(origin_labels.shape,labels.shape)
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
            origin_ignore_mask = origin_ignore_mask.astype(np.float32)*0.5 + 0.5*padded_cropped_ignore_mask.astype(np.float32)

            origin_ignore_mask[origin_ignore_mask > 0] = 255
        #print("mosaic labels:",origin_labels[0,6])
        return origin_img,origin_ignore_mask,origin_labels




class MosaicDetectionVisdrone2(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup

    def __len__(self):
        return len(self._dataset)

    @Dataset.resize_getitem
    def __getitem__(self, idx):

        self._dataset._input_dim = self.input_dim
        #print(self._dataset.pull_item)
        items1 , items2,flow = self._dataset.pull_item(idx)

        img, label, ignore_mask,img_info, id_ = items1
        #print(self.preproc)
        #print(flow.shape,img.shape)
        img, ignore_mask,flow,label = self.preproc(img,ignore_mask,label, self.input_dim,flow=flow)
        #print(img.shape,flow.shape)

        img2, label2, ignore_mask2,img_info2, id_2 = items2
        img2, ignore_mask2,label2 = self.preproc(img2,ignore_mask2,label2, self.input_dim)

        return (img,ignore_mask,label,img_info, id_),(img2,ignore_mask2,label2,img_info2, id_2),flow