
import torch.utils.data as data
import os
import cv2
import numpy as np
from tqdm import tqdm
import ipdb

import torch
import torch.nn as nn


import argparse
import os
import platform
import sys
from pathlib import Path
from loguru import logger
import torch

from trackers.mc_bytetracker import MCByteTracker
# from utils.write_results import write_results

import ipdb
import cv2

from tqdm import tqdm

from spatial_correlation_sampler import SpatialCorrelationSampler
import numpy as np

def read_flow(name: str) -> np.ndarray:
    """Read flow file with the suffix '.flo'.

    This function is modified from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    Copyright (c) 2011, LMB, University of Freiburg.

    Args:
        name (str): Optical flow file path.

    Returns:
        ndarray: Optical flow
    """

    with open(name, 'rb') as f:

        header = f.read(4)
        if header.decode('utf-8') != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape(
            (height, width, 2))

    return flow

def process_img(img,means = np.array([0.574859,0.5284268,0.48245227]),
                stds = np.array([0.20307621,0.16383478,0.16327296])):
    img = img.astype(np.float32)
    img /= 255
    img -= means
    img /= stds
    # ipdb.set_trace()
    return img


class MotionDataset(data.Dataset):
    def __init__(self,data_roots,txt_root,img_size,flow_foldername = None):
        
        # self.data_root = data_root
        self.flow_foldername = flow_foldername
        self.img_size = img_size
       
        self.datalen = 0

        self.txt_root = txt_root
        zfill_count = 7
        score_thr = 0.01
        self.datas = []

        if isinstance(data_roots,str):
            data_roots = [data_roots]

        for data_root in data_roots:
            seqnames = os.listdir(data_root)
            for seqname in seqnames:
                seqfolder = os.path.join(data_root,seqname)

                if not os.path.isdir(seqfolder):
                    continue
                # img_folder = os.path.join(seqfolder,"img1")
                img_folder = seqfolder
                imgnames = sorted(os.listdir(img_folder))

                for i in range(len(imgnames)-1):
                    imgname1,imgname2 = imgnames[i],imgnames[i+1]
                    imgpath1 = os.path.join(img_folder,imgname1)
                    imgpath2 = os.path.join(img_folder,imgname2)
                    self.datas.append({"imgpath1":imgpath1,"imgpath2":imgpath2})
    def is_valid_path(self,imgpath1,imgpath2):
        folder1 = imgpath1.split('/')[-2]
        folder2 = imgpath2.split('/')[-2]

        if folder1 != folder2:
            return False

        _,imgname1 = os.path.split(imgpath1)
        _,imgname2 = os.path.split(imgpath2)
        name1,ext = os.path.splitext(imgname1)
        name2,ext = os.path.splitext(imgname2)

        if int(name1) + 1 != int(name2):
            return False

        return True


    def __getitem__(self,index):

        data1 = self.datas[index]
        imgpath1 = data1["imgpath1"]
        imgpath2 = data1["imgpath2"]

        img1 = cv2.imread(imgpath1)
        img2 = cv2.imread(imgpath2)

        src_h,src_w,_ = img1.shape

        img1 = cv2.resize(img1,self.img_size,cv2.INTER_LINEAR)
        img2 = cv2.resize(img2,self.img_size,cv2.INTER_LINEAR)

        img1 = process_img(img1)
        img2 = process_img(img2)

        w,h = self.img_size
        w0,h0 = int(w/8),int(h/8)
        # is_obj_center = np.zeros([h0,w0])

        img1 = torch.from_numpy(img1.transpose(2,0,1)).float()
        img2 = torch.from_numpy(img2.transpose(2,0,1)).float()

        scale_h,scale_w = src_h / h , src_w / w
        datas = {"data1":img1,"data2":img2,"scale_hw":np.array([scale_h,scale_w])}

        if self.flow_foldername is not None:
            flow_path = imgpath1.replace("sequences",self.flow_foldername).replace(".jpg",".flo")
            flow = read_flow(flow_path)
            flow_h,flow_w,_ = flow.shape
            flwo_scale_w,flwo_scale_h = w0 / flow_w , h0 / flow_h
            flow = cv2.resize(flow,(w0,h0),cv2.INTER_LINEAR)
            flow[:,:,0] = flow[:,:,0] * flwo_scale_w
            flow[:,:,1] = flow[:,:,1] * flwo_scale_h
            sw,sh = w0 / src_w , h0 / src_h
            flow = flow.transpose(2,0,1)
            datas["flow"] = flow
        return datas
    def __len__(self):
        return len(self.datas)


class MotionDatasetUAVDT(data.Dataset):
    def __init__(self,data_root,txt_root,img_size,flow_foldername = None):
        
        self.data_root = data_root
        self.flow_foldername = flow_foldername
        self.img_size = img_size
        seqnames = os.listdir(data_root)
        self.datalen = 0

        self.txt_root = txt_root
        zfill_count = 7
        score_thr = 0.01
        self.datas = []

        print(data_root)
        for seqname in seqnames:
            seqfolder = os.path.join(data_root,seqname)
            img_folder = os.path.join(seqfolder,"img1")
            imgnames = sorted(os.listdir(img_folder))

            for i in range(len(imgnames)-1):
                imgname1,imgname2 = imgnames[i],imgnames[i+1]
                imgpath1 = os.path.join(img_folder,imgname1)
                imgpath2 = os.path.join(img_folder,imgname2)
                self.datas.append({"imgpath1":imgpath1,"imgpath2":imgpath2})


    def is_valid_path(self,imgpath1,imgpath2):
        folder1 = imgpath1.split('/')[-2]
        folder2 = imgpath2.split('/')[-2]

        if folder1 != folder2:
            return False

        _,imgname1 = os.path.split(imgpath1)
        _,imgname2 = os.path.split(imgpath2)
        name1,ext = os.path.splitext(imgname1)
        name2,ext = os.path.splitext(imgname2)

        if int(name1) + 1 != int(name2):
            return False

        return True


    def __getitem__(self,index):

        data1 = self.datas[index]
        imgpath1 = data1["imgpath1"]
        imgpath2 = data1["imgpath2"]

        img1 = cv2.imread(imgpath1)
        img2 = cv2.imread(imgpath2)

        src_h,src_w,_ = img1.shape

        img1 = cv2.resize(img1,self.img_size,cv2.INTER_LINEAR)
        img2 = cv2.resize(img2,self.img_size,cv2.INTER_LINEAR)

        img1 = process_img(img1)
        img2 = process_img(img2)

        w,h = self.img_size
        w0,h0 = int(w/8),int(h/8)
        img1 = torch.from_numpy(img1.transpose(2,0,1)).float()
        img2 = torch.from_numpy(img2.transpose(2,0,1)).float()

        scale_h,scale_w = src_h / h , src_w / w
        datas = {"data1":img1,"data2":img2,"scale_hw":np.array([scale_h,scale_w])}

        if self.flow_foldername is not None:
            flow_path = imgpath1.replace("img1",self.flow_foldername).replace(".jpg",".flo")
            flow = read_flow(flow_path)

            flow_h,flow_w,_ = flow.shape
            flwo_scale_w,flwo_scale_h = w0 / flow_w , h0 / flow_h

            flow = cv2.resize(flow,(w0,h0),cv2.INTER_LINEAR)
            
            flow[:,:,0] = flow[:,:,0] * flwo_scale_w
            flow[:,:,1] = flow[:,:,1] * flwo_scale_h

            flow = flow.transpose(2,0,1)
            datas["flow"] = flow
        return datas
    def __len__(self):
        return len(self.datas)


if __name__ == "__main__":

    root = "/home/ymf/datas/visdrone/VisDrone2019-MOT-train/sequences"
    txt_root = "/home/ymf/datas/visdrone/VisDrone2019-MOT-train/annotations"

    dataset = MotionDataset(root,txt_root,(1088,608),flow_foldername="flow_gma_768x1024")

    dataloader = data.DataLoader(dataset, batch_size=8, num_workers=4,shuffle=True)


    for i,data in enumerate(tqdm(dataloader)):
        img1,img2,flow = data["img1"],data["img2"],data["flow"]
        print(img1.shape,img2.shape,flow.shape)