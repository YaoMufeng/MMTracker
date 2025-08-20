import sys
sys.path.append('core')

import argparse
import os
import numpy as np
import torch

import datasets
from utils import frame_utils

from emd import EMD
from utils.utils import InputPadder, forward_interpolate
from tqdm import tqdm
import cv2
import ipdb
def write_flow(flow: np.ndarray, flow_file: str) -> None:
    """Write the flow in disk.

    This function is modified from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    Copyright (c) 2011, LMB, University of Freiburg.

    Args:
        flow (ndarray): The optical flow that will be saved.
        flow_file (str): The file for saving optical flow.
    """

    with open(flow_file, 'wb') as f:
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
def get_img(path,img_size):
            
    img1 = frame_utils.read_gen(path)
    img1 = np.array(img1).astype(np.uint8)[..., :3]
    img1 = cv2.resize(img1,(img_size),cv2.INTER_LINEAR)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    return img1


@torch.no_grad()
def pred_visdrone(model):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    modelname = "EMD-Flow"
    roots = ["/home/ymf/datas/visdrone/VisDrone2019-MOT-train/",
             "/home/ymf/datas/visdrone/VisDrone2019-MOT-val/",
             "/home/ymf/datas/visdrone/VisDrone2019-MOT-test-dev/"]
    
    img_size = (136*4,76*4)

    dst_h,dst_w = 76,136
    src_h,src_w = img_size[1],img_size[0]
    # dst_folder = f"flow_gma_{dst_h}x{dst_w}"

    for root in roots:
        seq_root = os.path.join(root,"sequences")
        flow_save_root = os.path.join(root,f"flow_{modelname}_{dst_h}x{dst_w}")
        flow_vis_save_root = os.path.join(root,f"flow_{modelname}_{dst_h}x{dst_w}_vis")
        os.makedirs(flow_save_root,exist_ok=True)
        os.makedirs(flow_vis_save_root,exist_ok=True)

        for seqname in tqdm(os.listdir(seq_root)):
            current_seq_folder = os.path.join(seq_root,seqname)
            current_flow_folder = os.path.join(flow_save_root,seqname)
            current_flow_vis_folder = os.path.join(flow_vis_save_root,seqname)
            os.makedirs(current_flow_folder,exist_ok=True)
            os.makedirs(current_flow_vis_folder,exist_ok=True)

            imagenames = sorted(os.listdir(current_seq_folder))

            for i in tqdm(range(len(imagenames)-1)):
                imgpath1 = os.path.join(current_seq_folder,imagenames[i])
                imgpath2 = os.path.join(current_seq_folder,imagenames[i+1])
                image1 = get_img(imgpath1,img_size)
                image2 = get_img(imgpath2,img_size)

                image1 = image1[None].cuda()
                image2 = image2[None].cuda()
                flow = model(image1, image2, test_mode=True)
                flow = flow.squeeze(0).cpu().numpy().transpose(1,2,0)
                # ipdb.set_trace()
                sw,sh = dst_w / src_w , dst_h / src_h
                flow = cv2.resize(flow,(dst_w,dst_h),cv2.INTER_LINEAR)
                flow[:,:,0] = flow[:,:,0] * sw
                flow[:,:,1] = flow[:,:,1] * sh

                # write_flow(flow,dstpath)
                write_flow(flow, os.path.join(current_flow_folder,imagenames[i].replace(".jpg",".flo")))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--iters8', type=int, default=6)
    parser.add_argument('--iters4', type=int, default=0)

    parser.add_argument('--model_type', type=str, default='S')
    args = parser.parse_args()

    assert args.iters8 > 0 and args.iters4 >= 0
    assert args.model_type == 'S' or args.model_type == 'M' or args.model_type == 'L'

    model = torch.nn.DataParallel(EMD(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()


    pred_visdrone(model.module)




