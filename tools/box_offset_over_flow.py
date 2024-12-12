import os
import cv2
import numpy as np
import ipdb

from tqdm import tqdm
zfill_count = 7

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

def box_over_flow(txt_paths,img_root,flow_root,save_root):
    

    for txt_path in txt_paths:
        lines = None
        with open(txt_path,"r") as f:
            lines = f.readlines()
        
        obj_dict = {}
        for line in lines:
            arr = line.strip("\n").split(",")

            frame_id,obj_id,x,y,w,h=int(arr[0]),int(arr[1]),int(arr[2]),int(arr[3]),int(arr[4]),int(arr[5])

            if obj_id not in obj_dict.keys():
                obj_dict[obj_id] = {frame_id:(x,y,w,h)}
            else:
                obj_dict[obj_id][frame_id] = (x,y,w,h)
        
        _,filename = os.path.split(txt_path)
        name,ext = os.path.splitext(filename)
        save_seq_root = os.path.join(save_root,name)
        os.makedirs(save_seq_root,exist_ok=True)


        offset_frame_dict = {}
        for obj_id,frame_dict in obj_dict.items():
            for frame_id,bbox in frame_dict.items():
                if frame_id + 1 not in frame_dict.keys():
                    continue

                bbox2 = frame_dict[frame_id+1]
                x1,y1,w1,h1 = bbox
                x2,y2,w2,h2 = bbox2

                cx1,cy1,cx2,cy2 = (x1+0.5*w1),(y1+0.5*h1),(x2+0.5*w2),(y2+0.5*h2)

                # d = np.sqrt((cx1-cx2)**2+(cy1-cy2)**2)

                dx = cx2 - cx1
                dy = cy2 - cy1

                if frame_id not in offset_frame_dict.keys():
                    offset_frame_dict[frame_id] = {obj_id:(x1,y1,w1,h1,dx,dy)}
                else:
                    offset_frame_dict[frame_id][obj_id] = (x1,y1,w1,h1,dx,dy)


        for frame_id,objs in tqdm(offset_frame_dict.items()):
            imgname = f"{str(frame_id).zfill(zfill_count)}.jpg"
            flowname = f"{str(frame_id).zfill(zfill_count)}.flo"
            imgpath = os.path.join(img_root,name,imgname)
            flowpath = os.path.join(flow_root,name,flowname)
            savepath = os.path.join(save_root,name,flowname)

            img = cv2.imread(imgpath)
            flow = read_flow(flowpath)

            h,w,c = img.shape
            flow_h,flow_w,flow_c = flow.shape

            sh,sw = h / flow_h , w / flow_w 
            #计算需要缩小的倍数
            for obj_id,bboxes in objs.items():
                x1,y1,w1,h1,dx,dy = bboxes
                fx1,fy1,fx2,fy2,fdx,fdy = int(x1 / sw) , int(y1 / sh) , int((x1+w2)/ sw) , int((y1+h2) / sh) , dx / sw , dy / sh
                fw2,fh2 = fx2-fx1,fy2-fy1
                flow_box = flow[fy1:fy2,fx1:fx2,:]

                # flow_box[...,0] = flow_box[...,0]*0.5 + fdx*0.5
                # flow_box[...,1] = flow_box[...,1]*0.5 + fdy*0.5

                flow_box[...,0] = fdx*0.5
                flow_box[...,1] = fdy*0.5

                flow[fy1:fy2,fx1:fx2,:] = flow_box
                # rand = np.random.normal(size=(fh2,fw2))
                # ipdb.set_trace()

            write_flow(flow,savepath)




def visdrone():
    splits = ["train","val","test-dev"]
    # splits = ["val"]
    for split in splits:
        txt_root = f"/home/ymf/datas/visdrone/VisDrone2019-MOT-{split}/annotations/"
        txt_paths = [os.path.join(txt_root,txtname) for txtname in os.listdir(txt_root)]
        img_root = f"/home/ymf/datas/visdrone/VisDrone2019-MOT-{split}/sequences/"
        flow_root = f"/home/ymf/datas/visdrone/VisDrone2019-MOT-{split}/flow_emd_gma_76x136/"
        save_root = f"/home/ymf/datas/visdrone/VisDrone2019-MOT-{split}/offset_1_flow_emd_gma_76x136/"
        os.makedirs(save_root,exist_ok=True)

        box_over_flow(txt_paths,img_root,flow_root,save_root)

def uavdt():

    splits = ["train"]
    # splits = ["val"]
    for split in splits:
        txt_root = f"/home/ymf/datas/uavdt/visdrone/VisDrone2019-MOT-{split}/annotations/"
        txt_paths = [os.path.join(txt_root,txtname) for txtname in os.listdir(txt_root)]
        img_root = f"/home/ymf/datas/uavdt/visdrone/VisDrone2019-MOT-{split}/sequences/"
        flow_root = f"/home/ymf/datas/uavdt/visdrone/VisDrone2019-MOT-{split}/flow_EMD-Flow_76x136/"
        save_root = f"/home/ymf/datas/uavdt/visdrone/VisDrone2019-MOT-{split}/offset_flow_EMD-Flow_76x136/"
        os.makedirs(save_root,exist_ok=True)

        box_over_flow(txt_paths,img_root,flow_root,save_root)


def main():
    visdrone()
    # uavdt()


if __name__ == "__main__":
    main()