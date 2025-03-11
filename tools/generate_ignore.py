import os
import cv2
import numpy as np

from tqdm import tqdm
root = "/xxx/visdrone/VisDrone2019-MOT-train"


seq_root = os.path.join(root,"sequences")
ignore_root = os.path.join(root,"ignore_mask")
gt_root = os.path.join(root,"annotations")
os.makedirs(ignore_root,exist_ok=True)

for seqname in tqdm(os.listdir(seq_root)):
    seq_folder = os.path.join(seq_root,seqname)
    gt_path = os.path.join(gt_root,f"{seqname}.txt")
    ignore_folder = os.path.join(ignore_root,seqname)
    os.makedirs(ignore_folder,exist_ok=True)

    frame_dict = {}
    with open(gt_path,"r") as f:
        for line in f.readlines():
            arr = line.strip("\n").split(",")

            frame_id,obj_id,x,y,w,h,conf,class_id = int(arr[0]),int(arr[1]),int(arr[2]),int(arr[3]),int(arr[4]),int(arr[5]),float(arr[6]),int(arr[7])

            if class_id != 0:
                continue#0为ignore区域
            
            if frame_id not in frame_dict.keys():
                frame_dict[frame_id] = [(x,y,w,h)]
            else:
                frame_dict[frame_id].append((x,y,w,h))
    
    for imgname in tqdm(os.listdir(seq_folder)):
        name,ext = os.path.splitext(imgname)
        frame_id = int(name)

        img_path = os.path.join(seq_folder,imgname)
        msk_path = os.path.join(ignore_folder,imgname)
        img = cv2.imread(img_path)
        h,w,c = img.shape
        msk = np.zeros([h,w]).astype(np.uint8)

        if frame_id not in frame_dict.keys():
            pass
        else:
            box_list = frame_dict[frame_id]
            for box in box_list:
                x,y,w,h = box
                points = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],dtype=np.int32)
                #print(points.dtype)
                cv2.fillPoly(msk,[points],color=[255])

        cv2.imwrite(msk_path,msk)
