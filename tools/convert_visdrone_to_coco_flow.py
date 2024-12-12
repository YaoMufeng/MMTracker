from __future__ import annotations
import os
from unicodedata import category
import numpy as np
import json
import cv2
from tqdm import tqdm
import ipdb


DATA_PATH = '/home/ymf/datas/visdrone'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')


basename = "VisDrone2019-MOT"
SPLITS = ["train"]
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

zfill_count=7
#for antiuav
categories = [
    {
"id":1,"name":"pedestrain"
},
{
"id":2,"name":"people"
},
    {
"id":3,"name":"bicycle"
},
    {
"id":4,"name":"car"
},
    {
"id":5,"name":"van"
},
    {
"id":6,"name":"truck"
},
    {
"id":7,"name":"tricycle"
},
    {
"id":8,"name":"awning-tricycle"
},
    {
"id":9,"name":"bus"
},
    {
"id":10,"name":"motor"
}
]

interested_class = [1,4,5,6,9]
maps = [1,2,3,4,5]

ten_to_five_dict = {}
flow_limit = 17
min_flow_score = 0.1
express_range = 1 - min_flow_score

for i in range(len(maps)):
    ten_to_five_dict[interested_class[i]]=maps[i]

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

ranges = [2,4,8,16]
probs = np.array([0.534457,0.184115,0.133738,0.0788,0.0688])
dst_probs = 1 / len(probs)



#modified for trainval
def base(flow_name="linear"):
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    #flow_func = flow_score_func[flow_name]
    out_path = os.path.join(OUT_PATH, '{}_flow_{}.json'.format("trainval",flow_name))
    print(f"save to {out_path}")
    
    
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    out = {'images': [], 'annotations': [], 'videos': [],
               'categories': categories}
    
    for split in SPLITS:
        #if split == "test":
        #    data_path = os.path.join(DATA_PATH, 'test')
        #else:
        data_path = os.path.join(DATA_PATH,f"{basename}-{split}")

        img_folder = os.path.join(data_path,"sequences")
        seqs = os.listdir(img_folder)

        for seq in tqdm(sorted(seqs)):
            #if '.DS_Store' in seq:
            #    continue
            #if 'mot' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
            #    continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(img_folder,seq)
            img_path = seq_path
            #img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(data_path,"annotations",f"{seq}.txt")

            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])  # half and half

            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                              [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]


            anno_dict = {}

            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            print('{} ann images'.format(int(anns[:, 0].max())))
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                    continue
                track_id = int(anns[i][1])
                cat_id = int(anns[i][7])#很重要！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                ann_cnt += 1

                category_id = cat_id
                #print(cat_id)
                if category_id <=0 or category_id > 10:
                    continue

                bbox = anns[i][2:6].tolist()
                conf = float(anns[i][6])
                area = float(anns[i][4] * anns[i][5])
                if frame_id not in anno_dict.keys():
                    anno_dict[frame_id] = [(frame_id,track_id,cat_id,ann_cnt,bbox,conf,area)]
                else:
                    anno_dict[frame_id].append((frame_id,track_id,cat_id,ann_cnt,bbox,conf,area))
                # ann = {'id': ann_cnt,
                #            'category_id': category_id,
                #            'image_id': image_cnt + frame_id,
                #            'track_id': tid_curr,
                #            'bbox': anns[i][2:6].tolist(),
                #            'conf': float(anns[i][6]),
                #            'iscrowd': 0,
                #            'area': float(anns[i][4] * anns[i][5])}
                # out['annotations'].append(ann)

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue

                #_path = os.path.join(data_path, f'{seq}/img1/{str(i+1).zfill(zfill_count)}.jpg') # for dvb
                img_name = f'{seq}/{str(i+1).zfill(zfill_count)}.jpg'

                _path = os.path.join(img_folder,img_name)

                #flow = cv2.imread(_path)
                #ipdb.set_trace()
                #print(_path)
                

                flow_path = _path.replace("sequences","offset_flow_emd_gma_76x136").replace(".jpg",".flo")
                
                img = cv2.imread(_path)

                img_h,img_w,img_c = img.shape

                if os.path.exists(flow_path):
                    flow = read_flow(flow_path)
                    flow = cv2.resize(flow,(img_w,img_h),cv2.INTER_LINEAR)
                else:
                    print(f"not exists: {flow_path}")
                    flow = np.zeros([img_h,img_w,2])
                

                name = f"{basename}-{split}/sequences/{img_name}"
                print(name)
                image_info = {'file_name': name,  # image name.
                            'id': image_cnt + i + 1,  # image number in the entire training set.
                            'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                            'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                            'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                            'video_id': video_cnt,
                            'height': img_h, 'width': img_w}
                out['images'].append(image_info)


                if i + 1 not in anno_dict.keys():
                    continue
                anno_list = anno_dict[i + 1]

                for anno in anno_list:
                    frame_id,track_id,cat_id,ann_cnt,bbox,conf,area = anno
                    x,y,w,h = bbox

                    cx,cy = int(x + 0.5*w),int(y + 0.5*h)
                    flow_box = flow[cy,cx,:]
                    flow_val = np.sqrt(flow_box[...,0]**2 + flow_box[...,1]**2)
                    flow_mean = flow_val.mean()

                    flow_score = flow_mean
                    flow_score_str = f"{flow_score:.3f}"

                    ann = {'id': ann_cnt,
                               'category_id': cat_id,
                               'image_id': image_cnt + frame_id,
                               'track_id': track_id,
                               'bbox': bbox,
                               'conf': conf,
                               'iscrowd': 0,
                               'area': area,
                               'flow_score':flow_score_str}
                    out['annotations'].append(ann)
                #print(image_info)
            print('{}: {} images'.format(seq, num_images))
            
                

                
            image_cnt += num_images
            #print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))

        #out_str = json.dumps(out,indent=2)
    json.dump(out, open(out_path, 'w'),indent=2)





if __name__ == '__main__':
    #modified for trainval
    flow_name = "boxover_emd_gma"
    base(flow_name)
    
