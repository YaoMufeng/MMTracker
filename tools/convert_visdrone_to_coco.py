from __future__ import annotations
import os
from unicodedata import category
import numpy as np
import json
import cv2
from tqdm import tqdm

# Use the same script for MOT16
#DATA_PATH = '/hdd/yaomf/datas/drone_vs_bird/raw_data'
DATA_PATH = '/hdd/yaomf/datas/visdrone'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
#SPLITS = ['train', 'test']  # --> split training data to train_half and val_half.

basename = "VisDrone2019-MOT"
SPLITS = ["train",'val','test-dev'] 
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

for i in range(len(maps)):
    ten_to_five_dict[interested_class[i]]=maps[i]


flow_limit = 17
min_flow_score = 0.1
express_range = 1 - min_flow_score

def base():
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        #if split == "test":
        #    data_path = os.path.join(DATA_PATH, 'test')
        #else:
        data_path = os.path.join(DATA_PATH,f"{basename}-{split}")

        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': categories}

        img_folder = os.path.join(data_path,"sequences")
        seqs = os.listdir(img_folder)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
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

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue

                #_path = os.path.join(data_path, f'{seq}/img1/{str(i+1).zfill(zfill_count)}.jpg') # for dvb
                img_name = f'{seq}/{str(i+1).zfill(zfill_count)}.jpg'
                _path = os.path.join(img_folder,img_name)
                img = cv2.imread(_path)
                #print(_path)
                height, width = img.shape[:2]
                image_info = {'file_name': img_name,  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
                #print(image_info)
            print('{}: {} images'.format(seq, num_images))
            
                
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

            if CREATE_SPLITTED_ANN and ('half' in split):
                anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if int(anns[i][0]) - 1 >= image_range[0] and
                                         int(anns[i][0]) - 1 <= image_range[1]], np.float32) 
                anns_out[:, 0] -= image_range[0]
                gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split))
                fout = open(gt_out, 'w')
                for o in anns_out:
                    fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                    int(o[6]), int(o[7]), o[8]))
                fout.close()
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
                iscrowd = 0
                if category_id <=0 or category_id > 10:
                    iscrowd = 1
                ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': tid_curr,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6]),
                           'iscrowd': iscrowd,
                           'area': float(anns[i][4] * anns[i][5])}
                out['annotations'].append(ann)
                
            image_cnt += num_images
            #print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))

def trainval():
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    out_path = os.path.join(OUT_PATH, 'trainval.json')

    splits = ["train","val"]

    out = {'images': [], 'annotations': [], 'videos': [],
               'categories': categories}
    
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0
    tid_last = -1
    
    for split in splits:
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

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue

                #_path = os.path.join(data_path, f'{seq}/img1/{str(i+1).zfill(zfill_count)}.jpg') # for dvb
                img_name = f'{seq}/{str(i+1).zfill(zfill_count)}.jpg'
                _path = os.path.join(img_folder,img_name)
                img = cv2.imread(_path)
                #print(_path)
                height, width = img.shape[:2]
                image_info = {'file_name': img_name,  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
                #print(image_info)
            print('{}: {} images'.format(seq, num_images))
            
                
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

            if CREATE_SPLITTED_ANN and ('half' in split):
                anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if int(anns[i][0]) - 1 >= image_range[0] and
                                         int(anns[i][0]) - 1 <= image_range[1]], np.float32) 
                anns_out[:, 0] -= image_range[0]
                gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split))
                fout = open(gt_out, 'w')
                for o in anns_out:
                    fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                    int(o[6]), int(o[7]), o[8]))
                fout.close()
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
                ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': tid_curr,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6]),
                           'iscrowd': 0,
                           'area': float(anns[i][4] * anns[i][5])}
                out['annotations'].append(ann)
                
            image_cnt += num_images
            #print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))


def trainval5():
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    out_path = os.path.join(OUT_PATH, 'trainval5.json')

    splits = ["train","val"]

    #out_path = os.path.join(OUT_PATH, 'test-dev5.json')

    #splits = ["test-dev"]

    out = {'images': [], 'annotations': [], 'videos': [],
               'categories': categories}
    
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0
    tid_last = -1
    
    for split in splits:
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

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue

                #_path = os.path.join(data_path, f'{seq}/img1/{str(i+1).zfill(zfill_count)}.jpg') # for dvb
                img_name = f'{seq}/{str(i+1).zfill(zfill_count)}.jpg'
                _path = os.path.join(img_folder,img_name)
                img = cv2.imread(_path)
                #print(_path)
                height, width = img.shape[:2]
                image_info = {'file_name': img_name,  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
                #print(image_info)
            print('{}: {} images'.format(seq, num_images))
            
                
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

            if CREATE_SPLITTED_ANN and ('half' in split):
                anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if int(anns[i][0]) - 1 >= image_range[0] and
                                         int(anns[i][0]) - 1 <= image_range[1]], np.float32) 
                anns_out[:, 0] -= image_range[0]
                gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split))
                fout = open(gt_out, 'w')
                for o in anns_out:
                    fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                    int(o[6]), int(o[7]), o[8]))
                fout.close()
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
                # if category_id <=0 or category_id > 10:
                #     continue

                if category_id not in interested_class:
                    continue

                new_id = ten_to_five_dict[category_id]
                ann = {'id': ann_cnt,
                           'category_id': new_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': tid_curr,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6]),
                           'iscrowd': 0,
                           'area': float(anns[i][4] * anns[i][5])}
                out['annotations'].append(ann)
                
            image_cnt += num_images
            #print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))


if __name__ == '__main__':
    trainval5()
    
