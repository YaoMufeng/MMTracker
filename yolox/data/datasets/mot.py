from fileinput import filename
import cv2
import numpy as np
from pycocotools.coco import COCO

import os

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

import pdb
is_visdrone = False
# is_visdrone = True
visdrone_basename = "VisDrone2019-MOT"
print("is visdrone:",is_visdrone)

class MOTDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None
        ,interested_seqs=None
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file

        #print(data_dir)
        if not os.path.exists(data_dir):
            data_dir = data_dir.replace(name,"")
        # json_load_path = os.path.join(self.data_dir, "annotations", self.json_file)
        #
        json_load_path = json_file
        print(json_load_path)

        self.coco = COCO(json_load_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        print("class ids:",self.class_ids)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        # img_file = os.path.join(
        #     self.data_dir, self.name, file_name
        # )

        img_file = os.path.join(self.data_dir, file_name)

        #print(self.data_dir,self.name,file_name)

        split = self.data_dir.split("/")[-1]
        _split = split

        # print(self.data_dir,self.name,file_name,img_file)

        if split == "test" and "visdrone" in img_file:
            split = "test-dev"
        
        img_file = img_file.replace("test/test","test").replace("val/val","val")
        if not os.path.exists(img_file):
            name_arr = file_name.split("/")
            name,ext = os.path.splitext(name_arr[-1])
            if len(name) == 6:
                zfill_count = 5
            elif len(name) == 5:
                zfill_count = 6
            elif len(name) ==7:
                zfill_count=7

            #print("img_file not exists:",img_file)
            new_name = str(int(name)).zfill(zfill_count) + ext
            img_file = img_file.replace(name_arr[-1],new_name)

            
            if is_visdrone or "visdrone" in img_file:
                #print("imgfile:",img_file)
                #print(img_file)
                img_file=img_file.replace("test/test","test").replace(_split,f"{visdrone_basename}-{split}/sequences").replace("sequences/test","sequences")
                if not os.path.exists(img_file):
                    img_file=img_file.replace("VisDrone2019-MOT-train","VisDrone2019-MOT-val")
            if "zzzk" in img_file:
                img_file = img_file.replace(_split,f"{split}/sequences")
            if "zzzk" in img_file:
                pass
            
            # print(img_file) 
            img_file = os.path.join(self.data_dir,self.name,f"{new_name}{ext}")
        #print("datadir:",self.data_dir)
        #print(img_file,_split,split)
        img = cv2.imread(img_file)

        # print(img_file, self.data_dir, self.name, file_name)
        # ipdb.set_trace()

        if img is None:
            print(img_file,os.path.exists(img_file))
        # print(img_file)
        # ipdb.set_trace()
        assert img is not None

        h,w,c = img.shape
        ignore_mask = np.zeros([h,w])
        return img, res.copy(),ignore_mask, img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target,ignore_mask, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        # ignore_mask = np.zeros_like(img)
        return img, target, img_info, img_id



class MOTFlowDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None
        ,interested_seqs=None
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file

        #print(data_dir)
        if not os.path.exists(data_dir):
            data_dir = data_dir.replace(name,"")
        json_load_path = os.path.join(self.data_dir, "annotations", self.json_file)
        print(json_load_path)
        self.coco = COCO(json_load_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        print("class ids:",self.class_ids)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
            #pdb.set_trace()
        num_objs = len(objs)

        res = np.zeros((num_objs, 7))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]
        
            res[ix,6] = float(obj["flow_score"])


        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )

        #print(self.data_dir,self.name,file_name)

        split = self.data_dir.split("/")[-1]
        _split = split
        if split == "test" and "visdrone" in img_file:
            split = "test-dev"
        img_file = img_file.replace("test/test","test")
        if not os.path.exists(img_file):
            #print(img_file)
            name_arr = file_name.split("/")
            name,ext = os.path.splitext(name_arr[-1])
            if len(name) == 6:
                zfill_count = 5
            elif len(name) == 5:
                zfill_count = 6
            elif len(name) ==7:
                zfill_count=7

            #print("img_file not exists:",img_file)
            new_name = str(int(name)).zfill(zfill_count) + ext
            img_file = img_file.replace(name_arr[-1],new_name)

            if is_visdrone or "visdrone" in img_file:
                #print("imgfile:",img_file)
                img_file=img_file.replace("test/test","test").replace(_split,f"{visdrone_basename}-{split}/sequences").replace("sequences/test","sequences")
                if not os.path.exists(img_file):
                    img_file=img_file.replace("VisDrone2019-MOT-train","VisDrone2019-MOT-val")
            #img_file = os.path.join(self.data_dir,self.name,f"{new_name}{ext}")
        # print("datadir:",self.data_dir)
        # print(img_file,_split,split)
        # print(img_file)
        img = cv2.imread(img_file)
        #if img is None:
        #print(img_file,os.path.exists(img_file))
        # if img is None:
        #     print(img_file)
        assert img is not None

        #pdb.set_trace()
        #print("imginfo:",img_info)

        #print("res:",res,res.shape)
        ignore_path = img_file.replace("sequences","ignore_mask").replace("img1","ignore_mask")
        ignore_mask = cv2.imread(ignore_path,cv2.IMREAD_UNCHANGED)
        #if ignore_mask.shape[2] == 3:
        #    ignore_mask = ignore_mask.mean(axis=2)
        #ignore_mask[ignore_mask > 127] = 255
        #print(ignore_mask.shape)
        #return img, {"res":res.copy(),"ignore_mask":ignore_mask}, img_info, np.array([id_])
        #img_infos = {"origin_info":img_info,"ignore_mask":ignore_mask}
        
        return img, res.copy(),ignore_mask,img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, ignore_mask,img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target,ignore_mask,img_info, img_id
    

import ipdb
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

class MOTFlowDataset2(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None
        ,interested_seqs=None
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file

        #print(data_dir)
        if not os.path.exists(data_dir):
            data_dir = data_dir.replace(name,"")
        json_load_path = os.path.join(self.data_dir, "annotations", self.json_file)
        print(json_load_path)
        self.coco = COCO(json_load_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        print("class ids:",self.class_ids)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
            #pdb.set_trace()
        num_objs = len(objs)

        res = np.zeros((num_objs, 7))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]
        
            res[ix,6] = float(obj["flow_score"])


        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]
    def pull_item(self, index):
        #保证next img 不越界
        if index >= len(self.annotations) - 1:
            index = len(self.annotations) - 2
        
        id_ = self.ids[index]
        res, img_info, file_name = self.annotations[index]
        res2,img_info2,file_name2 = self.annotations[index+1]
        
        folder1,name1 = os.path.split(file_name)
        folder2,name2 = os.path.split(file_name2)

        
        while folder1 != folder2 and index >=0:
            print(file_name,file_name2)
            index -= 1
            res, img_info, file_name = self.annotations[index]
            res2,img_info2,file_name2 = self.annotations[index+1]
            folder1,name1 = os.path.split(file_name)
            folder2,name2 = os.path.split(file_name2)

        items1,img_file = self.pull_item_single(index)
        items2,_ = self.pull_item_single(index + 1)

        #print(img_file)
        flow_file = img_file.replace("sequences/","fastflownet-512x384/").replace("img1/","fastflownet-512x384/").replace(".jpg",".flo")
        
        #print(flow_file,os.path.exists(flow_file))
        
        flow = read_flow(flow_file)
        return items1,items2,flow

    def pull_item_single(self, index):
        id_ = self.ids[index]
        res, img_info, file_name = self.annotations[index]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        split = self.data_dir.split("/")[-1]
        _split = split
        if split == "test":
            split = "test-dev"
        img_file = img_file.replace("test/test","test")
        if not os.path.exists(img_file):
            name_arr = file_name.split("/")
            name,ext = os.path.splitext(name_arr[-1])
            if len(name) == 6:
                zfill_count = 5
            elif len(name) == 5:
                zfill_count = 6
            elif len(name) ==7:
                zfill_count=7
            
            if "img" in name:
                zfill_count = 6
                new_name = "img" + str(int(name.strip("img"))).zfill(zfill_count) + ext
            else:
                new_name = str(int(name)).zfill(zfill_count) + ext
            img_file = img_file.replace(name_arr[-1],new_name)
            # print("====",img_file,"visdrone" in img_file,"----")
            if is_visdrone or "visdrone" in img_file:
                img_file=img_file.replace("test/test","test").replace(_split,f"{visdrone_basename}-{split}/sequences").replace("sequences/test","sequences")
                if not os.path.exists(img_file):
                    img_file=img_file.replace("VisDrone2019-MOT-train","VisDrone2019-MOT-val")
            if "uavdt" in img_file:
                print(img_file)
        img = cv2.imread(img_file)
        assert img is not None
        ignore_path = img_file.replace("sequences","ignore_mask").replace("img1","ignore_mask")
        ignore_mask = cv2.imread(ignore_path,cv2.IMREAD_UNCHANGED)

        return (img, res.copy(),ignore_mask,img_info, np.array([id_])),img_file

    # @Dataset.resize_getitem
    # def __getitem__(self, index):
    #     """
    #     One image / label pair for the given index is picked up and pre-processed.

    #     Args:
    #         index (int): data index

    #     Returns:
    #         img (numpy.ndarray): pre-processed image
    #         padded_labels (torch.Tensor): pre-processed label data.
    #             The shape is :math:`[max_labels, 5]`.
    #             each label consists of [class, xc, yc, w, h]:
    #                 class (float): class index.
    #                 xc, yc (float) : center of bbox whose values range from 0 to 1.
    #                 w, h (float) : size of bbox whose values range from 0 to 1.
    #         info_img : tuple of h, w, nh, nw, dx, dy.
    #             h, w (int): original shape of the image
    #             nh, nw (int): shape of the resized image without padding
    #             dx, dy (int): pad size
    #         img_id (int): same as the input index. Used for evaluation.
    #     """
    #     items1,items2,flow = self.pull_item(index)
    #     img, target, ignore_mask,img_info, img_id = items1
    #     img2,target2,ignore_mask2,img_info2,img_id2 = items2 

    #     if self.preproc is not None:
    #         img,ignore_mask,target,flow = self.preproc(img,ignore_mask,target,self.input_dim,flow=flow)
    #         img2,ignore_mask2,target2 = self.preproc(img2,ignore_mask2,target2,self.input_dim)
    #     print("mot resize get item")
    #     return (img, target,ignore_mask,img_info, img_id),(img2, target2,ignore_mask2,img_info2, img_id2)
    




    
