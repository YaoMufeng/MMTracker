from collections import defaultdict

from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

from yolox.tracker.mc_bytetracker import MCByteTracker
from collections import OrderedDict

import motmetrics as mm
import contextlib
import io
import os
import itertools
import json
import tempfile
import time
#import ipdb
import pdb
import glob
from pathlib import Path
import numpy as np
import gc
interested_class = [1,4,5,6,9]

import ipdb

def clean_gt_and_ts(gt, ts):
    valid_cls = np.array([1,4,5,6,9]).reshape(1,-1)  # pedestrian,car,van,truck,bus
    columns = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused']

    def filter_cls(df):
        valid = (df['ClassId'].values.reshape(-1,1) == valid_cls).any(axis=1)
        valid &= df['Confidence'] > 0.
        return df[valid]

    def filter_ignore(df, ignore_regions):
        ignore = df['ClassId'] == -1
        frame_ids = np.array([v[0] for v in df.index.values])
        for frame_id, regions in ignore_regions.items():
            x1, y1, x2, y2 = np.split(np.array(regions).reshape(-1, 4).T, 4, axis=0)
            f_ind = frame_ids == frame_id
            xc, yc = (df['X'][f_ind] + df['Width'][f_ind] / 2.).values.reshape(-1,1), \
                     (df['Y'][f_ind] + df['Height'][f_ind] / 2.).values.reshape(-1,1)
            ignore[f_ind] = ((xc > x1) & (xc < x2) & (yc > y1) & (yc < y2)).any(axis=1)
        return df[~ignore]

    _gt, _ts = OrderedDict(), OrderedDict()
    for k, label in tqdm(gt.items(), desc='filtering bboxes'):
        ignore_indices = (label['ClassId'] == 0) | (label['ClassId'] == 11)
        ig_x1, ig_y1, ig_w, ig_h = label['X'][ignore_indices].values, \
                                   label['Y'][ignore_indices].values, \
                                   label['Width'][ignore_indices].values, \
                                   label['Height'][ignore_indices].values
        ig_x2, ig_y2 = ig_x1 + ig_w, ig_y1 + ig_h
        ignore_regions = {}
        frame_ids = np.array([v[0] for v in label.index.values])
        for i, frame_id in enumerate(frame_ids[ignore_indices]):
            if frame_id not in ignore_regions:
                ignore_regions[frame_id] = []
            ignore_regions[frame_id].append([ig_x1[i], ig_y1[i], ig_x2[i], ig_y2[i]])
        
        label = filter_cls(label)
        label = filter_ignore(label, ignore_regions)
        pred = filter_cls(ts[k])
        pred = filter_ignore(pred, ignore_regions)
        
        _gt[k] = label
        _ts[k] = pred

    return _gt, _ts


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        #pdb.set_trace()
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names
def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s:.4f},{ids},-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores,class_ids in results:
            #print(results)
            for tlwh, track_id, score,class_id in zip(tlwhs, track_ids, scores,class_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2),ids=class_id)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MCMOTEval:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes,testdev=False,interested_seqs=None):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args
        self.args.num_classes = num_classes
        self.interested_seqs = interested_seqs

        file_name = os.path.join(self.args.output_dir, self.args.experiment_name)
        os.makedirs(file_name, exist_ok=True)
        save_folder = self.args.save_folder
        results_folder = os.path.join(file_name,save_folder)
        os.makedirs(results_folder, exist_ok=True)
        self.results_folder = results_folder

        logger.info(f"results folder: {self.results_folder}")

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        multi_class = False,
        _continue=False
    ):
        logger.info("mcmot evaluate:")
        if result_folder is None:
            result_folder = self.results_folder
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = MCByteTracker(self.args)
        ori_thresh = self.args.track_thresh
        args = self.args
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            img_file_name = info_imgs[4]
            arr = img_file_name[0].split('/')
            video_name = arr[0]

            if "VisDrone2019-MOT" in video_name:
                video_name = arr[-2]

            # print(video_name)
            # ipdb.set_trace()
            if self.interested_seqs is not None and video_name not in self.interested_seqs:
                continue

            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()

                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = MCByteTracker(self.args)
                    if len(results) != 0:
                        print("result folder:",result_folder)
                        print(video_names[video_id - 1])
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                
                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_class_ids = []

                for class_id,t_list in online_targets.items():
                    for t in t_list:
                        tlwh = t.tlwh
                        tid = t.track_id
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_class_ids.append(int(class_id+1))

                results.append((frame_id, online_tlwhs, online_ids, online_scores,online_class_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)


        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        mm.lap.default_solver = 'lap'


        gt_type = ''
        #print('gt_type', gt_type)
        if args.mot20:
            gtfiles = glob.glob(os.path.join('datasets/MOT20/train', '*/gt/gt{}.txt'.format(gt_type)))
        else:
            gtfiles = glob.glob(os.path.join('datasets/mot/train', '*/gt/gt{}.txt'.format(gt_type)))
        
        if args.test_root is not None:
            gtfiles = glob.glob(os.path.join(args.test_root, '*/gt/gt{}.txt'.format(gt_type)))
        if args.test_visdrone:
            gtfiles = glob.glob(os.path.join(args.test_root,"VisDrone2019-MOT-test-dev/annotations","*.txt"))
        if args.test_uavdt:
            gtfiles = glob.glob(os.path.join(args.test_root,"*/gt/gt_whole.txt"))
        
        if "zzzk" in args.test_root:
            # print("entered!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            gtfiles = glob.glob(os.path.join(args.test_root,"test/annotations","*.txt"))
            
        #print('gt_files', gtfiles)
        tsfiles = [f for f in glob.glob(os.path.join(result_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

        # ipdb.set_trace()
        logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
        logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
        logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
        logger.info('Loading files.')
        # import ipdb
        # ipdb.set_trace()
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        if args.test_visdrone:
            gt = OrderedDict([(Path(f).parts[-1].replace(".txt",""), mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        if args.test_uavdt:
            gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        if "zzzk" in args.test_root:
            gt = OrderedDict([(Path(f).parts[-1].replace(".txt",""), mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        
        #print(gt)
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
        mh = mm.metrics.create()

        gt, ts = clean_gt_and_ts(gt, ts)

          
        # accs, names = compare_dataframes(gt, ts)

        # logger.info('Running metrics')
        # metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
        #             'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
        #             'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
        # summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        # # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
        # # print(mm.io.render_summary(
        # #   summary, formatters=mh.formatters, 
        # #   namemap=mm.io.motchallenge_metric_names))
        # div_dict = {
        #     'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        #     'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
        # for divisor in div_dict:
        #     for divided in div_dict[divisor]:
        #         summary[divided] = (summary[divided] / summary[divisor])
        # fmt = mh.formatters
        # change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
        #                     'partially_tracked', 'mostly_lost']
        # for k in change_fmt_list:
        #     fmt[k] = fmt['mota']
        # print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

        # metrics = mm.metrics.motchallenge_metrics + ['num_objects']
        # summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        # print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
        # logger.info('Completed')


        #====================

        accs, names = compare_dataframes(gt, ts)
        
        logger.info('Running metrics')
        metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        div_dict = {
            'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
            'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
        for divisor in div_dict:
            for divided in div_dict[divisor]:
                summary[divided] = (summary[divided] / summary[divisor])
        fmt = mh.formatters
        change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost']
        for k in change_fmt_list:
            fmt[k] = fmt['mota']
        print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

        metrics = mm.metrics.motchallenge_metrics + ['num_objects']
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
        logger.info('Completed')

        #====================

        mota = summary['mota']['OVERALL']
        idf1 = summary['idf1']['OVERALL']

        return mota,idf1

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list
