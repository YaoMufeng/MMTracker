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
from yolox.tracker.mc_bytetracker_bias_nokalman import MCByteTrackerBias
from yolox.tracker.mc_bytetracker_bias_nokalman_conv import MCByteTrackerBiasConv
from collections import OrderedDict

import motmetrics as mm
import contextlib
import io
import os
import itertools
import json
import time

import glob
from pathlib import Path

from torchvision.transforms import Resize





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
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{ids},-1,-1\n'
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

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

class MCMOTEvaluatorBiasWarp:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes,flow_model=None,testdev=False,interested_seqs=None):
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
        self.flow_model = flow_model

        self.resize = Resize([384,512])
        self.inverse_resize = Resize([self.img_size[0],self.img_size[1]])

        file_name = os.path.join(self.args.output_dir, self.args.experiment_name)
        os.makedirs(file_name, exist_ok=True)
        save_folder = self.args.save_folder
        results_folder = os.path.join(file_name,save_folder)
        os.makedirs(results_folder, exist_ok=True)
        self.results_folder = results_folder

        self.save_json = args.save_json

        logger.info(f"results folder: {self.results_folder}")

        self.means = torch.Tensor([0.574859,0.5284268,0.48245227]).reshape([1,3,1,1])
        self.stds = torch.Tensor([0.20307621,0.16383478,0.16327296]).reshape([1,3,1,1])

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
        logger.info("mcmot bias evaluate:")
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


        current_tracker = None
        if self.args.conv_bias:
            current_tracker = MCByteTrackerBiasConv
        else:
            current_tracker = MCByteTrackerBias

        if self.args.no_bias:
            current_tracker = MCByteTracker

        tracker = current_tracker(self.args)
        logger.info(f"current tracker:{current_tracker}")
        ori_thresh = self.args.track_thresh
        
        root = self.args.test_root
        if self.args.test_visdrone:
            root = os.path.join(root,"VisDrone2019-MOT-test-dev")
        if self.args.val_visdrone:
            root = os.path.join(root,"VisDrone2019-MOT-val")
        logger.info(root)
        prev_imgs = None
        flow,flow_fpn = None,None
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            img_file_name = info_imgs[4]
            video_name = img_file_name[0].split('/')[0]

            #print(self.args.test_root,info_imgs)
            flow_path = os.path.join(root,"flow",img_file_name[0]).replace(".jpg",".flo")
            if self.args.test_uavdt:
                flow_path = os.path.join(root,img_file_name[0]).replace(".jpg",".flo").replace("img1","flow")
            #flow = read_flow(flow_path)
            #print(flow)

            if self.interested_seqs is not None and video_name not in self.interested_seqs:
                continue
            w_ratio , h_ratio = 1,1
            imgs = imgs.type(tensor_type)

            # ipdb.set_trace()
            if self.flow_model is not None and prev_imgs is not None:
                
                #print("prev device: ",prev_imgs.device)
                self.means = self.means.to(prev_imgs.device)
                self.stds = self.stds.to(prev_imgs.device)
                raw_prev_imgs = (prev_imgs*self.stds + self.means)
                raw_imgs = (imgs*self.stds + self.means)
                
                raw_prev_imgs,raw_imgs = self.resize(raw_prev_imgs),self.resize(raw_imgs)

                if isinstance(self.flow_model,FastFlowNet):
                    #print("fast flownet!")
                    #raw_prev_imgs,raw_imgs = raw_prev_imgs/127 - 1 , raw_imgs/127 - 1
                    raw_prev_imgs,raw_imgs, _ = centralize(raw_prev_imgs,raw_imgs)
                    flow = self.flow_model(torch.cat([raw_prev_imgs,raw_imgs],dim=1).half())
                    flow = flow*20
                    flow = F.interpolate(flow,scale_factor=4).float()
                    flow_batch,flow_c,flow_h,flow_w = flow.shape
                else:
                    raw_prev_imgs,raw_imgs = 255*raw_prev_imgs,255*raw_imgs
                    raw_prev_imgs,raw_imgs = raw_prev_imgs/127 - 1 , raw_imgs/127 - 1
                    flow = self.flow_model(torch.cat([raw_prev_imgs,raw_imgs],dim=1))[0]['flow']
                    flow = torch.from_numpy(flow.transpose(2,0,1)).unsqueeze(0).cuda()
                    flow_batch,flow_c,flow_h,flow_w = flow.shape
                    #flow_h,flow_w,_ = flow.shape
                    #print(flow.shape)
                
                fpn_w,fpn_h = int(self.img_size[1] / 8) , int(self.img_size[0] / 8)

                sw,sh = fpn_w / flow_w , fpn_h / flow_h

                flow_fpn = F.interpolate(flow,(fpn_h,fpn_w),mode='bilinear')
                flow_fpn[:,0:,:,:] *= sw
                flow_fpn[:,1:,:,:] *= sh
                #print(flow.shape,flow_fpn.shape)
                #print(type(flow))
                
                #print(flow.shape)
                
                # prev_img_np = raw_prev_imgs.squeeze(0).cpu().numpy().transpose(1,2,0)
                # img_np = raw_imgs.squeeze(0).cpu().numpy().transpose(1,2,0)
                # flow = inference_model(self.flow_model,prev_img_np,img_np)
                # flow_h,flow_w,flow_c = flow.shape


                #print(flow,type(flow))
                #flow = flow[0]['flow']
                #print(flow.shape)
                


                img_h, img_w = int(info_imgs[0]), int(info_imgs[1])
                w_ratio , h_ratio = img_w / flow_w , img_h / flow_h

                #print(flow.shape,(self.img_size[1],self.img_size[0]))

                #flow = cv2.resize(flow,(self.img_size[1],self.img_size[0]),interpolation=cv2.INTER_LINEAR)
                #print(type(flow))
                #flow = self.inverse_resize(flow)
            ## fpn相比输入图像，下采样倍数分别是 8,16,32,提取的光流是 512x384, 所以输入给fpn的光流是重采样为 608/8 x 1088/8
            #print(self.img_size,info_imgs)
            
            
            
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
                    tracker = current_tracker(self.args)
                    if len(results) != 0:
                        print("result folder:",result_folder)
                        print(video_names[video_id - 1])
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                


                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                # ipdb.set_trace()
                    
                # feat = model.backbone(imgs)
                # outputs = model.head(feat)
                outputs = model(imgs,optical_flow=flow_fpn)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            if self.args.save_json:
                output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
                data_list.extend(output_results)

            # run tracking
            #if self.flow_model is None:
            #    flow = read_flow(flow_path)
            
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size,flow,flow_ratios=(w_ratio,h_ratio))
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
                        online_class_ids.append(class_id+1)

                results.append((frame_id, online_tlwhs, online_ids, online_scores,online_class_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

            prev_imgs = imgs
            #gc.collect()


        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        if self.args.save_json:
            _,_,eval_results = self.evaluate_prediction(data_list, statistics)
            logger.info(eval_results)
        synchronize()

        self.show_time(statistics)

        mm.lap.default_solver = 'lap'

        args = self.args
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
            
        #print('gt_files', gtfiles)
        tsfiles = [f for f in glob.glob(os.path.join(result_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

        logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
        logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
        logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
        logger.info('Loading files.')

        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        if args.test_visdrone:
            gt = OrderedDict([(Path(f).parts[-1].replace(".txt",""), mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        if args.test_uavdt:
            gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
        #print(gt)
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
        mh = mm.metrics.create()
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
        mota = summary['mota']['OVERALL']
        idf1 = summary['idf1']['OVERALL']

        return mota,idf1
    def show_time(self,statistics):
        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )
        print(time_info)
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

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            #_, tmp = tempfile.mkstemp()

            tmp = os.path.join(self.results_folder,"tmp.json")
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            os.remove(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
