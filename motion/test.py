from tkinter.filedialog import SaveFileDialog
from turtle import width
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
# from yolox.evaluators import MCMOTEvaluator,MOTEvaluator,MCMOTEvaluatorBias,MCMOTEvaluatorBiasWarp,MCMOTEvaluatorBiasPublic,MCMOTEvaluatorBiasGMM
from motion.motion_evaluator_debug import MotionEvaluator
import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

import pdb
# from flow_io import read_flow

# from mmflow.apis import inference_model, init_model

import sys
import torch

sys.path.append("/home/ymf/codes/flow_estimation")
sys.path.append("/home/ymf/codes/ByteTrack/tools/")


from motion_model import MotionMamba

interested_class = [1,4,5,6,9]
def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--save_folder", type=str, default="default_save_folder", help="save_folder")
    parser.add_argument("--test_root", type=str, default=None, help="test folder root")
    parser.add_argument("--test_visdrone", type=bool, default=False, help="test folder root")
    parser.add_argument("--test_uavdt", type=bool, default=False, help="test folder root")
    parser.add_argument("--val_visdrone", type=bool, default=False, help="test folder root")
    parser.add_argument("--loss_type", type=str, default="baseline", help="test folder root")
    parser.add_argument("--output_dir", type=str, default="../Bytetrack_outputs", help="outputdir")
    parser.add_argument("--save_json", type=bool, default=False, help="outputdir")
    parser.add_argument("--conv_bias",type=bool,default=False)
    parser.add_argument("--no_bias",type=bool,default=False)
    parser.add_argument("--use_warp",type=bool,default=False)
    parser.add_argument("--filtering",type=bool,default=False)
    return parser

import pdb
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

@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank

    file_name = os.path.join(args.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    save_folder = args.save_folder

    results_folder = os.path.join(file_name,save_folder)
    os.makedirs(results_folder, exist_ok=True)


    args.class_numbers = exp.class_numbers

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    model = exp.get_model(args)

    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    
    torch.set_grad_enabled(False)

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    _evaluator = MotionEvaluator

    logger.info(f"===============================evaluator:{_evaluator}")


    # snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/baseline_emd_motion_light_prelu/best_0.3377073999590642.pth"
    # snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/flowldam_emd_motion_light_multiscale_softshrink/best_0.3399036384133801.pth"
    
    snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/MotionMamba_uavdt_boxover_offsetldam_v3_emd_softshrink_multiscale_mse_l1/best_mota_0.4899_idf1_0.6559.pth"
    # snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/MotionMamba_boxover_1_trainval_flowldam_tv_emd_gma_motion_softshrink_multiscale_mse_l1/best_mota_0.4772_idf1_0.5634.pth"
    # snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/MotionMamba_boxover_1_trainval_baseline_tv_emd_gma_motion_softshrink_multiscale_mse_l1/best_mota_0.4352_idf1_0.5222.pth"

    # snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/MotionMamba_uavdt_boxover_offsetldam_tv_emd_softshrink_multiscale_mse_l1/best_mota_0.4921_idf1_0.6499.pth"

    # snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/MotionMamba_boxover_trainval_flowldam_tv_emd_gma_motion_softshrink_multiscale_mse_l1/best_mota_0.4776_idf1_0.5637.pth"
    # snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/MotionMamba_uavdt_ldam_emd_softshrink/best_0.482389324935357.pth"
    # snap_to_load = "/home/ymf/codes/ByteTrack/motion/weights/mamba_trainval_flowldam_tv_emd_gma_motion_softshrink_multiscale_mse_l1/best_mota_0.4738_idf1_0.5648.pth"

    # motion_model = MotionModel()
    # motion_model = MotionModelLightSingle().cuda()
    # motion_model = MotionModelLight().cuda()
    # motion_model = MotionModelSingle().cuda()
    # motion_model = MotionModelLight().cuda()
    # motion_model = MotionModelLightSoftshrink().cuda()
    # motion_model = MotionModelSoftshrinkMultiScale().cuda()
    # motion_model = MambaSoftshrinkMultiScale().cuda()
    motion_model = MotionMamba().cuda()
    # motion_model = MotionModelLightPRELU().cuda()
    
    motion_model.load_state_dict(torch.load(snap_to_load))
    if args.fuse:
        logger.info("\tFusing model...")
        motion_model = fuse_model(motion_model)
    if args.fp16:
        motion_model.half()
    motion_model.eval().cuda()
    

    evaluator = _evaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        flow_model=motion_model)

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None



    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
    logger.info("\n" + f"{summary}")

    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    if exp.val_ann == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    print('gt_type', gt_type)
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
    if args.val_visdrone:
        gtfiles = glob.glob(os.path.join(args.test_root,"VisDrone2019-MOT-val/annotations","*.txt"))
        
    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    if args.test_visdrone:
        gt = OrderedDict([(Path(f).parts[-1].replace(".txt",""), mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
    
    mh = mm.metrics.create()


    # for k,v in gt.items():
    #      v2 = v[(v.ClassId==1)|(v.ClassId==4)|(v.ClassId==5)|(v.ClassId==6)|(v.ClassId==9)]
    #      v2[v.ClassId==4].ClassId = 2
    #      v2[v.ClassId==5].ClassId = 3
    #      v2[v.ClassId==6].ClassId = 4
    #      v2[v.ClassId==9].ClassId = 5
    #      gt[k] = v2

    # for k,v in ts.items():
    #     v2 = v[(v.ClassId==1)|(v.ClassId==4)|(v.ClassId==5)|(v.ClassId==6)|(v.ClassId==9)]
    #     ts[k] = v2
    #pdb.set_trace()
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters, 
    #   namemap=mm.io.motchallenge_metric_names))
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


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args,args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
