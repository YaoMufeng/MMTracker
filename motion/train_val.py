
from loguru import logger
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, setup_logger
# from yolox.evaluators import MCMOTEvaluator,MOTEvaluator,MCMOTEvaluatorBias,MCMOTEvaluatorBiasWarp,MCMOTEvaluatorBiasPublic,MCMOTEvaluatorBiasGMM
from motion.motion_evaluator_debug import MotionEvaluator
import argparse
import os
import random
import warnings
import glob
import motmetrics as mm

import sys
import torch
from torch import nn
sys.path.append("/home/ymf/codes/flow_estimation")
sys.path.append("/home/ymf/codes/ByteTrack/tools/")

from motion_model import MotionMamba
from motion_dataset import MotionDataset
import torch.utils.data as data
from tqdm import tqdm
import ipdb
import torch.nn.functional as F

interested_class = [1,4,5,6,9]
def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url",default=None,type=str,help="url used to set up distributed training",)
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("-f","--exp_file",default=None,type=str,help="pls input your expriment description file",)
    parser.add_argument("--fp16",dest="fp16",default=False,action="store_true",help="Adopting mix precision evaluating.",)
    parser.add_argument("--fuse",dest="fuse",default=False,action="store_true",help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt",dest="trt",default=False,action="store_true",help="Using TensorRT model for testing.",)
    parser.add_argument("--test",dest="test",default=False,action="store_true",help="Evaluating on test-dev set.",)
    parser.add_argument("--speed",dest="speed",default=False,action="store_true",help="speed test only.",)
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER,)
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
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names



def train_epoch(det_model,motion_model,dataloader,optimizer,scheduler,loss_func,epoch,loss_func2 = None,fp16=False):
    torch.set_grad_enabled(True)

    motion_model.train()
    total_loss = 0 
    total_iter = 0
    pbar = tqdm(dataloader,ncols=0,dynamic_ncols=False)

    det_model.eval()
    
    loss_funcs = [loss_func]
    if loss_func2 is not None:
        loss_funcs.append(loss_func2)
    print(loss_funcs)
    for i,data in enumerate(pbar):
        img1 = data["data1"].cuda(non_blocking=True)
        img2 = data["data2"].cuda(non_blocking=True)
        flow = data["flow"].cuda(non_blocking=True)

        if fp16:
            img1,img2,flow = img1.half(),img2.half(),flow.half()
        # is_obj_center = data["is_obj_center"].cuda()
        with torch.no_grad():
            (f1c1,f1c2,f1c3),(f2c1,f2c2,f2c3) = det_model.backbone(img1),det_model.backbone(img2)

        # ipdb.set_trace()
        
        loss2 = 0
        loss3 = 0
        loss = 0
        for _loss_func in loss_funcs:
            if isinstance(motion_model,MotionMamba):
                pred1,pred2,pred3 = motion_model((f1c1.clone(),f1c2.clone(),f1c3.clone()),
                                    (f2c1.clone(),f2c2.clone(),f2c3.clone()))
                flow2 = F.interpolate(flow,scale_factor = 0.5,mode='bilinear')*0.5
                flow1 = F.interpolate(flow2,scale_factor = 0.5,mode='bilinear')*0.5
                loss += _loss_func(pred1,flow1)*0.25 + _loss_func(pred2,flow2)*0.5 + _loss_func(pred3,flow)
            else:
                flow_estim = motion_model((f1c1.clone(),f1c2.clone(),f1c3.clone()),
                                    (f2c1.clone(),f2c2.clone(),f2c3.clone()))
                loss += _loss_func(flow_estim,flow)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss.item())
        loss_val = loss.item()
        total_loss += loss_val
        total_iter += 1
        mean_loss = total_loss / total_iter

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        pbar.set_description(f"epoch {epoch} ,lr {lr:.6f}, loss: {loss_val:.4f}({mean_loss:.4f})")
    scheduler.step()


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting, ")

    is_distributed = num_gpu > 1
    # set environment variables for distributed training
    cudnn.benchmark = True
    rank = args.local_rank
    # rank = get_local_rank()
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



    motion_model = MotionMamba().cuda()
    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)

    _evaluator = MotionEvaluator
    logger.info(f"===============================evaluator:{_evaluator}")
    evaluator = _evaluator(args=args,dataloader=val_loader,img_size=exp.test_size,
        confthre=exp.test_conf,nmsthre=exp.nmsthre,num_classes=exp.num_classes,flow_model=motion_model)
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
        assert (not args.fuse and not is_distributed and args.batch_size == 1), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None


    roots = ["/home/ymf/datas/visdrone/VisDrone2019-MOT-train/sequences",]
    snap_to_load = None
    
    if args.fp16:
        motion_model = motion_model.half()
        model = model.half()
    
    if snap_to_load is not None:
        print("load from ",snap_to_load)
        motion_model.load_state_dict(torch.load(snap_to_load))    

    
    
    


    txt_root = "/home/ymf/datas/visdrone/VisDrone2019-MOT-train/annotations"
    batch_size = 12

    test_root = "/home/ymf/datas/visdrone/VisDrone2019-MOT-test-dev/sequences"
    dataset = MotionDataset(roots,txt_root,(1088,608),flow_foldername="offset_1_flow_emd_gma_76x136")
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=6,pin_memory=True,shuffle=True)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    gamma = 0.98
    loss_func = mse_loss
    loss_func2 = l1_loss

    lr = 1e-4

    optimizer = torch.optim.AdamW(motion_model.parameters(),betas=(0.5,0.99),lr=lr,weight_decay=1e-2)
    # optimizer = torch.optim.SGD(motion_model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    
    max_epoch = 100

    best_mota = 0
    best_idf1 = 0
    save_dir = "./motion/weights"
    expname = "MotionMamba_softshrink_multiscale_mse_l1"

    os.makedirs(save_dir,exist_ok=True)
    exp_save_dir = os.path.join(save_dir,expname)
    os.makedirs(exp_save_dir,exist_ok=True)

    # for n,p in model.named_parameters():
    #     p.requires_grad = False

    log_path = os.path.join(exp_save_dir,"log.txt")
    logf = open(log_path,"w",buffering=1)

    best_mota_idf1 = 0


    for epoch in range(max_epoch):
        train_epoch(model,motion_model,dataloader,optimizer,scheduler,loss_func,epoch,loss_func2,args.fp16)
        
        motion_model = motion_model.eval()
        mota,idf1 = evaluator.evaluate(model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder)
        
        mota_idf1 = mota + idf1
        if idf1 > best_idf1:
            # best_mota_idf1 = mota_idf1
            best_mota = mota
            best_idf1 = idf1
            torch.save(motion_model.state_dict(),os.path.join(exp_save_dir,f"best_mota_{best_mota:.4f}_idf1_{best_idf1:.4f}.pth"))

        log_info = f"epoch {epoch} mota:{mota:.4f},idf1:{idf1:.4f}, best mota: {best_mota:.4f} idf1:{best_idf1:.4f}\n"
        print(log_info)
        logf.write(log_info)
        logf.flush()

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
