# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir
from loguru import logger

root = "/home/ymf/datas/visdrone"


means = (0.574859,0.5284268,0.48245227)
stds = (0.20307621,0.16383478,0.16327296)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 10
        self.class_numbers = [234305,94396,40255,505301,46940,30498,28338,41349,9653,102819]
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "trainval_flow_boxover_emd_gma.json"
        #self.train_ann = "train.json"
        self.val_ann = "test-dev.json"    # change to train.json when running on training set
        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        self.random_size = (18, 32)
        self.max_epoch = 10
        self.print_interval = 20
        self.eval_interval = 1
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 1
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        
        self.seed = 1002
    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTFlowDataset,
            MOTDataset,
            TrainTransformVisdrone,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetectionVisdrone,
        )
        dataset = MOTFlowDataset(
        #dataset = MOTDataset(
            # data_dir=os.path.join(root,"train"),
            data_dir=root,
            json_file=os.path.join(root,"annotations",self.train_ann),
            name='',
            img_size=self.input_size,
            preproc=TrainTransformVisdrone(
                rgb_means=means,
                std=stds,
                max_labels=500,
            ),
        )

        dataset = MosaicDetectionVisdrone(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformVisdrone(
                rgb_means=means,
                std=stds,
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False,interested_seqs=None):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(root,"VisDrone2019-MOT-test-dev/sequences"),
            json_file=os.path.join(root,"annotations",self.val_ann),
            img_size=self.test_size,
            # name='test',   # change to train when running on training set
            name = '',
            preproc=ValTransform(
                rgb_means=means,
                std=stds,
            )
            #,interested_seqs=None
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator,MCMOTEvaluator,MCMOTEval,MCMOTEvalBias

        batch_size = 1
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = MCMOTEval(
            args=self.args,
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev
        )
        # evaluator = MCMOTEvaluator(
        #     args=self.args,
        #     dataloader=val_loader,
        #     img_size=self.test_size,
        #     confthre=self.test_conf,
        #     nmsthre=self.nmsthre,
        #     num_classes=self.num_classes,
        #     testdev=testdev
        # )
        # evaluator = COCOEvaluator(
        #     dataloader=val_loader,
        #     img_size=self.test_size,
        #     confthre=self.test_conf,
        #     nmsthre=self.nmsthre,
        #     num_classes=self.num_classes,
        #     testdev=testdev,
        # )

        return evaluator
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            #SGD

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})

            
            #AdamW

            # optimizer = torch.optim.AdamW(pg0, lr=lr)
            # optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay})  # add pg1 with weight_decay
            # optimizer.add_param_group({"params": pg2})

            logger.info(f"optimizer is:{optimizer}")
            self.optimizer = optimizer
        return self.optimizer
    

    def get_model(self,args):
        from yolox.models import YOLOPAFPN, YOLOX2, YOLOXHead,YOLOX_Mamba

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        # if getattr(self, "model", None) is None:
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels,args=args)
        # self.model = YOLOX_Mamba(backbone, head)
        self.model = YOLOX_Mamba(backbone, head,width = self.width)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        print("get model-------------==================")
        print(self.model)
        return self.model