# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


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
        self.train_ann = "train_flow_value.json"
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
        self.no_aug_epochs = 2
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
            data_dir=os.path.join(root,"train"),
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
            data_dir=os.path.join(root,"test"),
            json_file=os.path.join(root,"annotations",self.val_ann),
            img_size=self.test_size,
            name='test',   # change to train when running on training set
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
        from yolox.evaluators import COCOEvaluator,MCMOTEvaluator,MCMOTEval

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
