# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.datasets import UCIHAR,WISDM,IMWSHA,Oppo
import timm
from util.misc import get_next_run_number
from config import DATASET_CONFIG

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed

import util.lr_decay as lrd # changed - added for optimizer
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

#import models_vit
import torch.nn as nn

from engine_finetune import train_one_epoch, evaluate
from torchvision.models import vit_b_16 

# helper function
def parse_list(input_string):
    # Strip brackets and split the string into a list of integers
    return [int(x) for x in input_string.strip('[]').split(',')]


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_tiny_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Leo: Can not pass in list like this                     
    # parser.add_argument('--input_size', default=None, type=list,  # changed
    #                     help='images input size')
    # parser.add_argument('--patch_size', default=None, type=list,  # changed - added
    #                     help='images patch size')

    parser.add_argument('--input_size', type=int, default=200, 
                        help='Input size "')
    parser.add_argument('--patch_size', type=int, default=20, 
                        help='Patch size')
    parser.add_argument('--fintune_mask_ratio', type=int, default=0, 
                        help='fintune_mask_ratio')
    parser.add_argument('--ds_name', default='ucihar_7', type=str)
    parser.add_argument('--patch_num', default=10, type=int,  # changed - added
                        help='number of patches')
    parser.add_argument('--in_chans', default=6, type=int,  # changed - added
                        help='number of channels')
    parser.add_argument('--remark', default='model_mae_linprob',
                        help='model_remark')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-2, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--layer_decay', type=float, default=0,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    

    # * Finetuning params
    # parser.add_argument('--finetune', default='',
    #                     help='finetune from checkpoint')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--checkpoint', default='/home/jovyan/persistent-data/leo/output_dir/mae_pretrain_vit_base.pth',  # for beit, use /home/jovyan/persistent-data/leo/beit_checkpoint-3999.pth
                        type=str,help='model checkpoint for evaluation')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/MAE_Accelerometer/data/200', type=str, # changed
                        help='dataset path')
    parser.add_argument('--alt', action='store_true',
                        help='using [n, c, l, 1] format instead') # changed - added
    parser.set_defaults(alt=False) # changed - added
    parser.add_argument('--nb_classes', default=7, type=int, # changed
                        help='number of the classification types')
    parser.add_argument('--normalization', action='store_true',
                        help='train and test data set normalization') # changed - added
    parser.set_defaults(normalization=False) # changed - added

    parser.add_argument('--output_dir', default='/home/jovyan/persistent-data/leo/downstream/ckpt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/home/jovyan/persistent-data/leo/downstream/log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser



# to make training engine happy
class LinearProbWrapper(nn.Module):
    def __init__(self,model,):
        
        super().__init__()
        # init backbone
        self.backbone = model

    def forward(self, x,mask_ratio=0):
        #print('input shape:',x.shape)
        x_out = self.backbone(x)

        return x_out
    


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False # changed from True  
    

    ## Leo #####################################
    if args.ds_name == 'ucihar_6' or args.ds_name == 'ucihar_7':
        dataset_train = UCIHAR(args.data_path, is_test=False,
                               nb_classes=args.nb_classes,mix_up=False,transform=True)
        dataset_val = UCIHAR(args.data_path,is_test=True,
                             nb_classes=args.nb_classes,mix_up=False,transform=True)
    elif args.ds_name == 'wisdm':
        dataset_train = WISDM(data_path=args.data_path,is_test=False,transform=True)
        dataset_val = WISDM(data_path=args.data_path,is_test=True,transform=True)
    elif args.ds_name == 'imwsha':
        dataset_train = IMWSHA(data_path=args.data_path,is_test=False,transform=True)
        dataset_val = IMWSHA(data_path=args.data_path,is_test=True,transform=True)
    elif args.ds_name == 'oppo':
        dataset_train = Oppo(data_path=args.data_path,is_test=False,transform=True)
        dataset_val = Oppo(data_path=args.data_path,is_test=True,transform=True)
    else:
        raise NotImplementedError('The specified dataset is not implemented.')

    print("Number of Training Samples:", len(dataset_train))
    print("Number of Testing Samples:", len(dataset_val))

    ############################################## data changed - end ################################################

    sampler_train = torch.utils.data.RandomSampler(dataset_train) #changed 
    sampler_test = torch.utils.data.SequentialSampler(dataset_val) #changed 
    
    if args.log_dir is not None and not args.eval:  #changed - global_rank == 0 and ommitted
        path = os.path.join(args.log_dir, 'linprob')
        os.makedirs(path, exist_ok=True)
        run_number = get_next_run_number(path)
        log_run_dir = os.path.join(path, f'run_{args.remark}_{run_number}')
        os.makedirs(log_run_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_run_dir)

    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    

    # Define Linear Prob model
    model = vit_b_16(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'], strict=False)
    model.heads = nn.Linear(768,args.nb_classes,bias=True)
    
    # freeze weight
    if not args.finetune:
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.heads.named_parameters():
            p.requires_grad = True

    model = LinearProbWrapper(model)
    print("Model = %s" % str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of training params : %.2f' % (n_parameters))

    n_total_parameters = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %.2f' % (n_total_parameters))

    model.to(device)
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    print("lr: %.3e" % args.lr)

    optimizer =  torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        )

    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed: 
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler,
            max_norm=None, 
            log_writer=log_writer,
            args=args, device=device,
        )
        if args.output_dir and (epoch + 1 == args.epochs): # changed - added and~ for less frequent dump
            misc.save_model(
                args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # max_accuracy = max(max_accuracy, test_stats["acc1"])
        # print(f'Max accuracy: {max_accuracy:.2f}%')

        # save the best epoch
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="best")
                    
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc3', test_stats['acc3'], epoch) # changed 
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.in_chans = DATASET_CONFIG[args.ds_name]['in_chans']
    args.nb_classes = DATASET_CONFIG[args.ds_name]['nb_classes']
    # if 'lr' in  DATASET_CONFIG[args.ds_name]:
    #     args.lr = DATASET_CONFIG[args.ds_name]['lr']

    args.blr = 1e-2 #DATASET_CONFIG[args.ds_name]["blr"]
    args.batch_size = DATASET_CONFIG[args.ds_name]["bs"]
    
    if args.finetune:
        args.blr = args.blr*0.1
        args.warmup_epochs = 5 # default is 10 for lp
        if 'weight_decay' in DATASET_CONFIG[args.ds_name]:
            args.weight_decay = DATASET_CONFIG[args.ds_name]['weight_decay']
        args.remark = args.remark + '_' + args.ds_name + '_finetune'
    else:
        args.remark = args.remark + '_' + args.ds_name + '_lp'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# python -m vit_baseline --data_path data/200 --remark vit_baseline_finetune --weight_decay 5e-2 --ds_name wisdm  --log_dir ../../../data/leo/downstream/log --output_dir ../../../data/leo/downstream/output_dir

# python -m vit_baseline --data_path data/200 --remark vit_baseline_linprob --ds_name imwsha 

# CUDA_VISIBLE_DEVICES=1 python -m vit_baseline --data_path data/200 --remark beit_ft --ds_name ucihar_7 --finetune

"""

CUDA_VISIBLE_DEVICES=0 \
python -m vit_baseline \
--data_path data/200 \
--remark wearable_beit \
--ds_name oppo \
--checkpoint /home/jovyan/persistent-data/leo/beit_output_dir/wearable_beit/checkpoint-79.pth \
--finetune

python -m vit_baseline \
--data_path data/200 \
--remark beit_3999 \
--ds_name oppo \
--checkpoint /home/jovyan/persistent-data/leo/beit_checkpoint-3999.pth \
--finetune

python -m vit_baseline \
--data_path data/200 \
--remark vit \
--ds_name oppo \
--checkpoint /home/jovyan/persistent-data/leo/output_dir/mae_pretrain_vit_base.pth \
--finetune

"""

