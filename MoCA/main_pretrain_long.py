# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import wandb
import h5py


import timm
import torch.nn as nn
from functools import partial
from timm.optim import create_optimizer_v2

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models_mae import MaskedAutoencoderViT
from engine_pretrain_long import train_one_epoch,plot_masked_series
import random
import pickle

import sys
sys.path.append('/DeepPostures_MAE/MSSE-2021-pt')
from commons import get_dataloaders_dist,data_aug
import random
from einops import rearrange
from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus' )
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_freq', default=5, type=int,
                        help='save frequency, default 5 epochs')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--input_size', default=4200, type=int,  # changed
                        help='images input size')
    parser.add_argument('--patch_size', default=100, type=int,  # changed
                        help='images patch size')
    parser.add_argument('--nvar', default=3, type=int,  # changed - added
                        help='number of channels')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--masking_scheme', default='None', type=str,
                        help='Masking scheme')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/niddk-data-central/iWatch/pre_processed_seg/H', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='/niddk-data-central/leo_workspace/MoCA_result/ckpt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/niddk-data-central/leo_workspace/MoCA_result/log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', 
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
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

    parser.add_argument('--remark', default='',
                        help='model_remark')
    
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    np.random.seed(seed)

    cudnn.benchmark =  True

    #### Build dataset ####
    with open("/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl", "rb") as f:
        split_data = pickle.load(f)

        train_subjects = split_data["train"]
        
    random.shuffle(train_subjects)

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

    data_loader_train, _, _ = get_dataloaders_dist(
    pre_processed_dir=args.data_path,
    bi_lstm_win_size=42, # chap_adult
    batch_size=args.batch_size,
    train_subjects=train_subjects,
    valid_subjects=None,
    test_subjects=None,
    rank=global_rank,
    world_size=num_tasks,
    transform=data_aug,)
    ###########################

    #print('training sample: ',len(dataset_train))

    if global_rank == 0 and args.log_dir is not None:
        wandb.login(key='32b6f9d5c415964d38bfbe33c6d5c407f7c19743')   
        log_writer = wandb.init(
            project='iWatch-MoCA-Long',  # Specify your project
            config= vars(args),
            dir=args.log_dir,
            name=args.remark,)

    else:
        log_writer = None
    
    model = MaskedAutoencoderViT(img_size=[args.nvar,args.input_size],patch_size=[1,args.patch_size],
                                in_chans=1,embed_dim=768, depth=12, num_heads=12,
                                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # new version timm
    optimizer = create_optimizer_v2(
    model_without_ddp,
    opt='adamw',
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.95)
    )

    print(optimizer)
    loss_scaler = NativeScaler()
    scheduler = CosineLRScheduler(
    optimizer,
    t_initial=args.epochs,
    warmup_t=args.warmup_epochs,
    warmup_lr_init=args.min_lr,
    t_in_epochs=True)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
   
    # fix a sample for plot ###########
    x, y = next(iter(data_loader_train))
    tmp_sample = x[3:4] # 1, 42, 100, 3
    tmp_sample = rearrange(tmp_sample, 'b w l c -> b 1 c (w l)') # 1,1,3,4200
    #tmp_label = 'sitting' if y[3] == 0 else 'non-sitting'
    print(f"tmp_sample shape: {tmp_sample.shape}")
    #print(f"tmp_label: {tmp_label}")
    ############################################

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        # if args.distributed:
        #    data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

        # Avoid NCCL Comm error
        if torch.distributed.is_initialized():
            print('Watiing for all processes to finish')
            torch.distributed.barrier()
        
        
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            print('Saving checkpoint')
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            ### save a reconstruction plot in tensorboard
            if misc.is_main_process():
                print('plotting')
                # Create a matplotlib figure
                with torch.no_grad():
                    tmp_sample = tmp_sample.to(device, non_blocking=True)
                    tmp_sample = tmp_sample.detach().clone().to(device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        tmp_loss, tmp_pred, tmp_mask = model_without_ddp(tmp_sample, 
                                                                         mask_ratio=args.mask_ratio,
                                                                         masking_scheme = args.masking_scheme)

                tmp_pred = model_without_ddp.unpatchify(tmp_pred) # bs, 1, nvar, L
                
                tmp_mask = tmp_mask.reshape(shape=(args.nvar,int(model_without_ddp.num_patches//args.nvar)))
                fig = plot_masked_series(tmp_mask.cpu(),tmp_pred.cpu(),tmp_sample.squeeze().cpu(),
                                            title=f'epoch_{epoch}_loss = {tmp_loss}')

                # Log the figure to TensorBoard
                log_writer.log({f"Reconstruction": wandb.Image(fig)})
                plt.close(fig)
                
                torch.cuda.empty_cache()
            
        scheduler.step(epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    initial_timestamp = datetime.datetime.now()
    args.remark = f'{args.remark}ps_{args.patch_size}_mask_{args.mask_ratio}_bs_{args.batch_size}_blr_{args.lr}_epoch_{args.epochs}'
    print(f'Start Training: {args.remark}')

    args.log_dir = os.path.join(args.log_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    args.output_dir = os.path.join(args.output_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)



'''

torchrun --nproc_per_node=4 main_pretrain_long.py \
--data_path /niddk-data-central/iWatch/pre_processed_pt/H \
--batch_size 32 \
--world_size 4 \
--epochs 50 \
--warmup_epochs 5 \
--remark iWatch-Hip-Long


torchrun --nproc_per_node=4 main_pretrain.py \
--data_path /niddk-data-central/iWatch/pre_processed_seg/W \
--batch_size 256 \
--world_size 4 \
--epochs 100 \
--warmup_epochs 10 \
--remark iWatch-Wrist
'''
