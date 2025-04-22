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

import torch
import torch.backends.cudnn as cudnn
import wandb

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.datasets import iWatch, data_aug
from util.misc import get_next_run_number

import timm
import torch.nn as nn
from functools import partial

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

#from leo_model_mae import MaskedAutoencoderViT
from models_mae import MaskedAutoencoderViT

from engine_pretrain import train_one_epoch,plot_masked_series


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
    
    parser.add_argument('--input_size', default=100, type=int,  # changed
                        help='images input size')
    parser.add_argument('--patch_size', default=5, type=int,  # changed
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
    parser.add_argument('--num_workers', default=8, type=int)
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

    dataset_train = iWatch(data_path=args.data_path,
                            set_type='train',
                            transform=data_aug)

    print('training sample: ',len(dataset_train))

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    
    if global_rank == 0 and args.log_dir is not None:
        wandb.login(key='32b6f9d5c415964d38bfbe33c6d5c407f7c19743')   
        log_writer = wandb.init(
            project='iWatch-MoCA',  # Specify your project
            config= vars(args),
            dir=args.log_dir,
            name=args.remark,)

    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
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
    
    # following re: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
   
    # fix a sample for plot ###########
    tmp_sample,_ = next(iter(data_loader_train))  
    tmp_sample = tmp_sample[6:7].to(device)
    ############################################

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
           data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)
        
        
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs): # changed - adjusted save_model frequency
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            ### save a reconstruction plot in tensorboard
            if misc.is_main_process():
                # Create a matplotlib figure
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        tmp_loss, tmp_pred, tmp_mask = model_without_ddp(tmp_sample, 
                                                                         mask_ratio=args.mask_ratio,
                                                                         masking_scheme = args.masking_scheme)

                tmp_pred = model_without_ddp.unpatchify(tmp_pred) # bs, 1, nvar, L
                
                tmp_mask = tmp_mask.reshape(shape=(args.nvar,int(model_without_ddp.num_patches//args.nvar)))
                fig = plot_masked_series(tmp_mask.cpu(),tmp_pred.cpu(),tmp_sample.squeeze().cpu(),
                                            title=f'epoch_{epoch}_loss = {tmp_loss}')

                # Log the figure to TensorBoard
                log_writer.log({f"Reconstruction": wandb.Image(fig)},step=epoch)
                plt.close(fig)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
                        

        if args.output_dir and misc.is_main_process():
            # if log_writer is not None:
            #     log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    initial_timestamp = datetime.datetime.now()
    args.remark = f'ps_{args.patch_size}_mask_{args.mask_ratio}_bs_{args.batch_size}_blr_{args.lr}_epoch_{args.epochs}'
    print(f'Start Training: {args.remark}')

    args.log_dir = os.path.join(args.log_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    args.output_dir = os.path.join(args.output_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)



'''

torchrun --nproc_per_node=4 main_pretrain.py \
--batch_size 512 \
--world_size 4 \
--remark spectral_mask \
--epochs 400 \
--warmup_epochs 40

'''
