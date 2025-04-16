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
from util.datasets import UCIHAR,WISDM,IMWSHA,Oppo,Capture24
import timm
#from util.misc import get_next_run_number
from config import DATASET_CONFIG

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed

import util.lr_decay as lrd # changed - added for optimizer
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit
import torch.nn as nn

from engine_finetune import train_one_epoch, evaluate

#from leo_model_mae import MaskedAutoencoderViT  
from models_mae import MaskedAutoencoderViT

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
    parser.add_argument('--bert_pos_embed',default=0,type=int,
                        help='using bert_pos_embed')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', type=int, default=200, 
                        help='Input size "')
    parser.add_argument('--patch_size', type=int, default=20, 
                        help='Patch size')

    parser.add_argument('--patch_num', default=10, type=int,  # changed - added
                        help='number of patches')
    parser.add_argument('--in_chans', default=6, type=int,  # changed - added
                        help='number of channels')
    parser.add_argument('--remark', default='model_mae_linprob',type=str,
                        help='model_remark')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-2, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    

    # * Finetuning params
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--checkpoint', default='/home/jovyan/persistent-data/MAE_Accelerometer/experiments/661169(p200_10_alt_0.0005)/checkpoint-3999.pth', 
                        type=str,help='model checkpoint for evaluation')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--fintune_mask_ratio',type=float,default=0,help='mask ratio during finetuning')                   
                        
                        

    # Dataset parameters
    parser.add_argument('--data_path', default='data/200', type=str, # changed
                        help='dataset path')
    parser.add_argument('--alt', action='store_true',
                        help='using [n, c, l, 1] format instead') # changed - added
    parser.set_defaults(alt=False) # changed - added
    parser.add_argument('--nb_classes', default=7, type=int, # changed
                        help='number of the classification types')
    parser.add_argument('--normalization', action='store_true',
                        help='train and test data set normalization') # changed - added
    parser.set_defaults(normalization=False) # changed - added

    parser.add_argument('--output_dir', default='../persistent-data/leo/optim_mask/downstream/ckpt', # ../persistent-data/leo/output_dir/downstream/ckpt
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../persistent-data/leo/optim_mask/downstream/log',
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
    parser.add_argument('--ds_name', default='ucihar_7', type=str)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser




class LinearProb(nn.Module):
    def __init__(self,model,
                 num_classes=7,
                 in_chans=6,bert_pos_embed=False):
        
        super().__init__()
        # init backbone
        self.backbone = model
        embed_size = self.backbone.embed_dim
        self.bert_pos_embed = bert_pos_embed

        if bert_pos_embed:
            self.head = nn.Linear(embed_size,num_classes)
        else:
            #self.head = nn.Linear(in_chans*embed_size,num_classes)
            self.head = nn.Linear(embed_size,num_classes)
        self.embed_size = embed_size
        self.in_chans = in_chans

    def forward(self, x, mask_ratio=0):
        '''Input
        x: bs x nvar x 1 x L

        Output:
        cls: bs x num_class
        '''
        
        #with torch.no_grad():
        z,_,_ = self.backbone.forward_encoder(x,mask_ratio=mask_ratio,
                                              var_mask_ratio=0,time_mask_ratio=0) # only use encoder

        # z: [bs x nvar*(num_p+1) x E]
        bs, _, E = z.shape
        # only use CLS Token
        if self.bert_pos_embed:
            z = z[:,0,:] # bs, E

        else:
            # z = z[:,:,0,:]
            # z = z.reshape(shape=(bs,self.in_chans,-1,E)) 
            # # get cls_token for each variate
            # z = z[:,:,0,:] # bs x nvar x E

            # z = z.reshape(shape=(bs,self.in_chans*E))
            z = z[:,0,:]

        x_out = self.head(z)

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

    cudnn.benchmark = False 
    

    ## Leo #####################################
    if args.ds_name == 'ucihar_6' or args.ds_name == 'ucihar_7':
        dataset_train = UCIHAR(args.data_path, is_test=False,nb_classes=args.nb_classes,mix_up=False)
        dataset_val = UCIHAR(args.data_path,is_test=True,nb_classes=args.nb_classes,mix_up=False)
    elif args.ds_name == 'wisdm':
        dataset_train = WISDM(data_path=args.data_path,is_test=False)
        dataset_val = WISDM(data_path=args.data_path,is_test=True)
    elif args.ds_name == 'imwsha':
        dataset_train = IMWSHA(data_path=args.data_path,is_test=False)
        dataset_val = IMWSHA(data_path=args.data_path,is_test=True)
    elif args.ds_name == 'oppo':
        dataset_train = Oppo(data_path=args.data_path,is_test=False)
        dataset_val = Oppo(data_path=args.data_path,is_test=True)
    elif args.ds_name == 'capture24_4' or args.ds_name == 'capture24_10':
        dataset_train = Capture24(data_path=args.data_path,is_test=False,nb_classes=args.nb_classes)
        dataset_val = Capture24(data_path=args.data_path,is_test=True,nb_classes=args.nb_classes)
    else:
        raise NotImplementedError('The specified dataset is not implemented.')

    print('Using dataset',args.ds_name)
    print("Number of Training Samples:", len(dataset_train))
    print("Number of Testing Samples:", len(dataset_val))

    ############################################## data changed - end ################################################

    if args.ds_name in ['capture24_4','capture24_10']:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.log_dir is not None and not args.eval:  
        log_writer = SummaryWriter(log_dir=args.log_dir)
        print('log_dir: {}'.format(log_writer.log_dir))
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
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
            img_size=args.img_size, patch_size=[1, int(20)],  # changed - added to reflect input_size change
            num_classes=args.nb_classes, in_chans=1, 
            global_pool=False
        )

    # # load weight
    if not args.eval:
        print('Loading pre-trained checkpoint from',args.checkpoint)
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        checkpoint_model = checkpoint['model']
        interpolate_pos_embed(model, checkpoint_model,orig_size=(6,10),new_size=(args.img_size[0],int(args.img_size[1]//20)))

        #print(checkpoint_model.keys())
        decoder_keys = [k for k in checkpoint_model.keys() if 'decoder' in k]
        for key in decoder_keys:
            del checkpoint_model[key]

        print('shape after interpolate:',checkpoint_model['pos_embed'].shape)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    else:
        checkpoint = torch.load('/home/jovyan/MAE_Accelerometer/results/checkpoint-49(linearprobing).pth',map_location='cpu')
        checkpoint_model = checkpoint['model']
        print(checkpoint['args'])
        msg = model.load_state_dict(checkpoint_model, strict=True)
        model.to(device)
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
    #freeze weight
    if not args.finetune:
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    print("Model = %s" % str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of training params : %.2f' % (n_parameters))

    n_total_parameters = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %.2f' % (n_total_parameters))

    model.to(device)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    model_without_ddp = model
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    print("lr: %.3e" % args.lr)

    if args.distributed: #changed - hashed out
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimizer =  torch.optim.AdamW(
    #     model.parameters(),
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     )

    # print(optimizer)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

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
    if 'lr' in  DATASET_CONFIG[args.ds_name]:
        args.lr = DATASET_CONFIG[args.ds_name]['lr']

    args.blr = DATASET_CONFIG[args.ds_name]["blr"]
    args.batch_size = DATASET_CONFIG[args.ds_name]["bs"]
    args.img_size = DATASET_CONFIG[args.ds_name]["img_size"]
    
    if args.finetune:
        args.blr = args.blr*0.1
        args.warmup_epochs = 5 # default is 10 for lp
        if 'weight_decay' in DATASET_CONFIG[args.ds_name]:
            args.weight_decay = DATASET_CONFIG[args.ds_name]['weight_decay']
        args.remark = args.remark + '_' + args.ds_name + '_finetune'
    else:
        args.remark = args.remark + '_' + args.ds_name + '_lp'

    initial_timestamp = datetime.datetime.now()
    
    args.log_dir = os.path.join(args.log_dir,args.remark,initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    args.output_dir = os.path.join(args.output_dir,args.remark,initial_timestamp.strftime("%Y-%m-%d_%H-%M"))

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)



# Linear Probing

# python -m main_linprobe --ds_name ucihar_7 --checkpoint "/home/jovyan/persistent-data/leo/optim_mask/ckpt/covariance_raw_mask_2025-02-21_04-56/covariance_raw_mask_checkpoint-399.pth" --remark covariance_raw_mask

'''

CUDA_VISIBLE_DEVICES=0 \
python -m main_linprobe \
--ds_name capture \
--checkpoint "/home/jovyan/persistent-data/leo/output_dir/moca_aug_checkpoint-200.pth" \
--remark MoCA_200

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29500 \
    -m main_linprobe \
    --ds_name capture24_4 \
    --checkpoint "/home/jovyan/persistent-data/leo/output_dir/moca_aug_checkpoint-200.pth" \
    --remark MoCA_200

CUDA_VISIBLE_DEVICES=0 \
python -m main_linprobe \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/leo/optim_mask/ckpt/covariance_raw_split_mask_2025-03-16_16-16/covariance_raw_split_mask_checkpoint-200.pth" \
--remark covariance_split_mask 


CUDA_VISIBLE_DEVICES=0 \
python -m main_linprobe \
--ds_name imwsha \
--checkpoint "/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_2025-03-21_21-07/maxcut_checkpoint-200.pth" \
--remark maxcut_mask_200

python -m main_linprobe \
--ds_name ucihar_7 \
--checkpoint "/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_2025-04-03_21-29/maxcut_checkpoint-400.pth" \
--remark maxcut_mask_400


CUDA_VISIBLE_DEVICES=0 \
python -m main_linprobe \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/2971(p200_10_alt_0.0005_both)/checkpoint-19.pth" \
--remark MoCA_20pth

CUDA_VISIBLE_DEVICES=0 \
python -m main_linprobe \
--ds_name imwsha \
--checkpoint "/home/jovyan/persistent-data/leo/output_dir/syncmask_8_checkpoint-200.pth" \
--remark SyncMask_200

python -m main_linprobe \
--ds_name imwsha \
--checkpoint "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/28533(p200_10_syn_0.0005_both)/checkpoint-19.pth" \
--remark SyncMask_20

python -m main_linprobe \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/661169(p200_10_alt_0.0005)/checkpoint-3999.pth" \
--remark MoCA_noAug

python -m main_linprobe \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/185(p200_10_syn_0.0005)/checkpoint-3999.pth" \
--remark SyncMask_noAug

python -m main_linprobe \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/185(p200_10_syn_0.0005)/checkpoint-3999.pth" \
--remark random_init

python -m main_linprobe \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/leo/optim_mask/ckpt/feat_maxcut_100iter_init_MoCA_20_2025-04-08_01-47/feat_maxcut_100iter_init_MoCA_20_checkpoint-200.pth" \
--remark feat_maxcut_100iter_init_MoCA_20

python -m main_linprobe \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_500iter_2025-04-04_00-51/maxcut_500iter_checkpoint-200.pth" \
--remark raw_maxcut_500iter


python -m main_linprobe \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/leo/optim_mask/ckpt/feat_maxcut_100iter_init_scratch_2025-04-10_06-26/feat_maxcut_100iter_init_scratch_checkpoint-200.pth" \
--remark feat_maxcut_100iter_init_scratch

'''