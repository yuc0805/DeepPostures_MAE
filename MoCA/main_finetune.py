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
from config import FT_DATASET_CONFIG
import torch
import torch.backends.cudnn as cudnn
import wandb
from util.datasets import iWatch_HDf5, data_aug,collate_fn
import timm

from timm.models.layers import trunc_normal_
# from timm.data.mixup import Mixup
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
#from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models_vit import *

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', type=int, default=100, 
                        help='Input size "')
    parser.add_argument('--patch_size', type=int, default=5, 
                        help='Patch size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')


    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='unsyncmask_checkpoint-200.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--ds_name', default='iwatch', type=str)
    parser.add_argument('--data_path', default='/niddk-data-central/iWatch/pre_processed_seg/W', type=str, 
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int, # changed
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/niddk-data-central/leo_workspace/MoCA_result/FT/ckpt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/niddk-data-central/leo_workspace/MoCA_result/FT/log'',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', # changed
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
    
    parser.add_argument('--remark', default='Debug',help='additional training info')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    #device = torch.device('cpu') # changed - forced cpu

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True 

    if args.ds_name == 'iwatch':
        dataset_train = iWatch_HDf5(args.data_path, set_type='train', transform=data_aug)
        dataset_val = iWatch_HDf5(args.data_path, set_type='val', transform=None)
    else:
        raise NotImplementedError('The specified dataset is not implemented.')

    print('Using dataset',args.ds_name)
    print("Number of Training Samples:", len(dataset_train))
    print("Number of Testing Samples:", len(dataset_val))

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

    if args.log_dir is not None and not args.eval and global_rank == 0:  
        wandb.login(key='32b6f9d5c415964d38bfbe33c6d5c407f7c19743')
        log_writer = wandb.init(
            project='MoCA-iWatch-FT',  # Specify your project
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
        collate_fn=collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    print(f"Using in_chans={args.in_chans}") 
    model = VisionTransformer(
        img_size = args.input_size,patch_size=[1, int(args.patch_size)], 
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
        qkv_bias=True,in_chans=1, num_classes=args.nb_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),)

    print(model)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model,orig_size=(3,20), # FIXME: can also be [6,20] if using both HIP and Wrist
                              new_size=(args.input_size[0],int(args.input_size[1]//args.patch_size)))

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False) # changed from strict=False
        print(msg)

        if False:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed: #changed - hashed out
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    optimizer = create_optimizer_v2(
        model_without_ddp,
        opt='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,# default: 0 
        betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # if mixup_fn is not None:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler) 

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_metric = {'epoch':0, 'acc1':0.0, 'bal_acc':0.0, 'f1':0.0}
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer,  epoch, loss_scaler,
            args.clip_grad, mixup_fn=None,
            log_writer=log_writer,
            args=args, device=device,
        )
        if args.output_dir and (epoch + 1 == args.epochs): 
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(args,data_loader_val, model, device)
        print(f"Balanced Accuracy of the network on the {len(dataset_val)} test images: {test_stats['bal_acc']:.5f} and F1 score of {test_stats['f1']:.5f}%")

        # save the best epoch
        if max_accuracy < test_stats["bal_acc"]:
            max_accuracy = test_stats["bal_acc"]

            best_metric['epoch'] = epoch
            best_metric['bal_acc'] = test_stats['bal_acc']
            best_metric['acc1'] = test_stats['acc1']
            best_metric['f1'] = test_stats['f1']

            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="best")

        print(f'Max Balanced accuracy: {max_accuracy:.2f}%')


        if log_writer is not None:
            log_writer.log({'perf/test_acc1': test_stats['acc1'], 
                            'perf/bal_acc': test_stats['bal_acc'],
                            'perf/f1': test_stats['f1'],
                            'perf/test_loss': test_stats['loss'], 
                            'epoch': epoch})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if log_writer is not None:
        log_writer.log({f"best_epoch_{k}": v for k, v in best_metric.items()})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return max_accuracy


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    initial_timestamp = datetime.datetime.now()

    initial_timestamp = datetime.datetime.now()
        
    args.in_chans = 1
    args.nb_classes = FT_DATASET_CONFIG[args.ds_name]['nb_classes']
    args.blr = FT_DATASET_CONFIG[args.ds_name]["blr"]
        
    args.batch_size = FT_DATASET_CONFIG[args.ds_name]["bs"]
    args.input_size = FT_DATASET_CONFIG[args.ds_name]["input_size"]
    args.weight_decay = FT_DATASET_CONFIG[args.ds_name]["weight_decay"] if 'weight_decay' in FT_DATASET_CONFIG[args.ds_name] else args.weight_decay
    args.remark = args.remark + f'FT_blr_{args.blr}_bs_{args.batch_size}_wd_{args.weight_decay}'
    print(f'Start Training: {args.remark}')

    args.log_dir = os.path.join(args.log_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    args.output_dir = os.path.join(args.output_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)


'''

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2  \
    -m main_finetune \
    --ds_name capture24_4 \
    --finetune "/home/jovyan/persistent-data/leo/output_dir/moca_aug_checkpoint-200.pth" \
    --remark MoCA_200

python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/2971(p200_10_alt_0.0005_both)/checkpoint-19.pth" \
--remark MoCA_20pth


python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/leo/output_dir/moca_aug_checkpoint-200.pth" \
--remark MoCA

python -m main_finetune \
--ds_name imwsha \
--finetune "/home/jovyan/persistent-data/leo/output_dir/syncmask_8_checkpoint-200.pth" \
--remark SyncMask_200


python -m main_finetune \
--ds_name imwsha \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_2025-03-21_21-07/maxcut_checkpoint-100.pth" \
--remark maxcut

python -m main_finetune \
--ds_name ucihar_7 \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/covariance_raw_split_mask_2025-03-16_16-16/covariance_raw_split_mask_checkpoint-200.pth" \
--remark covariance_raw_split_mask

python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/covariance_raw_split_mask_2025-03-16_16-16/covariance_raw_split_mask_checkpoint-200.pth" \
--remark covariance_raw_split_mask

python -m main_finetune \
--ds_name wisdm \
--finetune "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/28533(p200_10_syn_0.0005_both)/checkpoint-19.pth" \
--remark SyncMask_20


python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/661169(p200_10_alt_0.0005)/checkpoint-3999.pth" \
--remark MoCA_noAug

python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/MAE_Accelerometer/experiments/185(p200_10_syn_0.0005)/checkpoint-3999.pth" \
--remark SyncMask_NoAug

python -m main_finetune \
--ds_name imwsha \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_2025-03-21_21-07/maxcut_checkpoint-200.pth" \
--remark maxcut_mask_200

python -m main_finetune \
--ds_name wisdm \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_2025-03-21_21-07/maxcut_checkpoint-300.pth" \
--remark maxcut_mask_300

python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_2025-03-21_21-07/maxcut_checkpoint-300.pth" \
--remark random_init

python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/feat_maxcut_100iter_init_MoCA_20_2025-04-08_01-47/feat_maxcut_100iter_init_MoCA_20_checkpoint-200.pth" \
--remark feat_maxcut_100iter_init_MoCA_20

python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/maxcut_500iter_2025-04-04_00-51/maxcut_500iter_checkpoint-200.pth" \
--remark raw_maxcut_500iter


python -m main_finetune \
--ds_name ucihar_7 \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/feat_maxcut_100iter_init_scratch_2025-04-10_06-26/feat_maxcut_100iter_init_scratch_checkpoint-200.pth" \
--remark feat_maxcut_100iter_init_scratch

CUDA_VISIBLE_DEVICES=1 \
python -m main_finetune \
--ds_name oppo \
--finetune "/home/jovyan/persistent-data/leo/optim_mask/ckpt/feat_maxcut_100iter_init_scratch_2025-04-10_06-26/feat_maxcut_100iter_init_scratch_checkpoint-200.pth" \
--remark feat_maxcut_100iter_init_scratch
'''