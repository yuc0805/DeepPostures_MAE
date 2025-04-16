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
from config import DATASET_CONFIG
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from util.datasets import UCIHAR,WISDM,IMWSHA,Oppo,Capture24
import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
#from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models_vit import *

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=None, type=list,  # changed
                        help='images input size')
    parser.add_argument('--patch_size', default=None, type=list,  # changed - added
                        help='images patch size')
    parser.add_argument('--patch_num', default=10, type=int,  # changed - added
                        help='number of patches per one row in the image')
    parser.add_argument('--in_chans', default=6, type=int,  # changed - added
                        help='number of channels')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--alt', action='store_true',
                        help='using [n, c, l, 1] format instead') # changed - added
    parser.set_defaults(alt=False) # changed - added


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

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='unsyncmask_checkpoint-200.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--ds_name', default='ucihar_7', type=str)
    parser.add_argument('--data_path', default='data/200', type=str, # changed
                        help='dataset path')
    parser.add_argument('--nb_classes', default=6, type=int, # changed
                        help='number of the classification types')
    parser.add_argument('--normalization', action='store_true',
                        help='train and test data set normalization') # changed - added
    parser.set_defaults(normalization=False) # changed - added

    parser.add_argument('--output_dir', default='../persistent-data/leo/optim_mask/downstream/ckpt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='../persistent-data/leo/optim_mask/downstream/log',
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
    
    parser.add_argument('--remark', default='moca',help='additional training info')

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
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

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
    
    print(f"Using in_chans={args.in_chans}") 
    model = VisionTransformer(
        img_size = args.img_size,patch_size=[1,20], 
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
        interpolate_pos_embed(model, checkpoint_model,orig_size=(6,10),new_size=(args.img_size[0],int(args.img_size[1]//20)))

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
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
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
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs): # changed - added and~ for less frequent dump
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc3', test_stats['acc3'], epoch) # changed - changed to acc3
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

    return max_accuracy


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.remark = args.remark + '_' + args.ds_name + '_finetune'
    initial_timestamp = datetime.datetime.now()
    args.log_dir = os.path.join(args.log_dir,args.remark,initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    args.output_dir = os.path.join(args.output_dir,args.remark,initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
        
    args.in_chans = 1
    args.nb_classes = DATASET_CONFIG[args.ds_name]['nb_classes']
    args.blr = DATASET_CONFIG[args.ds_name]["blr"]*0.1
    if 'lr' in  DATASET_CONFIG[args.ds_name]:
        args.lr = DATASET_CONFIG[args.ds_name]['lr']
        
    args.batch_size = DATASET_CONFIG[args.ds_name]["bs"]
    args.img_size = DATASET_CONFIG[args.ds_name]["img_size"]
    args.weight_decay = DATASET_CONFIG[args.ds_name]["weight_decay"] if 'weight_decay' in DATASET_CONFIG[args.ds_name] else args.weight_decay
    
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