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
import os
import time
from pathlib import Path
import copy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
# from timm.scheduler.cosine_lr import CosineLRScheduler
import seaborn as sns
import matplotlib.pyplot as plt

import timm
from config import FT_LONG_DATASET_CONFIG
from util.datasets import data_aug, iWatch 
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from timm.optim import create_optimizer_v2
from util.pos_embed import interpolate_pos_embed
import util.lr_decay as lrd  # for optimizer
import models_vit
from models_mae import LinearProbeModel
from engine_finetune_long import train_one_epoch, evaluate
from util.loss import BinaryFocalLoss
from models_mae import AttentionProbeModel

import pickle
import sys
# if os.path.exists('/DeepPostures_MAE/MSSE_2021_pt'):
#     sys.path.append('/DeepPostures_MAE/MSSE_2021_pt')
# elif os.path.exists('/app/DeepPostures_MAE/MSSE_2021_pt'):
#     sys.path.append('/app/DeepPostures_MAE/MSSE_2021_pt')
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print('Adding path to sys.path:', path)
sys.path.append(path)
from MSSE_2021_pt.commons import get_dataloaders_dist 
import random
from einops import rearrange
from tqdm import tqdm

from MSSE_2021_pt.model import CNNBiLSTMModel,CNNModel,CNNAttentionModel,MoCABiLSTMModel
from MSSE_2021_pt.utils import load_model_weights
from omegaconf import OmegaConf

def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--config', default=None, type=str,
                        help='path to config file (default: None, use default config)')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', type=int, default=None, 
                        help='Input size "')
    parser.add_argument('--patch_size', type=int, default=100, 
                        help='Patch size')

    parser.add_argument('--in_chans', default=None, type=int,  # changed - added
                        help='number of channels')
    parser.add_argument('--remark', default='Debug',type=str,
                        help='model_remark')
    parser.add_argument('--use_data_aug',default=1,type=int)
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--num_attn_layer', type=int, default=1,
                        help='number of attention layers in the AttentionProbeModel')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=None, metavar='LR', # default 1e-2
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--learnable_pos_embed', action='store_true', default=True,
                        help='use learnable position embedding (default: True)')
    parser.add_argument('--no_learnable_pos_embed', action='store_false', dest='learnable_pos_embed',
                        help='disable learnable position embedding')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--pos_weight', type=float, default=1.0, 
                        help='positive weight for BCE loss')
    parser.add_argument('--use_focal_loss', action='store_true',
                    help='Use focal loss instead of BCEWithLogitsLoss')
    parser.add_argument('--subset_ratio',type=float,default=1.0,
                        help='Subset ratio for the dataset, default is 1.0 (use all data)')
    # * Finetuning params
    parser.add_argument('--checkpoint', default='/home/jovyan/persistent-data/MAE_Accelerometer/experiments/661169(p200_10_alt_0.0005)/checkpoint-3999.pth', 
                        type=str,help='model checkpoint for evaluation') 
                        

    # Dataset parameters
    parser.add_argument('--data_path', default='/niddk-data-central/iWatch/pre_processed_seg/W', type=str, # changed
                        help='dataset path')
    
    parser.add_argument('--nb_classes', default=None, type=int, # changed
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/niddk-data-central/leo_workspace/MoCA_result/LP/log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=None, type=str,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--ds_name', default='iwatch', type=str)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    return parser    



def main(args):
    args_dict = vars(args)
    if args.config:
        cfg = OmegaConf.load(args.config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_cfg = misc.flatten_config_dict(cfg_dict)  # flatten the config dict
    else:
        flat_cfg = {}

    combined_config = {**args_dict, **flat_cfg}
    

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False 
    

    print('Using dataset',args.ds_name)
    transform=None
    if args.use_data_aug:
        transform=data_aug

    dataset_train = iWatch(
        set_type='train',
        root=args.data_path,
        transform=transform,
        subset_ratio=args.subset_ratio,)
    dataset_val = iWatch(
        set_type='val',
        root=args.data_path,
        transform=None,)
    print(f"using {args.subset_ratio} of train dataset, {len(dataset_train)} samples")

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False,
    )

    # if args.ds_name == 'iwatch':
    #     with open("/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl", "rb") as f:
    #         split_data = pickle.load(f)

    #     train_subjects = split_data["train"]
    #     valid_subjects = split_data["val"]
        

    #     random.shuffle(train_subjects)
    #     random.shuffle(valid_subjects)

    #     data_loader_train, data_loader_val, _ = get_dataloaders_dist(
    #     pre_processed_dir=args.data_path,
    #     bi_lstm_win_size=42, # chap_adult
    #     batch_size=args.batch_size,
    #     train_subjects=train_subjects,
    #     valid_subjects=valid_subjects,
    #     test_subjects=None,
    #     rank=global_rank,
    #     world_size=num_tasks,
    #     transform=data_aug,)        

    # else:
    #     raise NotImplementedError('The specified dataset is not implemented.')

    if args.log_dir is not None and not args.eval and global_rank == 0:  
        wandb.login(key='32b6f9d5c415964d38bfbe33c6d5c407f7c19743')
        log_writer = wandb.init(
            project='MoCA-Long-iWatch-FT',  # Specify your project
            config= combined_config,
            dir=args.log_dir,
            name=args.remark,)
      
    else:
        log_writer = None

    # TODO: package below into a model factory function
    # CHAP replicate #######
    if args.model == 'CNNBiLSTMModel':
        model = CNNBiLSTMModel(2,42,2)
        
        if os.path.exists("/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth"):
            transfer_learning_model_path = "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth"
        elif os.path.exists("/app/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth"):
            transfer_learning_model_path = "/app/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth"
        else:
            raise FileNotFoundError("CHAP_ALL_ADULTS.pth not found in any known location.")

        msg = load_model_weights(model, transfer_learning_model_path, weights_only=False)
    elif args.model == 'CNNAttentionModel':
        base_model = CNNBiLSTMModel(2,42,2)
        if cfg.model.transfer_learning_model_path:
            msg = load_model_weights(base_model, cfg.model.transfer_learning_model_path, weights_only=False)
            print(msg)
        
        # we only need the CNN extractor
        base_model = base_model.cnn_model

        base_model_hidden_dim = base_model.fc.out_features # 512
        model = CNNAttentionModel(base_model=base_model,
                                          base_model_hidden_dim=base_model_hidden_dim,
                                          num_layer=cfg.model.num_layers,
                                          hidden_dim=cfg.model.hidden_dim,
                                          num_heads=cfg.model.num_heads,
                                          ffn_multiplier=cfg.model.ffn_multiplier,
                                          drop_path_rate=cfg.model.drop_path_rate,
                                          learnable_pos_embed=cfg.model.learnable_pos_embed,)
    elif args.model == 'MoCABiLSTMModel':
        # prepare interaction layer
        base_model = CNNBiLSTMModel(2,42,2) # hidden_size=256*2 = 512
        if cfg.interaction_layer.transfer_learning_model_path:
            msg = load_model_weights(base_model, cfg.interaction_layer.transfer_learning_model_path, weights_only=False)
            print(msg)
        interaction_layer = base_model.bil_lstm

        # prepare feature extractor
        feature_extractor = models_vit.__dict__['vit_base_patch16'](
            img_size=[cfg.feature_extractor.nvar,cfg.feature_extractor.length],  #[3,100]
            patch_size=[1, cfg.feature_extractor.patch_size],  #[1,5]
            num_classes=args.nb_classes, 
            in_chans=1, 
            global_pool=False)
        if cfg.feature_extractor.transfer_learning_model_path:
            print('Loading transfer learning model for interaction layer from', cfg.interaction_layer.transfer_learning_model_path)
            checkpoint = torch.load(cfg.feature_extractor.transfer_learning_model_path,map_location='cpu',weights_only=False)
            checkpoint_model = checkpoint['model']
            decoder_keys = [k for k in checkpoint_model.keys() if 'decoder' in k]
            for key in decoder_keys:
                del checkpoint_model[key]
            msg = feature_extractor.load_state_dict(checkpoint_model, strict=False)
            print('msg')
        feature_extractor.head = nn.Identity()  # remove the head

        # assemble model
        # TODO: Add a assert that make sure feature extractor has same window-size as interaction layer
        model = MoCABiLSTMModel(
            feature_extractor=feature_extractor,
            interaction_layer=interaction_layer,
            num_classes=args.nb_classes,)
    elif args.model == 'shallow-moca':
        base_model = models_vit.__dict__['vit_base_patch16'](
            img_size=[3,100], patch_size=[1, 5], 
            num_classes=args.nb_classes, in_chans=1, 
            global_pool=False)
        
        checkpoint = torch.load(args.checkpoint,map_location='cpu')
        checkpoint_model = checkpoint['model']
        # interpolate_pos_embed(base_model, checkpoint_model,orig_size=(args.in_chans,int(100//args.patch_size)), 
        #                       new_size=(args.input_size[0],int(args.input_size[1]//args.patch_size)))
        #print(checkpoint_model.keys())
        decoder_keys = [k for k in checkpoint_model.keys() if 'decoder' in k]
        for key in decoder_keys:
            del checkpoint_model[key]

        print('shape after interpolate:',checkpoint_model['pos_embed'].shape)
        msg = base_model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        model = AttentionProbeModel(base_model, window_size=42,
                                    num_classes=args.nb_classes,
                                    hidden_dim=768,
                                    num_layer=args.num_attn_layer,
                                    learnable_pos_embed=args.learnable_pos_embed,)
        

    #######################
    else:
        backbone = models_vit.__dict__[args.model](
        img_size=args.input_size, patch_size=[1, int(args.patch_size)], 
        num_classes=args.nb_classes, in_chans=1, 
        global_pool=False,use_cls=False)

        # # load weight
        if not args.eval:
            print('Loading pre-trained checkpoint from',args.checkpoint)
            checkpoint = torch.load(args.checkpoint,map_location='cpu')
            checkpoint_model = checkpoint['model']
            interpolate_pos_embed(backbone, checkpoint_model,orig_size=(3,42),
                                new_size=(args.input_size[0],int(args.input_size[1]//args.patch_size)))
            

            #print(checkpoint_model.keys())
            decoder_keys = [k for k in checkpoint_model.keys() if 'decoder' in k]
            for key in decoder_keys:
                del checkpoint_model[key]

            print('shape after interpolate:',checkpoint_model['pos_embed'].shape)
            msg = backbone.load_state_dict(checkpoint_model, strict=False)
            print(msg)
        else:
            checkpoint = torch.load(args.eval,map_location='cpu')
            checkpoint_model = checkpoint['model']
            print(checkpoint['args'])
            msg = backbone.load_state_dict(checkpoint_model, strict=True)
            backbone.to(device)
            test_stats = evaluate(args,data_loader_val, backbone, device)
            print(f"Balanced Accuracy of the network: {test_stats['bal_acc']:.5f}% and F1 score of {test_stats['f1']:.5f}%")
            exit(0)

        model = LinearProbeModel(backbone, num_classes=args.nb_classes)

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

    if args.model == 'CNNBiLSTMModel':
        optimizer = create_optimizer_v2(
            model_without_ddp,
            opt='adamw',
            lr=args.lr,
            weight_decay=args.weight_decay,) # default: 0 )
    else:
        # add layer decay
        optimizer = create_optimizer_v2(
        model_without_ddp,
        opt='adamw',
        lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,)

    loss_scaler = NativeScaler()

    if args.use_focal_loss:
        criterion = BinaryFocalLoss(alpha=0.25,gamma=2.0, reduction="mean").to(device) # alpha is for balance, gamma is to ignore easy sample
    elif args.nb_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], dtype=torch.float32).to(device))
        
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    #if not args.CHAP:
    # scheduler = CosineLRScheduler(
    # optimizer,
    # t_initial=args.epochs,
    # warmup_t=args.warmup_epochs,
    # warmup_lr_init=args.min_lr,
    # t_in_epochs=True)
    # else:
    #     scheduler = None

    print("criterion = %s" % str(criterion))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_metric = {'epoch': 0, 'acc1': 0.0, 'bal_acc': 0.0, 'f1': 0.0}
    save_buffer = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed: 
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler,
            max_norm=args.clip_grad, 
            log_writer=log_writer,
            args=args, device=device,
        )

        test_stats = evaluate(args, data_loader_val, model, device)
        print(f"Balanced Accuracy of the network on test images: {test_stats['bal_acc']:.5f} and F1 score of {test_stats['f1']:.5f}%")

        if max_accuracy < test_stats["bal_acc"]:
            max_accuracy = test_stats["bal_acc"]

            best_metric['epoch'] = epoch
            best_metric['bal_acc'] = test_stats['bal_acc']
            best_metric['acc1'] = test_stats['acc1']
            best_metric['f1'] = test_stats['f1']
            best_metric['confmat'] = test_stats['confmat']

            # Save best state_dicts
            save_buffer = {
                'model': copy.deepcopy(model.state_dict()),
                'model_without_ddp': copy.deepcopy(model_without_ddp.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'loss_scaler': copy.deepcopy(loss_scaler.state_dict())
            }

        if log_writer is not None:
            log_writer.log({
                'perf/test_acc1': test_stats['acc1'], 
                'perf/bal_acc': test_stats['bal_acc'],
                'perf/f1': test_stats['f1'],
                'perf/test_loss': test_stats['loss'], 
                'epoch': epoch
            })

    # Save final model
    if args.output_dir:
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        if save_buffer is not None:
            print(f"Saving best model from epoch {best_metric['epoch']} with balanced accuracy {best_metric['bal_acc']:.4f}")

            # Reload state dicts into model before saving
            model.load_state_dict(save_buffer['model'])
            model_without_ddp.load_state_dict(save_buffer['model_without_ddp'])
            optimizer.load_state_dict(save_buffer['optimizer'])
            loss_scaler.load_state_dict(save_buffer['loss_scaler'])

            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch='best')

    # Logging confusion matrix
    if log_writer is not None:
        confmat = best_metric['confmat'].cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['sitting', 'non-sitting'], yticklabels=['sitting', 'non-sitting'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        log_writer.log({
            "best_epoch_bal_acc": best_metric['bal_acc'],
            "best_epoch_acc1": best_metric['acc1'],
            "best_epoch_f1": best_metric['f1'],
            "best_epoch": best_metric['epoch'],
            "best_epoch_confmat": wandb.Image(fig)
        })

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return max_accuracy


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    initial_timestamp = datetime.datetime.now()
    
    if args.in_chans is None:
        args.in_chans = FT_LONG_DATASET_CONFIG[args.ds_name]['in_chans']
    if args.nb_classes is None:
        args.nb_classes = FT_LONG_DATASET_CONFIG[args.ds_name]['nb_classes']
    if args.blr is None:
        args.blr = FT_LONG_DATASET_CONFIG[args.ds_name]["blr"]
    if args.batch_size is None:
        args.batch_size = FT_LONG_DATASET_CONFIG[args.ds_name]["bs"]
    if args.input_size is None:
        args.input_size = FT_LONG_DATASET_CONFIG[args.ds_name]["input_size"]
    if args.weight_decay is None:
        args.weight_decay = FT_LONG_DATASET_CONFIG[args.ds_name]["weight_decay"]


    #if args.CHAP:
        # args.lr = 1e-4
        # args.batch_size = 4
        # args.weight_decay = 1e-4
    args.remark = args.remark + f'set_{args.subset_ratio}_blr_{args.blr}_bs_{args.batch_size}_input_size_{args.input_size}'
    print(f'Start Training: {args.remark}')
    
    args.log_dir = os.path.join(args.log_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    args.output_dir = os.path.join(args.output_dir,args.remark,f'{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)



'''

torchrun --nproc_per_node=4  -m main_finetune_long \
--ds_name iwatch \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Hip-Longps_100_mask_0.75_bs_32_blr_None_epoch_50/2025-05-06_00-25/checkpoint-49.pth" \
--data_path "/niddk-data-central/iWatch/pre_processed_pt/H" \
--pos_weight 2.7953 \
--remark Hip_Long_50epoch

torchrun --nproc_per_node=4  -m main_finetune_long \
--ds_name iwatch \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Wrist-Longps_100_mask_0.75_bs_32_blr_None_epoch_50/2025-05-06_00-22/checkpoint-49.pth" \
--data_path "/niddk-data-central/iWatch/pre_processed_pt/W" \
--remark Wrist_Long_50epoch

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--epochs 50 \
--warmup_epochs 5 \
--batch_size 256 \
--weight_decay 5e-2 \
--subset_ratio 0.1 

torchrun --nproc_per_node=2  -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--pos_weight 2.8232 \
--epochs 50 \
--model CNNAttentionModel \
--config /DeepPostures_MAE/config/eval/CNNAttentionModel.yaml \
--warmup_epochs 10 \
--remark CNNAttentionModel \
--batch_size 128  \
--blr 1e-4 \
--weight_decay 5e-2 \
--layer_decay 0.4 



torchrun --nproc_per_node=4  -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--pos_weight 2.8232 \
--epochs 50 \
--model MoCABiLSTMModel \
--config /DeepPostures_MAE/config/eval/MoCABiLSTMModel.yaml \
--warmup_epochs 10 \
--remark MoCABiLSTMModel \
--batch_size 8 \
--blr 1e-4 \
--weight_decay 5e-2 \
--layer_decay 0.4

'''