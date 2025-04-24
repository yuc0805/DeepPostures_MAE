# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import f1_score, balanced_accuracy_score

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 1,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, device = torch.device,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples) #,mask_ratio=args.fintune_mask_ratio) # bs x num_classes
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        acc1, _ = accuracy(outputs, targets, topk=(1, 3))
        # additional metrics: f1, balanced_acc ##############
        preds = outputs.argmax(dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        f1 = f1_score(targets_np, preds, average='weighted')
        bal_acc = balanced_accuracy_score(targets_np, preds)
        #################################################
        batch_size = samples.shape[0]
    
        metric_logger.meters['train_acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['f1'].update(f1, n=batch_size)
        metric_logger.meters['bal_acc'].update(bal_acc, n=batch_size)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize() 

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)



        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.log(
                {'loss': loss_value_reduce, 
                 'lr': max_lr, 
                 'train acc1': acc1,
                 'train bal_acc':bal_acc,
                 'train f1':f1,
                 }, step=epoch_1000x)


            # log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('lr', max_lr, epoch_1000x)
            # log_writer.add_scalar('train acc1', acc1, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        
        #output = model(images,mask_ratio=0)
        output = model(images)
        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))     

        preds = output.argmax(dim=1).cpu().numpy()
        targets_np = target.cpu().numpy()
        f1 = f1_score(targets_np, preds, average='weighted')
        bal_acc = balanced_accuracy_score(targets_np, preds)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['f1'].update(f1, n=batch_size)
        metric_logger.meters['bal_acc'].update(bal_acc, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} bal_acc {bal_acc.global_avg:.3f} f1 {f1.global_avg:.3f} loss {losses.global_avg:.3f}'  
          .format(top1=metric_logger.acc1, bal_acc=metric_logger.bal_acc, f1=metric_logger.f1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}