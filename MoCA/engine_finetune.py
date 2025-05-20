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
from torchmetrics.classification import MulticlassRecall, MulticlassSpecificity,MulticlassF1Score
from torchmetrics import ConfusionMatrix

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
        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            targets = targets.float()


        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples).squeeze() # bs x num_classes or (bs, )
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if args.nb_classes == 2:
            preds = torch.round(torch.sigmoid(outputs))
            acc1 = (preds == targets).float().mean()
        else:
            acc1, _ = accuracy(outputs, targets, topk=(1, 3))

        batch_size = samples.shape[0]
        metric_logger.meters['train_acc1'].update(acc1.item(), n=batch_size)

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
                #  'train bal_acc':bal_acc,
                #  'train f1':f1,
                 }, step=epoch_1000x)


            # log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('lr', max_lr, epoch_1000x)
            # log_writer.add_scalar('train acc1', acc1, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args,data_loader, model, device):
    if args.nb_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=args.nb_classes).to(device)
    recall_metric = MulticlassRecall(args.nb_classes,
                            average='weighted',
                            zero_division=0).to(device)
    
    specificity_metric  = MulticlassSpecificity(num_classes=2,
                            average='weighted',
                            zero_division=0).to(device)
    f1_metric = MulticlassF1Score(num_classes=args.nb_classes,
                            average='weighted',
                            zero_division=0).to(device)
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            target = target.float()
            
        output = model(images).squeeze()
        print('output shape:', output.shape)
        print('target shape:', target.shape)
        loss = criterion(output, target)

        if args.nb_classes == 2:
            preds = torch.round(torch.sigmoid(output))
            acc1 = (preds == target).float().mean()
        else:
            acc1, _ = accuracy(output, target, topk=(1, 2))     
            preds = output.argmax(dim=1)
            
        recall_metric.update(preds, target)
        specificity_metric.update(preds, target)
        f1_metric.update(preds, target)
        confmat_metric.update(preds, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # Compute metrics
    recall_tm = recall_metric.compute().item()
    specificity_tm = specificity_metric.compute().item()
    confmat = confmat_metric.compute()
    bal_acc = 100 * (recall_tm + specificity_tm) / 2
    f1 = 100 * f1_metric.compute().item()

    print('* Acc@1 {top1.global_avg:.5f} bal_acc {bal_acc:.5f} f1 {f1:.5f} loss {losses.global_avg:.3f}'  
          .format(top1=metric_logger.acc1, bal_acc=bal_acc, f1=f1, losses=metric_logger.loss))

    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    eval_stats['f1']=f1
    eval_stats['bal_acc']=bal_acc
    eval_stats['confmat']=confmat

    return eval_stats