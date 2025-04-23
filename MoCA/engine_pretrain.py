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
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")

    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        #print('input sample shape: ',samples.shape)
        loss, _, _ = model(samples, mask_ratio=args.mask_ratio,
                           masking_scheme=args.masking_scheme)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, skipping this batch".format(loss_value))
            optimizer.zero_grad()
            continue

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
            

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)  


        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.log({
                'train_loss': loss_value_reduce,
                'lr': lr
            }, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_masked_series(mask, predict_series, target_series,
                       variate_labels=None, title=None):
    """
    Plots predicted vs. original series and highlights masked patches,
    with manual margins to avoid any overlap.
    """
    # convert tensors to numpy
    mask_np   = mask.cpu().numpy().squeeze()
    pred_np   = predict_series.cpu().numpy().squeeze()
    target_np = target_series.cpu().numpy().squeeze()

    # replace infinities and NaNs
    pred_np   = np.nan_to_num(pred_np,
                              nan=0.0,
                              posinf=np.nanmax(pred_np),
                              neginf=np.nanmin(pred_np))
    target_np = np.nan_to_num(target_np,
                              nan=0.0,
                              posinf=np.nanmax(target_np),
                              neginf=np.nanmin(target_np))

    # set up figure with extra height
    fig, axs = plt.subplots(1, 3,
                            figsize=(15, 6),
                            sharey=True)
    fig.patch.set_facecolor('white')

    # leave room for title and legend
    fig.subplots_adjust(left=0.05,
                        right=0.95,
                        top=0.90,
                        bottom=0.20,
                        wspace=0.3)

    if variate_labels is None:
        variate_labels = ["Acc X", "Acc Y", "Acc Z"]

    # shared y‑limits
    y_min = min(pred_np.min(), target_np.min())
    y_max = max(pred_np.max(), target_np.max())

    # compute width of each masked patch
    num_patches = mask_np.shape[1]
    patch_width = pred_np.shape[1] // num_patches

    for i, ax in enumerate(axs):
        ax.set_facecolor('white')
        ax.plot(pred_np[i],   label="Predicted", linewidth=2)
        ax.plot(target_np[i], label="Original",  linewidth=2)

        # highlight masked spans
        for j, m in enumerate(mask_np[i]):
            if m > 0:
                start, end = j * patch_width, (j + 1) * patch_width
                ax.axvspan(start, end, ymin=0, ymax=1,
                           color='gray', alpha=0.3)

        ax.set_ylim(y_min, y_max)
        ax.set_title(variate_labels[i], fontsize=14)
        ax.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.6)

    # shared axis labels
    fig.text(0.5, 0.10, "Time Step", ha='center', fontsize=12)
    fig.text(0.07, 0.5, "Signal Value", va='center',
             rotation='vertical', fontsize=12)

    # legend below the plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.05),
               ncol=2,
               frameon=False,
               fontsize=11)

    # figure title
    if title:
        fig.suptitle(title, fontsize=16, y=0.96)

    return fig
