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

            # log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('lr', lr, epoch_1000x)
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_masked_series(mask, predict_series, target_series, variate_labels=None, title=None):
    """
    Plots the unpatched series, original series, and highlights the masked regions.
    
    Parameters:
    - mask (torch.Tensor): The mask tensor of shape (6, 50).
    - predict_series (torch.Tensor): The predicted series tensor of shape (6, 200).
    - target_series (torch.Tensor): The original series tensor of shape (6, 200).
    - variate_labels (list, optional): List of labels for each variate. Default is None.
    - title (str, optional): Title for the entire plot. Default is None.
    
    Returns:
    - fig: Matplotlib figure object.
    """

    # Convert tensors to numpy arrays
    mask_np = mask.cpu().numpy().squeeze()
    predict_series_np = predict_series.cpu().numpy().squeeze()
    target_series_np = target_series.cpu().numpy().squeeze()

    # Ensure numerical stability
    if not np.isfinite(predict_series_np).all():
        predict_series_np = np.nan_to_num(predict_series_np, nan=0.0, 
                                          posinf=np.max(predict_series_np[np.isfinite(predict_series_np)]), 
                                          neginf=np.min(predict_series_np[np.isfinite(predict_series_np)]))
    if not np.isfinite(target_series_np).all():
        target_series_np = np.nan_to_num(target_series_np, nan=0.0, 
                                         posinf=np.max(target_series_np[np.isfinite(target_series_np)]), 
                                         neginf=np.min(target_series_np[np.isfinite(target_series_np)]))

    # Ensure mask is valid
    if mask_np.shape[1] == 0 or predict_series_np.shape[1] == 0 or target_series_np.shape[1] == 0:
        raise ValueError("Input tensors must have non-zero dimensions.")

    # Patch width
    num_patches = mask_np.shape[1]  # Number of patches (should be 50)
    patch_width = predict_series_np.shape[1] // num_patches

    # Create subplots: 2 rows, 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
    axs = axs.flatten()

    # Default variate labels if not provided
    if variate_labels is None:
        variate_labels = ["Acc_X", "Acc_Y", "Acc_Z"]

    # Compute global min and max
    global_min = min(np.min(predict_series_np), np.min(target_series_np))
    global_max = max(np.max(predict_series_np), np.max(target_series_np))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = -1, 1  # Set reasonable defaults

    # Plot each of the 6 variates
    for i in range(3):
        axs[i].plot(predict_series_np[i, :], label='Predicted Series', color='cornflowerblue', linewidth=3)
        axs[i].plot(target_series_np[i, :], label='Original Series', color='tomato', linewidth=3)

        # Set the y-axis limits
        axs[i].set_ylim([global_min, global_max])

        # Highlight masked regions
        for j, mask_value in enumerate(mask_np[i]):
            if mask_value > 0:
                start, end = j * patch_width, (j + 1) * patch_width
                axs[i].fill_betweenx(
                    [global_min, global_max], 
                    start, end, 
                    color='gray', 
                    alpha=0.5, 
                    label='Masked Area' if j == 0 else ""
                )

        axs[i].set_title(variate_labels[i], fontsize=14)
        axs[i].set_xticks([])  # Remove x-axis ticks
        axs[i].set_yticks([])  # Remove y-axis ticks

    # Set only one legend outside the plot
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)

    # Set the overall title for the plot
    if title:
        fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.subplots_adjust(right=0.8)  # Make space for the legend

    return fig
