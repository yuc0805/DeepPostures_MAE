import optuna
import torch
import argparse
import time
import os
import json
import datetime
import torch.nn as nn
import numpy as np

import main_finetune as finetune
from config import DATASET_CONFIG
# Import your dataset functions here
# For example:
# from util.datasets import UCIHAR, WISDM, IMWSHA, Oppo

def objective(trial):
    # Define hyperparameters to tune
    DATASET = 'imwsha'  # Example dataset; adjust as needed
    blr = trial.suggest_loguniform('blr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    # Set the arguments to pass to your training routine.
    args = argparse.Namespace(
        batch_size=64,            # Fixed batch size
        epochs=50,                # Number of epochs (you can adjust as needed)
        lr=None,                  # lr is computed based on blr and effective batch size
        blr=blr,
        weight_decay=weight_decay,
        model='vit_large_patch16',
        img_size=DATASET_CONFIG[DATASET]["img_size"],  
        patch_size=DATASET_CONFIG[DATASET]['nb_classes'],
        in_chans=1,
        drop_path=0.1,
        layer_decay=0.75,
        warmup_epochs=5,
        smoothing=0.1,
        num_workers=10,
        pin_mem=True,
        nb_classes=DATASET_CONFIG[DATASET]['nb_classes'],      
        data_path='data/200',      # Adjust to your data path
        output_dir='output',       # Adjust as needed
        log_dir='logs',            # Adjust as needed
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=0,
        ds_name=DATASET,        # Set dataset name; adjust as needed
        dist_url='env://',
        dist_on_itp=False,
        start_epoch=0,
        eval=False,
        dist_eval = False,
        finetune='/home/jovyan/persistent-data/leo/optim_mask/ckpt/feat_maxcut_100iter_init_MoCA_20_2025-04-08_01-47/feat_maxcut_100iter_init_MoCA_20_checkpoint-200.pth',
        accum_iter=1,
        resume='',
        clip_grad=None,
        remark=f'{DATASET}_finetune_lr{blr}_wd{weight_decay}',
        min_lr=1e-8
    )

    # Train and evaluate the model using the chosen hyperparameters.
    max_acc = finetune.main(args)
    # Return the top-1 accuracy as the objective to maximize.
    return max_acc

def optimize():
    study = optuna.create_study(direction='maximize')  # We want to maximize test accuracy.
    study.optimize(objective, n_trials=50)  # Adjust number of trials as needed.
    
    print("Best hyperparameters:", study.best_params)
    print("Best trial accuracy:", study.best_value)

if __name__ == "__main__":
    optimize()
