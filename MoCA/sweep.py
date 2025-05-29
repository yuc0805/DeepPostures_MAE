import os
import argparse
import wandb
from pathlib import Path
from main_finetune_long import main, get_args_parser
from config import DATASET_CONFIG

# Sweep configuration for Bayesian optimization including batch_size
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'bal_acc', 'goal': 'maximize'},
    'parameters': {
        'blr': {'min': 1e-5, 'max': 1e-2},
        'weight_decay': {'min': 0.0, 'max': 1e-1},
    }
}


def train_sweep():
    # parse fixed args and base dirs from command line
    parser = get_args_parser()
    args = parser.parse_args()

    # start a new run with swept params and base log dir
    run = wandb.init(
        config=sweep_config['parameters'],
        project=f"{args.remark}_sweep",
        reinit=True,
        dir=args.log_dir,
        name=None,
    )
    config = run.config

    # override args with sweep values and dataset settings
    args.batch_size = config.batch_size
    args.blr        = config.blr
    # args.weight_decay = config.get('weight_decay', args.weight_decay)
    args.in_chans   = 3
    args.nb_classes = 2
    args.img_size   = [3,100]
    args.log_dir = '/niddk-data-central/leo_workspace/CHAP/sweep/logs'
    args.output_dir = '/niddk-data-central/leo_workspace/CHAP/sweep/output_dir'
    # set up per-run log and output folders
    run_id = run.id
    args.log_dir    = os.path.join(args.log_dir,    run_id)
    args.output_dir = os.path.join(args.output_dir, run_id)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # call the main training function
    main(args)


if __name__ == '__main__':
    # create the sweep and launch agents
    sweep_id = wandb.sweep(sweep_config, project='inform-MoCA-LP-sweep')
    wandb.agent(sweep_id, function=train_sweep, count=20)


"""
python -m lp_hp_sweep \
--ds_name oppo \
--checkpoint "/home/jovyan/persistent-data/leo/output_dir/moca_aug_checkpoint-200.pth" \
--remark MoCA_200

CUDA_VISIBLE_DEVICES=1 python -m lp_hp_sweep \
--ds_name wisdm \
--checkpoint "/home/jovyan/persistent-data/leo/output_dir/moca_aug_checkpoint-200.pth" \
--remark MoCA_200



"""