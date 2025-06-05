import wandb
import subprocess
import os

def main():
    run = wandb.init()
    config = run.config

    # Construct the command to run the training script with torchrun
    command = [
        "torchrun",
        "--nproc_per_node=4",
        "main_finetune_long.py",
        "--ds_name", "iwatch",
        "--data_path", "/niddk-data-central/iWatch/pre_processed_long_seg/W",
        "--remark", f"Wrist_sweep_blr_{config.blr}_wd_{config.weight_decay}",
        "--blr", str(config.blr),
        "--weight_decay", str(config.weight_decay),
        "--pos_weight", str(config.pos_weight),
        "--batch_size", str(config.batch_size),
        "--epochs", "40",
        "--warmup_epochs", str(config.warmup_epochs),
        "--model","CNNBiLSTMModel",
        # Add other necessary arguments here
    ]

    # Execute the command
    subprocess.run(command)

if __name__ == "__main__":
    main()


# wandb sweep sweep/wrist_sweep.yaml
# wandb agent leo085/DeepPostures_MAE-MoCA/bizjzc3j