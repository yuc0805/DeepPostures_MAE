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
        "--data_path", "/niddk-data-central/iWatch/pre_processed_pt/W",
        "--remark", f"Wrist_sweep_blr_{config.blr}_wd_{config.weight_decay}",
        "--blr", str(config.blr),
        "--weight_decay", str(config.weight_decay),
        "--pos_weight", "2.8232",
        # Add other necessary arguments here
    ]

    # Execute the command
    subprocess.run(command)

if __name__ == "__main__":
    main()


# wandb sweep wrist_sweep.yaml