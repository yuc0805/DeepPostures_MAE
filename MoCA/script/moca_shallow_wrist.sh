#!/bin/bash
# Print the current directory
echo "Current directory: $(pwd)"

# List all files in the current directory
echo "Files:"
ls -lah

# Transfering SOL weight to this.
torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/SOLps_5_mask_0.5_bs_12_blr_None_epoch_100/2025-07-25_20-16/checkpoint-14.pth" \ 
--remark SOL_mask50_14epoch  \
--model 'shallow-moca' \
--blr 1e-3 \
--weight_decay 1e-3 \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--accum_iter 4 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--subset_ratio 1.0 


torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/SOLps_5_mask_0.75_bs_12_blr_None_epoch_100/2025-07-27_11-47/checkpoint-14.pth" \ 
--remark SOL_mask75_14epoch  \
--model 'shallow-moca' \
--blr 1e-3 \
--weight_decay 1e-3 \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--accum_iter 4 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--subset_ratio 1.0 


# torchrun --nproc_per_node=4 -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
# --checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Wristps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-30/checkpoint-49.pth" \
# --remark shallow-moca-avg  \
# --model 'shallow-moca' \
# --blr 1e-3 \
# --weight_decay 1e-3 \
# --epochs 40 \
# --warmup_epochs 8 \
# --batch_size 8 \
# --accum_iter 4 \
# --num_attn_layer 2 \
# --pos_weight=2.8232 \
# --subset_ratio 0.1 

# torchrun --nproc_per_node=4 -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
# --checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Wristps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-30/checkpoint-49.pth" \
# --remark shallow-moca-avg  \
# --model 'shallow-moca' \
# --blr 1e-3 \
# --weight_decay 1e-3 \
# --epochs 40 \
# --warmup_epochs 8 \
# --batch_size 8 \
# --accum_iter 4 \
# --num_attn_layer 2 \
# --pos_weight=2.8232 \
# --subset_ratio 0.01 


# torchrun --nproc_per_node=4 -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
# --checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Wristps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-30/checkpoint-49.pth" \
# --remark shallow-moca-avg  \
# --model 'shallow-moca' \
# --blr 1e-3 \
# --weight_decay 1e-3 \
# --epochs 40 \
# --warmup_epochs 8 \
# --batch_size 8 \
# --accum_iter 4 \
# --num_attn_layer 2 \
# --pos_weight=2.8232 \
# --subset_ratio 0.5 



##
# chmod +x script/moca_shallow_wrist.sh
# ./script/moca_shallow_wrist.sh