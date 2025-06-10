#!/bin/bash
# Print the current directory
echo "Current directory: $(pwd)"

# List all files in the current directory
echo "Files:"
ls -lah

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Wristps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-30/checkpoint-49.pth" \
--remark shallow-moca-ft  \
--model 'shallow-moca' \
--accum_iter 32 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--subset_ratio 0.1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Wristps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-30/checkpoint-49.pth" \
--remark shallow-moca-ft  \
--model 'shallow-moca' \
--accum_iter 32 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--subset_ratio 0.5 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Wristps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-30/checkpoint-49.pth" \
--remark shallow-moca-ft  \
--model 'shallow-moca' \
--accum_iter 32 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--subset_ratio 0.01 


torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Hipps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-23/checkpoint-49.pth" \
--remark shallow-moca-ft  \
--model 'shallow-moca' \
--accum_iter 32 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--subset_ratio 0.1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Hipps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-23/checkpoint-49.pth" \
--remark shallow-moca-ft  \
--model 'shallow-moca' \
--accum_iter 32 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--subset_ratio 0.5 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/iWatch-Hipps_5_mask_0.75_bs_512_blr_None_epoch_50/2025-05-05_01-23/checkpoint-49.pth" \
--remark shallow-moca-ft  \
--model 'shallow-moca' \
--num_attn_layer 2 \
--accum_iter 32 \
--pos_weight=2.8232 \
--subset_ratio 0.01 






##
# chmod +x script/moca_shallow.sh
# ./script/moca_shallow.sh