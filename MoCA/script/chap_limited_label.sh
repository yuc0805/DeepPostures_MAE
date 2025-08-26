#!/bin/bash
# Print the current directory
echo "Current directory: $(pwd)"

# List all files in the current directory
echo "Files:"
ls -lah


torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--checkpoint "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 0.01 \
--pos_weight 2.8232 \
--use_data_aug 1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--checkpoint "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 0.1 \
--pos_weight 2.8232 \
--use_data_aug 1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--checkpoint "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 0.5 \
--pos_weight 2.8232 \
--use_data_aug 1


torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--checkpoint "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 0.01 \
--pos_weight 2.7953 \
--use_data_aug 1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--checkpoint "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 0.1 \
--pos_weight 2.7953 \
--use_data_aug 1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark CHAP-FT  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--checkpoint "/DeepPostures_MAE/MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth" \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 0.5 \
--pos_weight 2.7953 \
--use_data_aug 1 


echo "All tasks completed."

##
# chmod +x script/chap_limited_label.sh
# ./script/chap_limited_label.sh