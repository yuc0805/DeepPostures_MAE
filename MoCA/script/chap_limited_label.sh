#!/bin/bash
# Print the current directory
echo "Current directory: $(pwd)"

# List all files in the current directory
echo "Files:"
ls -lah

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP  \
--blr 1e-2 \
--model CNNBiLSTMModel \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--weight_decay 5e-2 \
--subset_ratio 0.1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP  \
--blr 1e-2 \
--model CNNBiLSTMModel \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--weight_decay 5e-2 \
--subset_ratio 0.01 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP  \
--blr 1e-2 \
--model CNNBiLSTMModel \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--weight_decay 5e-2 \
--subset_ratio 0.5


torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark CHAP  \
--blr 1e-2 \
--model CNNBiLSTMModel \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--weight_decay 5e-2 \
--subset_ratio 0.1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark CHAP  \
--blr 1e-2 \
--model CNNBiLSTMModel \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--weight_decay 5e-2 \
--subset_ratio 0.01 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark CHAP  \
--blr 1e-2 \
--model CNNBiLSTMModel \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--weight_decay 5e-2 \
--subset_ratio 0.5

echo "All tasks completed."

##
# chmod +x script/chap_limited_label.sh
# ./script/chap_limited_label.sh