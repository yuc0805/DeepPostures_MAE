#!/bin/bash
# Print the current directory
echo "Current directory: $(pwd)"

# List all files in the current directory
echo "Files:"
ls -lah

torchrun --nproc_per_node=4 -m main_attnprobe \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark wrist_50epoch  \
--model vit_base_patch16 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--subset_ratio 0.1 


torchrun --nproc_per_node=4 -m main_attnprobe \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark hip_50epoch  \
--model vit_base_patch16 \
--num_attn_layer 2 \
--pos_weight=2.7953 \
--subset_ratio 0.1 


echo "All tasks completed."

##
# chmod +x script/moca_shallow.sh
# ./script/moca_shallow.sh