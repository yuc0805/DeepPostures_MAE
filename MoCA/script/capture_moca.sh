#!/bin/bash
# Print the current directory
echo "Current directory: $(pwd)"

# List all files in the current directory
echo "Files:"
ls -lah
 

 torchrun --nproc_per_node=2  -m main_linprobe \
--ds_name iwatch \
--checkpoint "/niddk-data-central/leo_workspace/Capture24-randRandom_100iter_init_checkpoint-200.pth" \
--data_path "/niddk-data-central/iWatch/pre_processed_seg/W" \
--in_chans 3 \
--input_size 300 \
--target_sr 30 \
--remark Capture_MoCA200


torchrun --nproc_per_node=2  -m main_linprobe \
--ds_name iwatch \
--checkpoint "/niddk-data-central/leo_workspace/Capture24-randRandom_100iter_init_checkpoint-200.pth" \
--data_path "/niddk-data-central/iWatch/pre_processed_seg/H" \
--in_chans 3 \
--input_size 300 \
--target_sr 30 \
--remark Capture_MoCA200