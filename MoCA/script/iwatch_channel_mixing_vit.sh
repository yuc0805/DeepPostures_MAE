pip install transformers

torchrun --nproc_per_node=2 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark cm-small \
--patch_nvar 3 \
--blr 1e-3 \
--weight_decay 1e-3 \
--layer_decay 1.0 \
--model vit-small \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 64 \
--subset_ratio 1.0 \
--pos_weight 2.7953  \
--input_size 4200 \
--patch_size 100 \
--use_data_aug 1 \
--use_pos_embed 

torchrun --nproc_per_node=2 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark cm-small \
--patch_nvar 3 \
--blr 1e-3 \
--weight_decay 1e-3 \
--layer_decay 1.0 \
--model vit-small \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 64 \
--subset_ratio 1.0 \
--pos_weight 2.8232  \
--input_size 4200 \
--patch_size 100 \
--use_data_aug 1 \
--use_pos_embed 




# chmod +x script/iwatch_channel_mixing_vit.sh
# ./script/iwatch_channel_mixing_vit.sh
