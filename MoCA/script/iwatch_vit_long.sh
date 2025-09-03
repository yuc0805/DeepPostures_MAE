pip install transformers

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark vit-long \
--blr 1e-3 \
--weight_decay 1e-1 \
--model vit-long \
--epochs 200 \
--warmup_epochs 20 \
--batch_size 128 \
--subset_ratio 1.0 \
--pos_weight 2.7953 \
--input_size 4200 \
--patch_size 100 \
--use_data_aug 1 \
--use_pos_embed \
--patch_emb 'sundial'

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark vit-long \
--blr 1e-3 \
--weight_decay 1e-1 \
--model vit-long \
--epochs 200 \
--warmup_epochs 20 \
--batch_size 128 \
--subset_ratio 1.0 \
--pos_weight 2.8232  \
--input_size 4200 \
--patch_size 100 \
--use_data_aug 1 \
--use_pos_embed \
--patch_emb 'sundial'

# chmod +x script/iwatch_vit_long.sh
# ./script/iwatch_vit_long.sh
