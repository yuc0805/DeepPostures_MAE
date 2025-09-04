pip install transformers

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark vit-long \
--blr 5e-4 \
--weight_decay 5e-2 \
--layer_decay 1.0 \
--model vit-long \
--epochs 200 \
--warmup_epochs 20 \
--batch_size 64 \
--subset_ratio 1.0 \
--pos_weight 2.7953 \
--input_size 4200 \
--patch_size 100 \
--use_data_aug 1 \
--use_pos_embed \
--patch_emb 'sundial'

# CHANNEL-MIXING
# torchrun --nproc_per_node=4 -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
# --remark vit-long \
# --blr 5e-4 \
# --weight_decay 5e-2 \
# --layer_decay 1.0 \
# --model vit-long2 \
# --epochs 200 \
# --warmup_epochs 20 \
# --batch_size 256 \
# --subset_ratio 1.0 \
# --pos_weight 2.7953 \
# --input_size 4200 \
# --patch_size 100 \
# --use_data_aug 1 \
# --use_pos_embed 



# chmod +x script/iwatch_vit_long.sh
# ./script/iwatch_vit_long.sh
