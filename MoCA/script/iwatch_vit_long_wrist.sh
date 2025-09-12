pip install transformers

# torchrun --nproc_per_node=2 -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
# --remark vit-base \
# --blr 1e-3 \
# --weight_decay 1e-3 \
# --layer_decay 1.0 \
# --model vit-base \
# --epochs 40 \
# --warmup_epochs 8 \
# --batch_size 64 \
# --subset_ratio 1.0 \
# --pos_weight 2.8232  \
# --input_size 4200 \
# --patch_size 100 \
# --use_data_aug 1 \
# --use_pos_embed \
# --patch_emb 'sundial' 
# # --use_rope 

torchrun --nproc_per_node=2 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark vit-small \
--blr 1e-3 \
--weight_decay 1e-3 \
--layer_decay 1.0 \
--model vit-small \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 64 \
--subset_ratio 0.5 \
--pos_weight 2.8232  \
--input_size 4200 \
--patch_size 100 \
--use_data_aug 1 \
--use_pos_embed \
--patch_emb 'sundial' 

torchrun --nproc_per_node=2 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark vit-small \
--blr 1e-3 \
--weight_decay 1e-3 \
--layer_decay 1.0 \
--model vit-small \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 64 \
--subset_ratio 0.1 \
--pos_weight 2.8232  \
--input_size 4200 \
--patch_size 100 \
--use_data_aug 1 \
--use_pos_embed \
--patch_emb 'sundial' 

torchrun --nproc_per_node=2 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark vit-small \
--blr 1e-3 \
--weight_decay 1e-3 \
--layer_decay 1.0 \
--model vit-small \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 64 \
--subset_ratio 0.01 \
--pos_weight 2.8232  \
--input_size 4200 \
--patch_size 100 \
--use_data_aug 1 \
--use_pos_embed \
--patch_emb 'sundial' 

# torchrun --nproc_per_node=2 -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
# --remark vit-tiny \
# --blr 1e-3 \
# --weight_decay 1e-3 \
# --layer_decay 1.0 \
# --model vit-tiny \
# --epochs 40 \
# --warmup_epochs 8 \
# --batch_size 64 \
# --subset_ratio 1.0 \
# --pos_weight 2.8232  \
# --input_size 4200 \
# --patch_size 100 \
# --use_data_aug 1 \
# --use_pos_embed \
# --patch_emb 'sundial' 


# chmod +x script/iwatch_vit_long_wrist.sh
# ./script/iwatch_vit_long_wrist.sh
