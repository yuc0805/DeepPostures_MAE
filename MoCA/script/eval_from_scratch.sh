torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark CHAP-RandomInit  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 1.0 \
--pos_weight 2.8232 \
--use_data_aug 1 

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark CHAP-RandomInit  \
--blr 1e-3 \
--model CNNBiLSTMModel \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 32 \
--weight_decay 1e-3 \
--subset_ratio 1.0 \
--pos_weight 2.7953 \
--use_data_aug 1 


# mocca-shallow-from scratch
torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark shallow-moca-randomInit  \
--model 'shallow-moca' \
--blr 1e-3 \
--weight_decay 1e-3 \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--accum_iter 4 \
--num_attn_layer 2 \
--pos_weight=2.7953 \
--use_data_aug 1

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--remark shallow-moca-randomInit  \
--model 'shallow-moca' \
--blr 1e-3 \
--weight_decay 1e-3 \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 8 \
--accum_iter 4 \
--num_attn_layer 2 \
--pos_weight=2.8232 \
--use_data_aug 1

# chmod +x script/eval_from_scratch.sh
# ./script/eval_from_scratch.sh