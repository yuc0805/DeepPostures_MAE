pip install transformers

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--remark moca-shallow-random-init \
--blr 1e-3 \
--weight_decay 1e-3 \
--model shallow-moca \
--epochs 40 \
--warmup_epochs 8 \
--batch_size 4 \
--accum_iter 8 \
--subset_ratio 1.0 \
--pos_weight 1.0 \
--use_data_aug 1 \
--use_pos_embed \
--patch_emb 'sundial'

# chmod +x script/moca_shallow_iwatch_new.sh
# ./script/moca_shallow_iwatch_new.sh
