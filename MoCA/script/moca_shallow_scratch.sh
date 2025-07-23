torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/SOL/PASOS/train/SOL_10hz" \
--remark shallow-moca-scratch  \
--model shallow-moca \
--blr 1e-2 \
--weight_decay 5e-2 \
--epochs 40 \
--warmup_epochs 4 \
--batch_size 8 \
--accum_iter 2 \
--num_attn_layer 2 \
--use_pos_embed \
--pos_weight=1.0 \
--use_data_aug 1 \
--subset_ratio 1.0 

#   chmod +x script/moca_shallow_scratch.sh
#   ./script/moca_shallow_scratch.sh
