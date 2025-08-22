pip install transformers

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/SOL/PASOS/train/SOL_10hz" \
--remark moca-shallow-random-init \
--blr 1e-3 \
--model shallow-moca \
--epochs 20 \
--warmup_epochs 2 \
--batch_size 4 \
--accum_iter 8 \
--weight_decay 5e-2 \
--subset_ratio 1.0 \
--pos_weight 1.0 \
--use_data_aug 1 
