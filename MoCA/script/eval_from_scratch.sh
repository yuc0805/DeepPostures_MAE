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
