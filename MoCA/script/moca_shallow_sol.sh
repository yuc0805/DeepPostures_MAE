pip install transformers

torchrun --nproc_per_node=4 -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/SOL/PASOS/train/SOL_10hz" \
--remark moca-shallow \
--checkpoint "/niddk-data-central/leo_workspace/MoCA_result/ckpt/SOLps_5_mask_0.5_bs_12_blr_None_epoch_100/2025-08-22_03-32/checkpoint-99.pth" \
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
