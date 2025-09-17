

# python -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
# --model vit-small \
# --eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_1.0_blr_0.001_bs_64_input_size_4200/2025-09-11_19-42/checkpoint-best.pth" \
# --remark vit-small-hip \
# --patch_emb sundial \
# --batch_size 512 \
# --use_data_aug 0 \
# --make_prediction \
# --prediction_dir "/niddk-data-central/leo_workspace/iWatch-Validation/H" 

# python -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
# --model vit-small \
# --eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_1.0_blr_0.001_bs_64_input_size_4200/2025-09-11_16-19/checkpoint-best.pth" \
# --remark vit-small-wrist \
# --patch_emb sundial \
# --batch_size 512 \
# --use_data_aug 0 \
# --make_prediction \
# --prediction_dir "/niddk-data-central/leo_workspace/iWatch-Validation/W" 


# python -m main_finetune_long \
# --ds_name iwatch \
# --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
# --model vit-small \
# --eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_1.0_blr_0.001_bs_64_input_size_4200/2025-09-11_16-19/checkpoint-best.pth" \
# --remark vit-small-wrist \
# --patch_emb sundial \
# --batch_size 512 \
# --use_data_aug 0 \
# --make_prediction \
# --prediction_dir "/niddk-data-central/leo_workspace/iWatch-Validation/W" 


# limited label
python -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--model vit-small \
--eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_0.5_blr_0.001_bs_64_input_size_4200/2025-09-12_22-15/checkpoint-best.pth" \
--remark vit-small-hip \
--patch_emb sundial \
--batch_size 512 \
--use_data_aug 0

python -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--model vit-small \
--eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_0.1_blr_0.001_bs_64_input_size_4200/2025-09-13_15-17/checkpoint-best.pth" \
--remark vit-small-hip \
--patch_emb sundial \
--batch_size 512 \
--use_data_aug 0



python -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H" \
--model vit-small \
--eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_0.01_blr_0.001_bs_64_input_size_4200/2025-09-12_23-53/checkpoint-best.pth" \
--remark vit-small-hip \
--patch_emb sundial \
--batch_size 512 \
--use_data_aug 0


# Wrist
python -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--model vit-small \
--eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_0.5_blr_0.001_bs_64_input_size_4200/2025-09-12_19-38/checkpoint-best.pth" \
--remark vit-small-wrist \
--patch_emb sundial \
--batch_size 512 \
--use_data_aug 0


python -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--model vit-small \
--eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_0.1_blr_0.001_bs_64_input_size_4200/2025-09-12_21-32/checkpoint-best.pth" \
--remark vit-small-wrist \
--patch_emb sundial \
--batch_size 512 \
--use_data_aug 0


python -m main_finetune_long \
--ds_name iwatch \
--data_path "/niddk-data-central/iWatch/pre_processed_long_seg/W" \
--model vit-small \
--eval "/niddk-data-central/leo_workspace/MoCA_result/LP/ckpt/vit-smallset_0.01_blr_0.001_bs_64_input_size_4200/2025-09-12_22-27/checkpoint-best.pth" \
--remark vit-small-wrist \
--patch_emb sundial \
--batch_size 512 \
--use_data_aug 0