torchrun --nproc_per_node=8 main_pretrain.py \
--data_path /niddk-data-central/SOL/PASOS/train/SOL_10hz \
--batch_size 12 \
--world_size 8 \
--epochs 100 \
--warmup_epochs 10 \
--std_sampling \
--remark SOL \
--save_freq 2
